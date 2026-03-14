// /GrabDrop/app/src/main/java/com/grabdrop/gesture/RealGestureDetector.kt
package com.grabdrop.gesture

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.grabdrop.camera.ServiceLifecycleOwner
import com.grabdrop.capture.MediaStoreHelper
import com.grabdrop.overlay.OverlayManager
import com.grabdrop.service.ServiceState
import com.grabdrop.util.Constants
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

class RealGestureDetector(
    private val context: Context,
    private val overlayManager: OverlayManager
) {

    companion object {
        private const val TAG = "RealGesture"
        private const val IDLE_LOG_INTERVAL_FRAMES = 10
        private const val WAKEUP_LOG_INTERVAL_FRAMES = 5
        // Save a debug frame every N seconds to gallery for visual inspection
        private const val DEBUG_FRAME_INTERVAL_MS = 30_000L
        private const val ENABLE_DEBUG_FRAMES = true
    }

    private val _events = MutableSharedFlow<GestureEvent>(extraBufferCapacity = 10)
    val events: SharedFlow<GestureEvent> = _events

    enum class Stage { IDLE, WAKEUP }

    @Volatile
    private var currentStage = Stage.IDLE

    @Volatile
    private var running = false

    // Camera
    private val lifecycleOwner = ServiceLifecycleOwner()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null

    // Camera info for logging
    private var actualImageWidth = 0
    private var actualImageHeight = 0
    private var actualRotation = 0
    private var sensorRotation = -1

    // MediaPipe
    private var handDetector: HandLandmarkDetector? = null

    // Idle stage state
    private val idleWindow = ArrayDeque<HandState>()

    // Wakeup stage state
    private var wakeupStartTime = 0L
    private var wakeupTriggerState = HandState.NONE
    private var wakeupTargetState = HandState.NONE
    private var consecutiveTargetFrames = 0
    private var wakeupFrames = mutableListOf<HandState>()

    // Throttling
    @Volatile
    private var lastFrameTime = 0L

    // Cooldown
    @Volatile
    private var lastGestureTime = 0L

    // Frame counters
    private val totalFrameCount = AtomicLong(0)
    private var idleFrameCount = 0
    private var wakeupFrameCount = 0
    private var cameraFrameCount = AtomicLong(0)
    private var droppedFrameCount = AtomicLong(0)

    // Debug
    private var lastDebugFrameTime = 0L
    private var firstFrameLogged = false

    private var scope: CoroutineScope? = null

    fun start() {
        if (running) return
        running = true

        scope = CoroutineScope(
            Dispatchers.Main + SupervisorJob() +
                    CoroutineExceptionHandler { _, throwable ->
                        Log.e(TAG, "Coroutine error", throwable)
                        addLog("⚠️ Gesture detector error: ${throwable.message}")
                    }
        )

        scope?.launch {
            try {
                addLog("🔧 Initializing gesture detector...")
                handDetector = HandLandmarkDetector(context)
                if (handDetector?.isInitialized == true) {
                    addLog("✅ MediaPipe HandLandmarker loaded")
                } else {
                    addLog("❌ MediaPipe HandLandmarker FAILED to load")
                    return@launch
                }
                startCamera()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start", e)
                addLog("❌ Gesture detector start failed: ${e.message}")
            }
        }

        // Periodic stats logger
        scope?.launch {
            delay(5000)
            while (isActive && running) {
                delay(10_000)
                val cam = cameraFrameCount.get()
                val total = totalFrameCount.get()
                val dropped = droppedFrameCount.get()
                addLog(
                    "📊 Stats: cam=$cam proc=$total drop=$dropped stage=$currentStage " +
                            "img=${actualImageWidth}x${actualImageHeight} rot=$actualRotation sensor=$sensorRotation"
                )
            }
        }
    }

    fun stop() {
        Log.d(TAG, "Stopping")
        running = false

        mainHandler.post {
            try {
                lifecycleOwner.onPause()
                lifecycleOwner.onStop()
                lifecycleOwner.onDestroy()
            } catch (e: Exception) {
                Log.e(TAG, "Lifecycle stop error", e)
            }
        }

        try { cameraProvider?.unbindAll() } catch (_: Exception) {}
        try { handDetector?.close() } catch (_: Exception) {}
        handDetector = null

        overlayManager.hideWakeupIndicator()

        analysisExecutor.shutdown()
        scope?.cancel()
        scope = null

        addLog("🛑 Gesture detector stopped")
    }

    fun triggerMockRelease(delayMs: Long = 2_500L) {
        scope?.launch {
            delay(delayMs)
            if (running) {
                Log.d(TAG, ">>> Mock RELEASE triggered")
                _events.emit(GestureEvent.Release)
            }
        }
    }

    // --- Camera ---

    private suspend fun startCamera() {
        addLog("📷 Starting front camera...")

        val provider = getCameraProvider()
        cameraProvider = provider

        // Log available cameras
        val hasfront = provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)
        val hasBack = provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)
        addLog("📷 Available cameras: front=$hasfront back=$hasBack")

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy ->
            cameraFrameCount.incrementAndGet()
            if (!running) {
                imageProxy.close()
                return@setAnalyzer
            }
            processFrame(imageProxy)
        }

        mainHandler.post {
            try {
                lifecycleOwner.onCreate()
                lifecycleOwner.onStart()
                lifecycleOwner.onResume()

                provider.unbindAll()
                camera = provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    imageAnalysis
                )

                // Log camera sensor info
                val cameraInfo = camera?.cameraInfo
                sensorRotation = cameraInfo?.sensorRotationDegrees ?: -1
                addLog("📷 Camera bound! sensorRotation=$sensorRotation")
                addLog("📷 Analysis target: 640x480")

                Log.d(TAG, "Camera bound. sensorRotation=$sensorRotation")
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
                addLog("❌ Camera bind failed: ${e.message}")
            }
        }
    }

    private suspend fun getCameraProvider(): ProcessCameraProvider {
        return suspendCoroutine { cont ->
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener(
                { cont.resume(future.get()) },
                ContextCompat.getMainExecutor(context)
            )
        }
    }

    // --- Frame Processing ---

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val now = SystemClock.uptimeMillis()
            val minInterval = when (currentStage) {
                Stage.IDLE -> Constants.IDLE_FRAME_INTERVAL_MS
                Stage.WAKEUP -> Constants.WAKEUP_FRAME_INTERVAL_MS
            }

            if (now - lastFrameTime < minInterval) {
                droppedFrameCount.incrementAndGet()
                imageProxy.close()
                return
            }
            lastFrameTime = now
            totalFrameCount.incrementAndGet()

            val detector = handDetector
            if (detector == null) {
                imageProxy.close()
                return
            }

            // Log first frame details
            val rotation = imageProxy.imageInfo.rotationDegrees
            val width = imageProxy.width
            val height = imageProxy.height

            if (!firstFrameLogged) {
                firstFrameLogged = true
                actualImageWidth = width
                actualImageHeight = height
                actualRotation = rotation

                val planeCount = imageProxy.planes.size
                val pixelStride = if (planeCount > 0) imageProxy.planes[0].pixelStride else -1
                val rowStride = if (planeCount > 0) imageProxy.planes[0].rowStride else -1
                val bufferSize = if (planeCount > 0) imageProxy.planes[0].buffer.remaining() else -1

                val msg = "📷 First frame: ${width}x${height} rot=${rotation}° " +
                        "planes=$planeCount pixelStride=$pixelStride rowStride=$rowStride " +
                        "bufSize=$bufferSize sensorRot=$sensorRotation"
                Log.d(TAG, msg)
                addLog(msg)
            }

            actualImageWidth = width
            actualImageHeight = height
            actualRotation = rotation

            // Convert to bitmap with proper orientation
            val bitmap = imageProxyToBitmap(imageProxy)
            imageProxy.close()

            if (bitmap == null) {
                Log.w(TAG, "Bitmap conversion returned null")
                return
            }

            // Save debug frame periodically
            if (ENABLE_DEBUG_FRAMES && now - lastDebugFrameTime > DEBUG_FRAME_INTERVAL_MS) {
                lastDebugFrameTime = now
                saveDebugFrame(bitmap, rotation)
            }

            val detail = detector.detectDetailed(bitmap, now)
            bitmap.recycle()

            when (currentStage) {
                Stage.IDLE -> processIdle(detail)
                Stage.WAKEUP -> processWakeup(detail, now)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val planes = imageProxy.planes
            if (planes.isEmpty()) return null

            val buffer = planes[0].buffer
            val width = imageProxy.width
            val height = imageProxy.height
            val pixelStride = planes[0].pixelStride
            val rowStride = planes[0].rowStride

            // Handle row padding
            val rowPadding = rowStride - pixelStride * width
            val bitmapWidth = if (rowPadding > 0 && pixelStride > 0) {
                width + rowPadding / pixelStride
            } else {
                width
            }

            val rawBitmap = Bitmap.createBitmap(bitmapWidth, height, Bitmap.Config.ARGB_8888)
            buffer.rewind()
            rawBitmap.copyPixelsFromBuffer(buffer)

            // Crop out padding if present
            val croppedBitmap = if (bitmapWidth != width) {
                Bitmap.createBitmap(rawBitmap, 0, 0, width, height).also {
                    if (it !== rawBitmap) rawBitmap.recycle()
                }
            } else {
                rawBitmap
            }

            // Apply rotation AND horizontal flip for front camera
            val rotation = imageProxy.imageInfo.rotationDegrees
            val needsTransform = rotation != 0 || true  // always flip for front camera

            if (needsTransform) {
                val matrix = Matrix().apply {
                    // First rotate
                    if (rotation != 0) {
                        postRotate(rotation.toFloat())
                    }
                    // Then mirror horizontally (front camera is mirrored)
                    val rotatedWidth = if (rotation == 90 || rotation == 270) height else width
                    postScale(-1f, 1f, rotatedWidth / 2f, 0f)
                }

                val transformed = Bitmap.createBitmap(
                    croppedBitmap, 0, 0,
                    croppedBitmap.width, croppedBitmap.height,
                    matrix, true
                )
                if (transformed !== croppedBitmap) croppedBitmap.recycle()
                transformed
            } else {
                croppedBitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Bitmap conversion error: ${e.message}", e)
            null
        }
    }

    /**
     * Save a debug frame to gallery so you can visually inspect
     * what MediaPipe is actually seeing.
     */
    private fun saveDebugFrame(bitmap: Bitmap, rotation: Int) {
        try {
            // Save a copy — don't interfere with the detection bitmap
            val copy = bitmap.copy(bitmap.config, false)
            val uri = MediaStoreHelper.saveBitmap(
                context, copy, "GrabDrop_DEBUG_rot${rotation}"
            )
            copy.recycle()

            val msg = "🐛 Debug frame saved: ${bitmap.width}x${bitmap.height} rot=$rotation → $uri"
            Log.d(TAG, msg)
            addLog(msg)
        } catch (e: Exception) {
            Log.e(TAG, "Debug frame save failed", e)
        }
    }

    // --- Stage 1: IDLE ---

    private fun processIdle(detail: HandLandmarkDetector.DetectionDetail) {
        idleFrameCount++

        idleWindow.addLast(detail.state)
        while (idleWindow.size > Constants.IDLE_WINDOW_SIZE) {
            idleWindow.removeFirst()
        }

        // Periodic logging
        if (idleFrameCount % IDLE_LOG_INTERVAL_FRAMES == 0) {
            val windowStr = idleWindow.joinToString("") { state ->
                when (state) {
                    HandState.PALM -> "🖐"
                    HandState.FIST -> "✊"
                    HandState.UNKNOWN -> "❓"
                    HandState.NONE -> "·"
                }
            }
            val palmCount = idleWindow.count { it == HandState.PALM }
            val fistCount = idleWindow.count { it == HandState.FIST }
            val noneCount = idleWindow.count { it == HandState.NONE }
            val unknownCount = idleWindow.count { it == HandState.UNKNOWN }

            val logMsg = "👁️ IDLE #$idleFrameCount | $windowStr | " +
                    "P=$palmCount F=$fistCount N=$noneCount U=$unknownCount | " +
                    "det: ${detail.summary()}"
            Log.d(TAG, logMsg)
            addLog(logMsg)
        }

        if (idleWindow.size < Constants.IDLE_WINDOW_SIZE) return

        val palmCount = idleWindow.count { it == HandState.PALM }
        val fistCount = idleWindow.count { it == HandState.FIST }

        when {
            palmCount >= Constants.IDLE_TRIGGER_THRESHOLD -> {
                enterWakeup(triggerState = HandState.PALM, targetState = HandState.FIST)
            }
            fistCount >= Constants.IDLE_TRIGGER_THRESHOLD -> {
                enterWakeup(triggerState = HandState.FIST, targetState = HandState.PALM)
            }
        }
    }

    // --- Stage 2: WAKEUP ---

    private fun enterWakeup(triggerState: HandState, targetState: HandState) {
        val now = System.currentTimeMillis()
        if (now - lastGestureTime < Constants.GRAB_COOLDOWN_MS) {
            Log.d(TAG, "Wakeup suppressed (cooldown)")
            addLog("⏳ Wakeup suppressed — cooldown active")
            return
        }

        Log.d(TAG, ">>> WAKEUP: detected=$triggerState, looking for=$targetState")
        currentStage = Stage.WAKEUP
        wakeupStartTime = SystemClock.uptimeMillis()
        wakeupTriggerState = triggerState
        wakeupTargetState = targetState
        consecutiveTargetFrames = 0
        wakeupFrames.clear()
        wakeupFrameCount = 0
        idleWindow.clear()
        idleFrameCount = 0

        val triggerEmoji = if (triggerState == HandState.PALM) "🖐" else "✊"
        val targetEmoji = if (targetState == HandState.FIST) "✊" else "🖐"
        val motionName = if (targetState == HandState.FIST) "GRAB" else "RELEASE"

        addLog(
            "🔔 WAKEUP! Detected $triggerEmoji — watching 2s for " +
                    "$triggerEmoji→$targetEmoji ($motionName)"
        )

        val indicator = if (triggerState == HandState.FIST) "✊" else "🤚"
        mainHandler.post {
            overlayManager.showWakeupIndicator(indicator)
        }
    }

    private fun processWakeup(detail: HandLandmarkDetector.DetectionDetail, now: Long) {
        wakeupFrameCount++
        wakeupFrames.add(detail.state)

        val elapsed = now - wakeupStartTime
        val remaining = Constants.WAKEUP_DURATION_MS - elapsed

        if (detail.state == wakeupTargetState) {
            consecutiveTargetFrames++
        } else {
            if (consecutiveTargetFrames > 0) {
                Log.d(
                    TAG, "Streak broken at $consecutiveTargetFrames " +
                            "(got ${detail.state}, need $wakeupTargetState)"
                )
            }
            consecutiveTargetFrames = 0
        }

        if (wakeupFrameCount % WAKEUP_LOG_INTERVAL_FRAMES == 0 || consecutiveTargetFrames > 0) {
            val stateEmoji = when (detail.state) {
                HandState.PALM -> "🖐"
                HandState.FIST -> "✊"
                HandState.UNKNOWN -> "❓"
                HandState.NONE -> "·"
            }
            val progressBar = buildProgressBar(
                consecutiveTargetFrames,
                Constants.WAKEUP_CONFIRM_FRAMES
            )

            addLog(
                "⏱️ WK #$wakeupFrameCount ${"%.1f".format(remaining / 1000.0)}s | " +
                        "$stateEmoji str=$consecutiveTargetFrames/${Constants.WAKEUP_CONFIRM_FRAMES} " +
                        "$progressBar | ${detail.summary()}"
            )
        }

        if (elapsed > Constants.WAKEUP_DURATION_MS) {
            val summary = buildWakeupSummary()
            Log.d(TAG, "Wakeup timeout. $summary")
            addLog("⌛ WAKEUP timeout — no gesture. $summary")
            exitWakeup()
            return
        }

        if (consecutiveTargetFrames >= Constants.WAKEUP_CONFIRM_FRAMES) {
            val event = when (wakeupTargetState) {
                HandState.FIST -> {
                    Log.d(TAG, ">>> GRAB confirmed!")
                    addLog("✅ ✊ GRAB CONFIRMED! (palm→fist)")
                    GestureEvent.Grab
                }
                HandState.PALM -> {
                    Log.d(TAG, ">>> RELEASE confirmed!")
                    addLog("✅ 🖐 RELEASE CONFIRMED! (fist→palm)")
                    GestureEvent.Release
                }
                else -> null
            }

            if (event != null) {
                lastGestureTime = System.currentTimeMillis()
                scope?.launch { _events.emit(event) }
            }

            exitWakeup()
        }
    }

    private fun exitWakeup() {
        currentStage = Stage.IDLE
        consecutiveTargetFrames = 0
        wakeupFrames.clear()
        wakeupFrameCount = 0
        idleWindow.clear()
        idleFrameCount = 0

        mainHandler.post { overlayManager.hideWakeupIndicator() }

        addLog("👁️ Back to IDLE — scanning at ~10fps")
        Log.d(TAG, "Back to IDLE")
    }

    private fun buildWakeupSummary(): String {
        val total = wakeupFrames.size
        val palmCount = wakeupFrames.count { it == HandState.PALM }
        val fistCount = wakeupFrames.count { it == HandState.FIST }
        val noneCount = wakeupFrames.count { it == HandState.NONE }
        val unknownCount = wakeupFrames.count { it == HandState.UNKNOWN }

        val timeline = wakeupFrames.takeLast(30).joinToString("") { state ->
            when (state) {
                HandState.PALM -> "🖐"
                HandState.FIST -> "✊"
                HandState.UNKNOWN -> "❓"
                HandState.NONE -> "·"
            }
        }

        return "total=$total 🖐=$palmCount ✊=$fistCount ·=$noneCount ❓=$unknownCount | $timeline"
    }

    private fun buildProgressBar(current: Int, max: Int): String {
        val filled = current.coerceAtMost(max)
        val empty = max - filled
        return "[" + "█".repeat(filled) + "░".repeat(empty) + "]"
    }

    private fun addLog(message: String) {
        val time = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        ServiceState.addEvent("$time $message")
    }
}