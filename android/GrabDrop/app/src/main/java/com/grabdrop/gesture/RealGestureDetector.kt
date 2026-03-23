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

/**
 * Real-time gesture detector using MediaPipe for hand detection and TCN model for gesture classification.
 *
 * Detection flow:
 * 1. IDLE stage: Detect hand presence (any hand gesture triggers wakeup)
 * 2. WAKEUP stage: Use TCN model to classify gestures (grab/release/swipe_up/swipe_down/noise)
 */
class RealGestureDetector(
    private val context: Context,
    private val overlayManager: OverlayManager
) {

    companion object {
        private const val TAG = "RealGesture"
        private const val IDLE_LOG_INTERVAL_FRAMES = 10
        private const val WAKEUP_LOG_INTERVAL_FRAMES = 5
        private const val DEBUG_FRAME_INTERVAL_MS = 30_000L

        // Confidence threshold for TCN model predictions
        private const val CONFIDENCE_THRESHOLD = 0.5f
    }

    private val _events = MutableSharedFlow<GestureEvent>(extraBufferCapacity = 10)
    val events: SharedFlow<GestureEvent> = _events

    enum class Stage { IDLE, WAKEUP }

    @Volatile private var currentStage = Stage.IDLE
    @Volatile private var running = false

    private val lifecycleOwner = ServiceLifecycleOwner()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null

    private var actualImageWidth = 0
    private var actualImageHeight = 0
    private var actualRotation = 0
    private var sensorRotation = -1

    // Detectors
    private var handDetector: HandLandmarkDetector? = null
    private var gestureClassifier: GestureClassifier? = null

    // IDLE stage
    private val idleWindow = ArrayDeque<Boolean>()  // true = hand detected
    private var idleFrameCount = 0

    // WAKEUP stage
    private var wakeupStartTime = 0L
    private var wakeupFrameCount = 0

    // Cooldowns
    @Volatile private var lastFrameTime = 0L
    @Volatile private var lastGestureTime = 0L

    // Stats
    private val totalFrameCount = AtomicLong(0)
    private var cameraFrameCount = AtomicLong(0)
    private var droppedFrameCount = AtomicLong(0)
    private var lastDebugFrameTime = 0L
    private var firstFrameLogged = false

    private var scope: CoroutineScope? = null
    private val isDebug: Boolean get() = ServiceState.isDebug()

    fun start() {
        if (running) return
        running = true

        scope = CoroutineScope(
            Dispatchers.Main + SupervisorJob() +
                    CoroutineExceptionHandler { _, t -> Log.e(TAG, "Coroutine error", t) }
        )

        scope?.launch {
            try {
                addLog("🔧 Initializing gesture detector...")
                handDetector = HandLandmarkDetector(context)
                if (handDetector?.isInitialized == true) {
                    addLog("✅ MediaPipe loaded")
                } else {
                    addLog("❌ MediaPipe FAILED"); return@launch
                }

                gestureClassifier = GestureClassifier(context)
                if (gestureClassifier?.isInitialized == true) {
                    addLog("✅ TCN model loaded")
                } else {
                    addLog("⚠️ TCN model FAILED, will use fallback mode")
                    // Continue without TCN - fallback to basic hand detection
                }

                startCamera()
            } catch (e: Exception) {
                addLog("❌ Start failed: ${e.message}")
            }
        }

        scope?.launch {
            delay(5000)
            while (isActive && running) {
                delay(10_000)
                if (isDebug) {
                    addLog("📊 cam=${cameraFrameCount.get()} proc=${totalFrameCount.get()} " +
                            "drop=${droppedFrameCount.get()} stage=$currentStage")
                }
            }
        }
    }

    fun stop() {
        running = false
        mainHandler.post {
            try { lifecycleOwner.onPause(); lifecycleOwner.onStop(); lifecycleOwner.onDestroy() }
            catch (_: Exception) {}
        }
        try { cameraProvider?.unbindAll() } catch (_: Exception) {}
        try { handDetector?.close() } catch (_: Exception) {}
        try { gestureClassifier?.close() } catch (_: Exception) {}
        handDetector = null
        gestureClassifier = null
        overlayManager.hideWakeupIndicator()
        analysisExecutor.shutdown()
        scope?.cancel(); scope = null
    }

    fun triggerMockRelease(delayMs: Long = 2_500L) {
        scope?.launch { delay(delayMs); if (running) _events.emit(GestureEvent.Release) }
    }

    // --- Camera ---

    private suspend fun startCamera() {
        addLog("📷 Starting front camera...")
        val provider = getCameraProvider()
        cameraProvider = provider

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalysis.setAnalyzer(analysisExecutor) { proxy ->
            cameraFrameCount.incrementAndGet()
            if (!running) { proxy.close(); return@setAnalyzer }
            processFrame(proxy)
        }

        mainHandler.post {
            try {
                lifecycleOwner.onCreate(); lifecycleOwner.onStart(); lifecycleOwner.onResume()
                provider.unbindAll()
                camera = provider.bindToLifecycle(
                    lifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, imageAnalysis
                )
                sensorRotation = camera?.cameraInfo?.sensorRotationDegrees ?: -1
                addLog("📷 Camera bound! sensor=$sensorRotation")
                addLog("👁️ IDLE — scanning at ~${Constants.IDLE_FPS}fps")
            } catch (e: Exception) { addLog("❌ Camera bind failed: ${e.message}") }
        }
    }

    private suspend fun getCameraProvider(): ProcessCameraProvider =
        suspendCoroutine { cont ->
            val f = ProcessCameraProvider.getInstance(context)
            f.addListener({ cont.resume(f.get()) }, ContextCompat.getMainExecutor(context))
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
                droppedFrameCount.incrementAndGet(); imageProxy.close(); return
            }
            lastFrameTime = now
            totalFrameCount.incrementAndGet()

            val detector = handDetector ?: run { imageProxy.close(); return }

            if (!firstFrameLogged) {
                firstFrameLogged = true
                actualImageWidth = imageProxy.width
                actualImageHeight = imageProxy.height
                actualRotation = imageProxy.imageInfo.rotationDegrees
                addLog("📷 First frame: ${actualImageWidth}x${actualImageHeight} rot=$actualRotation")
            }

            val bitmap = imageProxyToBitmap(imageProxy)
            imageProxy.close()
            if (bitmap == null) return

            if (isDebug && now - lastDebugFrameTime > DEBUG_FRAME_INTERVAL_MS) {
                lastDebugFrameTime = now
                try {
                    val copy = bitmap.copy(bitmap.config, false)
                    MediaStoreHelper.saveBitmap(context, copy, "GrabDrop_DEBUG")
                    copy.recycle()
                    addLog("🐛 Debug frame saved")
                } catch (_: Exception) {}
            }

            val detail = detector.detectDetailed(bitmap, now)
            bitmap.recycle()

            when (currentStage) {
                Stage.IDLE -> processIdle(detail)
                Stage.WAKEUP -> processWakeup(detail, now)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame error", e); imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val planes = imageProxy.planes
            if (planes.isEmpty()) return null
            val buffer = planes[0].buffer
            val w = imageProxy.width; val h = imageProxy.height
            val ps = planes[0].pixelStride; val rs = planes[0].rowStride
            val rp = rs - ps * w
            val bw = if (rp > 0 && ps > 0) w + rp / ps else w

            val raw = Bitmap.createBitmap(bw, h, Bitmap.Config.ARGB_8888)
            buffer.rewind(); raw.copyPixelsFromBuffer(buffer)

            val cropped = if (bw != w) {
                Bitmap.createBitmap(raw, 0, 0, w, h).also { if (it !== raw) raw.recycle() }
            } else raw

            val rot = imageProxy.imageInfo.rotationDegrees
            val matrix = Matrix().apply {
                if (rot != 0) postRotate(rot.toFloat())
                val rw = if (rot == 90 || rot == 270) h else w
                postScale(-1f, 1f, rw / 2f, 0f)
            }
            val result = Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
            if (result !== cropped) cropped.recycle()
            result
        } catch (e: Exception) { Log.e(TAG, "Bitmap conv error", e); null }
    }

    // --- IDLE ---

    private fun processIdle(detail: HandLandmarkDetector.DetectionDetail) {
        idleFrameCount++
        val handDetected = detail.handsFound > 0
        idleWindow.addLast(handDetected)
        while (idleWindow.size > Constants.IDLE_WINDOW_SIZE) idleWindow.removeFirst()

        if (isDebug && idleFrameCount % IDLE_LOG_INTERVAL_FRAMES == 0) {
            // Show emoji based on hand state
            val emoji = when {
                !handDetected -> "·"
                detail.state == HandState.PALM -> "🖐"
                detail.state == HandState.FIST -> "✊"
                else -> "✋"
            }
            val windowStr = idleWindow.joinToString("") { if (it) "✋" else "·" }
            val handCount = idleWindow.count { it }
            addLog("👁️ IDLE #$idleFrameCount $emoji | $windowStr | hand=$handCount/${Constants.IDLE_WINDOW_SIZE} | ${detail.summary()}")
        }

        if (idleWindow.size < Constants.IDLE_WINDOW_SIZE) return

        // Wake up when hand is detected in enough frames
        val handDetectedCount = idleWindow.count { it }
        if (handDetectedCount >= Constants.IDLE_TRIGGER_THRESHOLD) {
            enterWakeup()
        }
    }

    // --- WAKEUP ---

    private fun enterWakeup() {
        val now = System.currentTimeMillis()
        if (now - lastGestureTime < Constants.GRAB_COOLDOWN_MS) {
            if (isDebug) addLog("⏳ Wakeup suppressed (cooldown)")
            return
        }

        currentStage = Stage.WAKEUP
        wakeupStartTime = SystemClock.uptimeMillis()
        wakeupFrameCount = 0
        idleWindow.clear()
        idleFrameCount = 0

        // Reset gesture classifier window
        gestureClassifier?.reset()

        addLog("🔔 WAKEUP! Hand detected — using TCN for gesture classification")

        // Show unified palm indicator
        mainHandler.post { overlayManager.showWakeupIndicator("🤚") }
    }

    private fun processWakeup(detail: HandLandmarkDetector.DetectionDetail, now: Long) {
        wakeupFrameCount++

        val elapsed = now - wakeupStartTime

        // Run TCN classification if hand is detected and classifier is available
        if (detail.handsFound > 0 && detail.rawLandmarks != null && gestureClassifier?.isInitialized == true) {
            val result = gestureClassifier?.addFrameAndClassify(detail.rawLandmarks)

            // result is null until we have 30 frames (full window)
            if (result != null && result.confidence >= CONFIDENCE_THRESHOLD) {
                val gesture = result.gesture

                // Only emit event if it's a valid gesture (not noise)
                if (gesture != "noise") {
                    val event = when (gesture) {
                        "grab" -> GestureEvent.Grab
                        "release" -> GestureEvent.Release
                        "swipe_up" -> GestureEvent.SwipeUp
                        "swipe_down" -> GestureEvent.SwipeDown
                        else -> null
                    }

                    if (event != null) {
                        val emoji = when (gesture) {
                            "grab" -> "✊"
                            "release" -> "🖐"
                            "swipe_up" -> "⬆️"
                            "swipe_down" -> "⬇️"
                            else -> "?"
                        }
                        addLog("✅ $emoji ${gesture.uppercase()} detected! conf=${"%.2f".format(result.confidence)} validFrames=${result.validFrames}")

                        lastGestureTime = System.currentTimeMillis()
                        scope?.launch { _events.emit(event) }
                        exitWakeup()
                        return
                    }
                }
            }

            // Log progress
            if (isDebug && wakeupFrameCount % WAKEUP_LOG_INTERVAL_FRAMES == 0) {
                val remaining = Constants.WAKEUP_DURATION_MS - elapsed
                val gestureInfo = if (result != null) {
                    "${result.gesture}(${ "%.2f".format(result.confidence)}) v=${result.validFrames}"
                } else {
                    "collecting (need 15)..."
                }
                addLog("⏱️ WK #$wakeupFrameCount ${"%.1f".format(remaining / 1000.0)}s | $gestureInfo | ${detail.summary()}")
            }
        } else {
            if (isDebug && wakeupFrameCount % WAKEUP_LOG_INTERVAL_FRAMES == 0) {
                val remaining = Constants.WAKEUP_DURATION_MS - elapsed
                val classifierStatus = if (gestureClassifier?.isInitialized != true) "NO_TCN" else "no hand"
                addLog("⏱️ WK #$wakeupFrameCount ${"%.1f".format(remaining / 1000.0)}s | $classifierStatus | ${detail.summary()}")
            }
        }

        // Timeout
        if (elapsed > Constants.WAKEUP_DURATION_MS) {
            if (isDebug) addLog("⌛ WAKEUP timeout")
            else addLog("⌛ Timeout")
            exitWakeup()
        }
    }

    // --- Wakeup Exit ---

    private fun exitWakeup() {
        currentStage = Stage.IDLE
        wakeupFrameCount = 0
        idleWindow.clear()
        idleFrameCount = 0
        gestureClassifier?.reset()
        mainHandler.post { overlayManager.hideWakeupIndicator() }
        addLog("👁️ Back to IDLE")
    }

    private fun addLog(msg: String) {
        val t = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        val fullMsg = "$t $msg"
        Log.d(TAG, fullMsg)
        ServiceState.addEvent(fullMsg)
    }
}
