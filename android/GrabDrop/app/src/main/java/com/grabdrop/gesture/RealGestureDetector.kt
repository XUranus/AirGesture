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
import com.grabdrop.util.AppSettings
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
import kotlin.math.abs

/**
 * Real-time gesture detector supporting two detection backends:
 *
 * - **Neural Network (TCN)**: Uses a Temporal Convolutional Network via ONNX Runtime
 *   for 5-class classification (grab/release/swipe_up/swipe_down/noise). More accurate
 *   but requires the ONNX model to be bundled.
 *
 * - **Legacy (rule-based)**: Uses hand landmark ratios to classify palm/fist and
 *   frame-to-frame velocity for swipe detection. No extra model needed.
 *
 * The user selects the preferred mode in Settings. If TCN mode is selected but the
 * model fails to load, the detector automatically falls back to legacy mode.
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

        // TCN confidence threshold
        private const val CONFIDENCE_THRESHOLD = 0.5f
    }

    /** Which detection backend is actually active for this session. */
    enum class DetectionMode { NEURAL_NETWORK, LEGACY }

    private val _events = MutableSharedFlow<GestureEvent>(extraBufferCapacity = 10)
    val events: SharedFlow<GestureEvent> = _events

    enum class Stage { IDLE, WAKEUP }

    @Volatile private var currentStage = Stage.IDLE
    @Volatile private var running = false

    /** Resolved at start(); may differ from user preference if model fails to load. */
    @Volatile private var activeMode = DetectionMode.LEGACY

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

    // ── Shared IDLE state ────────────────────────────────────────

    // In legacy mode the window holds HandState; in TCN mode just Boolean (hand detected).
    // We keep both representations so the same processIdle can serve both.
    private val idleWindowStates = ArrayDeque<HandState>()   // legacy
    private val idleWindowHand = ArrayDeque<Boolean>()       // TCN
    private var idleFrameCount = 0

    // ── WAKEUP state (shared) ────────────────────────────────────
    private var wakeupStartTime = 0L
    private var wakeupFrameCount = 0

    // ── Legacy WAKEUP state ──────────────────────────────────────
    private var wakeupTriggerState = HandState.NONE
    private var wakeupTargetState = HandState.NONE
    private var consecutiveTargetFrames = 0
    private var wakeupFrames = mutableListOf<HandState>()

    // Legacy swipe tracking
    private var swipeStartY = -1f
    private var swipePreviousY = -1f
    private var swipeConsecutiveUp = 0
    private var swipeConsecutiveDown = 0
    private var swipeCumulativeDisplacement = 0f
    private val swipeYHistory = mutableListOf<Float>()

    // ── Cooldowns ────────────────────────────────────────────────
    @Volatile private var lastFrameTime = 0L
    @Volatile private var lastGestureTime = 0L
    @Volatile private var lastSwipeTime = 0L

    // ── Stats ────────────────────────────────────────────────────
    private val totalFrameCount = AtomicLong(0)
    private var cameraFrameCount = AtomicLong(0)
    private var droppedFrameCount = AtomicLong(0)
    private var lastDebugFrameTime = 0L
    private var firstFrameLogged = false

    private var scope: CoroutineScope? = null
    private val isDebug: Boolean get() = ServiceState.isDebug()

    // ─────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────

    fun start() {
        if (running) return
        running = true

        scope = CoroutineScope(
            Dispatchers.Main + SupervisorJob() +
                    CoroutineExceptionHandler { _, t -> Log.e(TAG, "Coroutine error", t) }
        )

        scope?.launch {
            try {
                addLog("Initializing gesture detector...")

                // 1. MediaPipe hand detector (required for both modes)
                handDetector = HandLandmarkDetector(context)
                if (handDetector?.isInitialized == true) {
                    addLog("MediaPipe loaded")
                } else {
                    addLog("MediaPipe FAILED — cannot start"); return@launch
                }

                // 2. Decide detection mode
                val userWantsNN = AppSettings.useNeuralNetwork
                if (userWantsNN) {
                    gestureClassifier = GestureClassifier(context)
                    if (gestureClassifier?.isInitialized == true) {
                        activeMode = DetectionMode.NEURAL_NETWORK
                        addLog("TCN model loaded — using Neural Network detection")
                    } else {
                        activeMode = DetectionMode.LEGACY
                        addLog("TCN model failed to load — falling back to Legacy detection")
                        gestureClassifier = null
                    }
                } else {
                    activeMode = DetectionMode.LEGACY
                    addLog("Using Legacy (rule-based) detection")
                }

                startCamera()
            } catch (e: Exception) {
                addLog("Start failed: ${e.message}")
            }
        }

        // Periodic stats
        scope?.launch {
            delay(5000)
            while (isActive && running) {
                delay(10_000)
                if (isDebug) {
                    addLog("Stats: cam=${cameraFrameCount.get()} proc=${totalFrameCount.get()} " +
                            "drop=${droppedFrameCount.get()} stage=$currentStage mode=$activeMode")
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

    // ─────────────────────────────────────────────────────────────
    // Camera
    // ─────────────────────────────────────────────────────────────

    private suspend fun startCamera() {
        addLog("Starting front camera...")
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
                addLog("Camera bound! sensor=$sensorRotation")
                val modeLabel = if (activeMode == DetectionMode.NEURAL_NETWORK) "Neural Network" else "Legacy"
                addLog("IDLE — scanning at ~${Constants.IDLE_FPS}fps ($modeLabel)")
            } catch (e: Exception) { addLog("Camera bind failed: ${e.message}") }
        }
    }

    private suspend fun getCameraProvider(): ProcessCameraProvider =
        suspendCoroutine { cont ->
            val f = ProcessCameraProvider.getInstance(context)
            f.addListener({ cont.resume(f.get()) }, ContextCompat.getMainExecutor(context))
        }

    // ─────────────────────────────────────────────────────────────
    // Frame Processing
    // ─────────────────────────────────────────────────────────────

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
                addLog("First frame: ${actualImageWidth}x${actualImageHeight} rot=$actualRotation")
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
                    addLog("Debug frame saved")
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

    // ─────────────────────────────────────────────────────────────
    // IDLE
    // ─────────────────────────────────────────────────────────────

    private fun processIdle(detail: HandLandmarkDetector.DetectionDetail) {
        idleFrameCount++

        when (activeMode) {
            DetectionMode.NEURAL_NETWORK -> processIdleTcn(detail)
            DetectionMode.LEGACY         -> processIdleLegacy(detail)
        }
    }

    /** TCN IDLE: wake up as soon as any hand is seen consistently. */
    private fun processIdleTcn(detail: HandLandmarkDetector.DetectionDetail) {
        val handDetected = detail.handsFound > 0
        idleWindowHand.addLast(handDetected)
        while (idleWindowHand.size > Constants.IDLE_WINDOW_SIZE) idleWindowHand.removeFirst()

        if (isDebug && idleFrameCount % IDLE_LOG_INTERVAL_FRAMES == 0) {
            val emoji = when {
                !handDetected -> "."
                detail.state == HandState.PALM -> "P"
                detail.state == HandState.FIST -> "F"
                else -> "H"
            }
            val windowStr = idleWindowHand.joinToString("") { if (it) "H" else "." }
            val handCount = idleWindowHand.count { it }
            addLog("IDLE #$idleFrameCount $emoji | $windowStr | hand=$handCount/${Constants.IDLE_WINDOW_SIZE} | ${detail.summary()}")
        }

        if (idleWindowHand.size < Constants.IDLE_WINDOW_SIZE) return

        val handCount = idleWindowHand.count { it }
        if (handCount >= Constants.IDLE_TRIGGER_THRESHOLD) {
            enterWakeupTcn()
        }
    }

    /** Legacy IDLE: detect palm or fist via rule-based classification. */
    private fun processIdleLegacy(detail: HandLandmarkDetector.DetectionDetail) {
        idleWindowStates.addLast(detail.state)
        while (idleWindowStates.size > Constants.IDLE_WINDOW_SIZE) idleWindowStates.removeFirst()

        if (isDebug && idleFrameCount % IDLE_LOG_INTERVAL_FRAMES == 0) {
            val windowStr = idleWindowStates.joinToString("") { stateEmoji(it) }
            val p = idleWindowStates.count { it == HandState.PALM }
            val f = idleWindowStates.count { it == HandState.FIST }
            addLog("IDLE #$idleFrameCount | $windowStr | P=$p F=$f | ${detail.summary()}")
        }

        if (idleWindowStates.size < Constants.IDLE_WINDOW_SIZE) return

        val palmCount = idleWindowStates.count { it == HandState.PALM }
        val fistCount = idleWindowStates.count { it == HandState.FIST }

        when {
            palmCount >= Constants.IDLE_TRIGGER_THRESHOLD ->
                enterWakeupLegacy(HandState.PALM, HandState.FIST)
            fistCount >= Constants.IDLE_TRIGGER_THRESHOLD ->
                enterWakeupLegacy(HandState.FIST, HandState.PALM)
        }
    }

    // ─────────────────────────────────────────────────────────────
    // WAKEUP — enter
    // ─────────────────────────────────────────────────────────────

    private fun enterWakeupTcn() {
        val now = System.currentTimeMillis()
        if (now - lastGestureTime < Constants.GRAB_COOLDOWN_MS) {
            if (isDebug) addLog("Wakeup suppressed (cooldown)")
            return
        }

        currentStage = Stage.WAKEUP
        wakeupStartTime = SystemClock.uptimeMillis()
        wakeupFrameCount = 0
        idleWindowHand.clear()
        idleFrameCount = 0
        gestureClassifier?.reset()

        addLog("WAKEUP! Hand detected — classifying with Neural Network...")
        mainHandler.post { overlayManager.showWakeupIndicator("\uD83E\uDD1A") }  // raised hand emoji
    }

    private fun enterWakeupLegacy(triggerState: HandState, targetState: HandState) {
        val now = System.currentTimeMillis()
        if (now - lastGestureTime < Constants.GRAB_COOLDOWN_MS &&
            now - lastSwipeTime < Constants.SWIPE_COOLDOWN_MS) {
            if (isDebug) addLog("Wakeup suppressed (cooldown)")
            return
        }

        currentStage = Stage.WAKEUP
        wakeupStartTime = SystemClock.uptimeMillis()
        wakeupTriggerState = triggerState
        wakeupTargetState = targetState
        consecutiveTargetFrames = 0
        wakeupFrames.clear()
        wakeupFrameCount = 0
        idleWindowStates.clear()
        idleFrameCount = 0

        // Reset swipe tracking
        swipeStartY = -1f
        swipePreviousY = -1f
        swipeConsecutiveUp = 0
        swipeConsecutiveDown = 0
        swipeCumulativeDisplacement = 0f
        swipeYHistory.clear()

        val motion = if (targetState == HandState.FIST) "GRAB" else "RELEASE"
        addLog("WAKEUP! Detecting $motion or swipe (Legacy)...")

        val indicator = when (targetState) {
            HandState.FIST -> "\u270A"   // fist
            HandState.PALM -> "\uD83E\uDD1A"  // raised hand
            else -> "\uD83D\uDC4B"  // waving hand
        }
        mainHandler.post { overlayManager.showWakeupIndicator(indicator) }
    }

    // ─────────────────────────────────────────────────────────────
    // WAKEUP — process
    // ─────────────────────────────────────────────────────────────

    private fun processWakeup(detail: HandLandmarkDetector.DetectionDetail, now: Long) {
        wakeupFrameCount++

        when (activeMode) {
            DetectionMode.NEURAL_NETWORK -> processWakeupTcn(detail, now)
            DetectionMode.LEGACY         -> processWakeupLegacy(detail, now)
        }
    }

    // ── TCN wakeup ───────────────────────────────────────────────

    private fun processWakeupTcn(detail: HandLandmarkDetector.DetectionDetail, now: Long) {
        val elapsed = now - wakeupStartTime

        if (detail.handsFound > 0 && detail.rawLandmarks != null && gestureClassifier?.isInitialized == true) {
            val result = gestureClassifier?.addFrameAndClassify(detail.rawLandmarks)

            if (result != null && result.confidence >= CONFIDENCE_THRESHOLD && result.gesture != "noise") {
                val event = when (result.gesture) {
                    "grab"       -> GestureEvent.Grab
                    "release"    -> GestureEvent.Release
                    "swipe_up"   -> GestureEvent.SwipeUp
                    "swipe_down" -> GestureEvent.SwipeDown
                    else         -> null
                }
                if (event != null) {
                    val emoji = when (result.gesture) {
                        "grab" -> "fist"; "release" -> "palm"
                        "swipe_up" -> "up"; "swipe_down" -> "down"; else -> "?"
                    }
                    addLog("${result.gesture.uppercase()} detected! conf=${"%.2f".format(result.confidence)} frames=${result.validFrames}")
                    lastGestureTime = System.currentTimeMillis()
                    scope?.launch { _events.emit(event) }
                    exitWakeup()
                    return
                }
            }

            if (isDebug && wakeupFrameCount % WAKEUP_LOG_INTERVAL_FRAMES == 0) {
                val remaining = Constants.WAKEUP_DURATION_MS - elapsed
                val info = if (result != null) {
                    "${result.gesture}(${"%.2f".format(result.confidence)}) v=${result.validFrames}"
                } else "collecting..."
                addLog("WK #$wakeupFrameCount ${"%.1f".format(remaining / 1000.0)}s | $info | ${detail.summary()}")
            }
        } else if (isDebug && wakeupFrameCount % WAKEUP_LOG_INTERVAL_FRAMES == 0) {
            val remaining = Constants.WAKEUP_DURATION_MS - elapsed
            addLog("WK #$wakeupFrameCount ${"%.1f".format(remaining / 1000.0)}s | no hand | ${detail.summary()}")
        }

        if (elapsed > Constants.WAKEUP_DURATION_MS) {
            addLog("WAKEUP timeout")
            exitWakeup()
        }
    }

    // ── Legacy wakeup ────────────────────────────────────────────

    private fun processWakeupLegacy(detail: HandLandmarkDetector.DetectionDetail, now: Long) {
        wakeupFrames.add(detail.state)
        val elapsed = now - wakeupStartTime
        val remaining = Constants.WAKEUP_DURATION_MS - elapsed

        // 1. Check swipe first (higher priority, faster response)
        if (detail.handsFound > 0) {
            val swipeEvent = checkSwipeLegacy(detail)
            if (swipeEvent != null) {
                val dir = if (swipeEvent == GestureEvent.SwipeUp) "SWIPE UP" else "SWIPE DOWN"
                addLog("$dir CONFIRMED!")
                lastSwipeTime = System.currentTimeMillis()
                scope?.launch { _events.emit(swipeEvent) }
                exitWakeup()
                return
            }
        }

        // 2. Check grab / release
        if (detail.state == wakeupTargetState) {
            consecutiveTargetFrames++
        } else {
            consecutiveTargetFrames = 0
        }

        if (isDebug && (wakeupFrameCount % WAKEUP_LOG_INTERVAL_FRAMES == 0 || consecutiveTargetFrames > 0)) {
            val se = stateEmoji(detail.state)
            val pb = progressBar(consecutiveTargetFrames, Constants.WAKEUP_CONFIRM_FRAMES)
            val swipeInfo = "dy=${"%.3f".format(swipeCumulativeDisplacement)} " +
                    "up=$swipeConsecutiveUp dn=$swipeConsecutiveDown"
            addLog("WK #$wakeupFrameCount ${"%.1f".format(remaining / 1000.0)}s | " +
                    "$se str=$consecutiveTargetFrames/${Constants.WAKEUP_CONFIRM_FRAMES} $pb | $swipeInfo")
        }

        // Timeout
        if (elapsed > Constants.WAKEUP_DURATION_MS) {
            if (isDebug) addLog("Timeout — ${buildWakeupSummary()}")
            else addLog("WAKEUP timeout")
            exitWakeup()
            return
        }

        // Confirm
        if (consecutiveTargetFrames >= Constants.WAKEUP_CONFIRM_FRAMES) {
            val event = when (wakeupTargetState) {
                HandState.FIST -> { addLog("GRAB CONFIRMED!"); GestureEvent.Grab }
                HandState.PALM -> { addLog("RELEASE CONFIRMED!"); GestureEvent.Release }
                else -> null
            }
            if (event != null) {
                lastGestureTime = System.currentTimeMillis()
                scope?.launch { _events.emit(event) }
            }
            exitWakeup()
        }
    }

    // ── Legacy swipe detection ───────────────────────────────────

    private fun checkSwipeLegacy(detail: HandLandmarkDetector.DetectionDetail): GestureEvent? {
        if (System.currentTimeMillis() - lastSwipeTime < Constants.SWIPE_COOLDOWN_MS) return null

        val currentY = detail.centerY
        swipeYHistory.add(currentY)

        if (swipeStartY < 0f) {
            swipeStartY = currentY
            swipePreviousY = currentY
            return null
        }

        val frameVelocity = currentY - swipePreviousY
        swipePreviousY = currentY
        swipeCumulativeDisplacement = currentY - swipeStartY

        if (frameVelocity < -Constants.SWIPE_MIN_VELOCITY) {
            swipeConsecutiveUp++; swipeConsecutiveDown = 0
        } else if (frameVelocity > Constants.SWIPE_MIN_VELOCITY) {
            swipeConsecutiveDown++; swipeConsecutiveUp = 0
        }

        val totalDisplacement = abs(swipeCumulativeDisplacement)
        val hasEnoughFrames = swipeConsecutiveUp >= Constants.SWIPE_CONFIRM_FRAMES ||
                swipeConsecutiveDown >= Constants.SWIPE_CONFIRM_FRAMES
        val hasEnoughDisplacement = totalDisplacement >= Constants.SWIPE_DISPLACEMENT_THRESHOLD

        // Y-history trend check
        if (swipeYHistory.size >= Constants.SWIPE_CONFIRM_FRAMES) {
            val recentY = swipeYHistory.takeLast(Constants.SWIPE_CONFIRM_FRAMES)
            val startAvg = recentY.take(2).average().toFloat()
            val endAvg = recentY.takeLast(2).average().toFloat()
            val trendDisplacement = endAvg - startAvg

            if (abs(trendDisplacement) >= Constants.SWIPE_DISPLACEMENT_THRESHOLD) {
                return if (trendDisplacement < 0) GestureEvent.SwipeUp
                else GestureEvent.SwipeDown
            }
        }

        if (hasEnoughFrames && hasEnoughDisplacement) {
            return if (swipeCumulativeDisplacement < 0) GestureEvent.SwipeUp
            else GestureEvent.SwipeDown
        }

        return null
    }

    // ─────────────────────────────────────────────────────────────
    // Wakeup Exit
    // ─────────────────────────────────────────────────────────────

    private fun exitWakeup() {
        currentStage = Stage.IDLE
        wakeupFrameCount = 0
        idleWindowStates.clear()
        idleWindowHand.clear()
        idleFrameCount = 0
        // Legacy state
        consecutiveTargetFrames = 0
        wakeupFrames.clear()
        swipeYHistory.clear()
        // TCN state
        gestureClassifier?.reset()
        mainHandler.post { overlayManager.hideWakeupIndicator() }
        addLog("Back to IDLE")
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    private fun buildWakeupSummary(): String {
        val total = wakeupFrames.size
        val p = wakeupFrames.count { it == HandState.PALM }
        val f = wakeupFrames.count { it == HandState.FIST }
        val timeline = wakeupFrames.takeLast(30).joinToString("") { stateEmoji(it) }
        return "total=$total P=$p F=$f dy=${"%.3f".format(swipeCumulativeDisplacement)} | $timeline"
    }

    private fun stateEmoji(s: HandState) = when (s) {
        HandState.PALM -> "P"; HandState.FIST -> "F"
        HandState.UNKNOWN -> "?"; HandState.NONE -> "."
    }

    private fun progressBar(current: Int, max: Int): String {
        val f = current.coerceAtMost(max); val e = max - f
        return "[" + "#".repeat(f) + "-".repeat(e) + "]"
    }

    private fun addLog(msg: String) {
        val t = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        val fullMsg = "$t $msg"
        Log.d(TAG, fullMsg)
        ServiceState.addEvent(fullMsg)
    }
}
