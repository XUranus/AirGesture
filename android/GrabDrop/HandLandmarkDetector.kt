// /GrabDrop/app/src/main/java/com/grabdrop/gesture/HandLandmarkDetector.kt
package com.grabdrop.gesture

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.grabdrop.util.Constants
import kotlin.math.sqrt

class HandLandmarkDetector(context: Context) {

    companion object {
        private const val TAG = "HandLandmark"
        private const val MODEL_ASSET = "hand_landmarker.task"

        private const val WRIST = 0
        private const val INDEX_MCP = 5
        private const val INDEX_PIP = 6
        private const val INDEX_TIP = 8
        private const val MIDDLE_MCP = 9
        private const val MIDDLE_PIP = 10
        private const val MIDDLE_TIP = 12
        private const val RING_MCP = 13
        private const val RING_PIP = 14
        private const val RING_TIP = 16
        private const val PINKY_MCP = 17
        private const val PINKY_PIP = 18
        private const val PINKY_TIP = 20

        private val FINGER_LANDMARKS = listOf(
            Triple(INDEX_TIP, INDEX_PIP, INDEX_MCP),
            Triple(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            Triple(RING_TIP, RING_PIP, RING_MCP),
            Triple(PINKY_TIP, PINKY_PIP, PINKY_MCP),
        )

        val FINGER_NAMES = listOf("IDX", "MID", "RNG", "PNK")
    }

    data class DetectionDetail(
        val state: HandState,
        val fingerRatios: List<Float> = emptyList(),
        val extendedCount: Int = 0,
        val curledCount: Int = 0,
        val handsFound: Int = 0,
        val confidence: Float = 0f,
        val handedness: String = "?",
        // Hand center position (normalized 0-1)
        val centerX: Float = 0.5f,
        val centerY: Float = 0.5f,
        // Wrist position (normalized 0-1)
        val wristX: Float = 0.5f,
        val wristY: Float = 0.5f
    ) {
        fun summary(): String {
            if (handsFound == 0) return "NO_HAND"
            val ratioStr = fingerRatios.zip(FINGER_NAMES).joinToString(" ") { (r, n) ->
                "$n:${"%.2f".format(r)}"
            }
            return "$state e=$extendedCount c=$curledCount " +
                    "conf=${"%.2f".format(confidence)} " +
                    "pos=(${"%.2f".format(centerX)},${"%.2f".format(centerY)}) " +
                    "[$ratioStr]"
        }
    }

    private var handLandmarker: HandLandmarker? = null
    private var lastTimestamp = -1L
    var isInitialized = false
        private set

    init {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)
            isInitialized = true
            Log.d(TAG, "HandLandmarker initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize HandLandmarker", e)
        }
    }

    fun detect(bitmap: Bitmap, timestampMs: Long): HandState {
        return detectDetailed(bitmap, timestampMs).state
    }

    fun detectDetailed(bitmap: Bitmap, timestampMs: Long): DetectionDetail {
        val landmarker = handLandmarker ?: return DetectionDetail(state = HandState.NONE)

        if (bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
            return DetectionDetail(state = HandState.NONE)
        }

        val ts = if (timestampMs <= lastTimestamp) lastTimestamp + 1 else timestampMs
        lastTimestamp = ts

        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = landmarker.detectForVideo(mpImage, ts)
            classifyResult(result)
        } catch (e: Exception) {
            Log.e(TAG, "Detection error: ${e.message}")
            DetectionDetail(state = HandState.NONE)
        }
    }

    private fun classifyResult(result: HandLandmarkerResult): DetectionDetail {
        val hands = result.landmarks()
        if (hands.isEmpty()) return DetectionDetail(state = HandState.NONE)

        val landmarks = hands[0]
        if (landmarks.size < 21) {
            return DetectionDetail(state = HandState.UNKNOWN, handsFound = hands.size)
        }

        val confidence = if (result.handednesses().isNotEmpty() &&
            result.handednesses()[0].isNotEmpty()
        ) result.handednesses()[0][0].score() else 0f

        val handedness = if (result.handednesses().isNotEmpty() &&
            result.handednesses()[0].isNotEmpty()
        ) result.handednesses()[0][0].categoryName() else "?"

        val wrist = landmarks[WRIST]

        // Calculate hand center (average of wrist + all MCPs)
        val centerLandmarks = listOf(WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
        var sumX = 0f
        var sumY = 0f
        for (idx in centerLandmarks) {
            sumX += landmarks[idx].x()
            sumY += landmarks[idx].y()
        }
        val centerX = sumX / centerLandmarks.size
        val centerY = sumY / centerLandmarks.size

        var extended = 0
        var curled = 0
        val ratios = mutableListOf<Float>()

        for ((tipIdx, _, mcpIdx) in FINGER_LANDMARKS) {
            val tip = landmarks[tipIdx]
            val mcp = landmarks[mcpIdx]

            val tipToWrist = distance(tip.x(), tip.y(), wrist.x(), wrist.y())
            val mcpToWrist = distance(mcp.x(), mcp.y(), wrist.x(), wrist.y())

            if (mcpToWrist < 0.001f) { ratios.add(0f); continue }

            val ratio = tipToWrist / mcpToWrist
            ratios.add(ratio)

            when {
                ratio > Constants.FINGER_EXTENDED_THRESHOLD -> extended++
                ratio < Constants.FINGER_CURLED_THRESHOLD -> curled++
            }
        }

        val state = when {
            extended >= Constants.MIN_FINGERS_FOR_PALM -> HandState.PALM
            curled >= Constants.MIN_FINGERS_FOR_FIST -> HandState.FIST
            else -> HandState.UNKNOWN
        }

        return DetectionDetail(
            state = state,
            fingerRatios = ratios,
            extendedCount = extended,
            curledCount = curled,
            handsFound = hands.size,
            confidence = confidence,
            handedness = handedness,
            centerX = centerX,
            centerY = centerY,
            wristX = wrist.x(),
            wristY = wrist.y()
        )
    }

    private fun distance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2; val dy = y1 - y2
        return sqrt(dx * dx + dy * dy)
    }

    fun close() {
        try { handLandmarker?.close(); handLandmarker = null }
        catch (e: Exception) { Log.e(TAG, "Error closing", e) }
    }
}
