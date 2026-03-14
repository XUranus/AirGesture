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
        private const val THUMB_TIP = 4
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

        // tip, pip, mcp for each finger
        private val FINGER_LANDMARKS = listOf(
            Triple(INDEX_TIP, INDEX_PIP, INDEX_MCP),
            Triple(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            Triple(RING_TIP, RING_PIP, RING_MCP),
            Triple(PINKY_TIP, PINKY_PIP, PINKY_MCP),
        )

        private val FINGER_NAMES = listOf("IDX", "MID", "RNG", "PNK")
    }

    data class DetectionDetail(
        val state: HandState,
        val fingerRatios: List<Float>,
        val extendedCount: Int,
        val curledCount: Int,
        val handsFound: Int,
        val confidence: Float,
        val detailMsg: String = ""
    ) {
        fun summary(): String {
            if (handsFound == 0) return "NO_HAND"
            val ratioStr = fingerRatios.zip(FINGER_NAMES).joinToString(" ") { (r, n) ->
                "$n:${"%.2f".format(r)}"
            }
            return "$state e=$extendedCount c=$curledCount conf=${"%.2f".format(confidence)} [$ratioStr]"
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
            Log.d(TAG, "HandLandmarker initialized (low confidence thresholds)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize HandLandmarker", e)
        }
    }

    fun detect(bitmap: Bitmap, timestampMs: Long): HandState {
        return detectDetailed(bitmap, timestampMs).state
    }

    fun detectDetailed(bitmap: Bitmap, timestampMs: Long): DetectionDetail {
        val landmarker = handLandmarker ?: return DetectionDetail(
            HandState.NONE, emptyList(), 0, 0, 0, 0f, "landmarker=null"
        )

        if (bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
            return DetectionDetail(
                HandState.NONE, emptyList(), 0, 0, 0, 0f,
                "invalid bitmap ${bitmap.width}x${bitmap.height} recycled=${bitmap.isRecycled}"
            )
        }

        val ts = if (timestampMs <= lastTimestamp) lastTimestamp + 1 else timestampMs
        lastTimestamp = ts

        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = landmarker.detectForVideo(mpImage, ts)
            classifyResult(result)
        } catch (e: Exception) {
            Log.e(TAG, "Detection error: ${e.message}")
            DetectionDetail(HandState.NONE, emptyList(), 0, 0, 0, 0f, "error: ${e.message}")
        }
    }

    private fun classifyResult(result: HandLandmarkerResult): DetectionDetail {
        val hands = result.landmarks()
        if (hands.isEmpty()) {
            return DetectionDetail(HandState.NONE, emptyList(), 0, 0, 0, 0f, "no hands in result")
        }

        val landmarks = hands[0]
        if (landmarks.size < 21) {
            return DetectionDetail(
                HandState.UNKNOWN, emptyList(), 0, 0, hands.size, 0f,
                "only ${landmarks.size} landmarks"
            )
        }

        // Get confidence
        val confidence = if (result.handednesses().isNotEmpty() &&
            result.handednesses()[0].isNotEmpty()
        ) {
            result.handednesses()[0][0].score()
        } else {
            0f
        }

        val handedness = if (result.handednesses().isNotEmpty() &&
            result.handednesses()[0].isNotEmpty()
        ) {
            result.handednesses()[0][0].categoryName()
        } else {
            "?"
        }

        val wrist = landmarks[WRIST]

        var extendedCount = 0
        var curledCount = 0
        val ratios = mutableListOf<Float>()
        val fingerDetails = mutableListOf<String>()

        for ((idx, triple) in FINGER_LANDMARKS.withIndex()) {
            val (tipIdx, pipIdx, mcpIdx) = triple
            val tip = landmarks[tipIdx]
            val pip = landmarks[pipIdx]
            val mcp = landmarks[mcpIdx]

            // Method 1: tip-to-wrist vs mcp-to-wrist ratio
            val tipToWrist = distance(tip.x(), tip.y(), wrist.x(), wrist.y())
            val mcpToWrist = distance(mcp.x(), mcp.y(), wrist.x(), wrist.y())

            // Method 2: tip-to-mcp vs pip-to-mcp (curl detection)
            val tipToMcp = distance(tip.x(), tip.y(), mcp.x(), mcp.y())
            val pipToMcp = distance(pip.x(), pip.y(), mcp.x(), mcp.y())

            if (mcpToWrist < 0.001f) {
                ratios.add(0f)
                continue
            }

            val ratio = tipToWrist / mcpToWrist
            ratios.add(ratio)

            // Use both methods for more robust detection
            val isExtended = ratio > Constants.FINGER_EXTENDED_THRESHOLD
            val isCurled = ratio < Constants.FINGER_CURLED_THRESHOLD

            if (isExtended) {
                extendedCount++
                fingerDetails.add("${FINGER_NAMES[idx]}:EXT(${"%.2f".format(ratio)})")
            } else if (isCurled) {
                curledCount++
                fingerDetails.add("${FINGER_NAMES[idx]}:CRL(${"%.2f".format(ratio)})")
            } else {
                fingerDetails.add("${FINGER_NAMES[idx]}:MID(${"%.2f".format(ratio)})")
            }
        }

        val state = when {
            extendedCount >= Constants.MIN_FINGERS_FOR_PALM -> HandState.PALM
            curledCount >= Constants.MIN_FINGERS_FOR_FIST -> HandState.FIST
            else -> HandState.UNKNOWN
        }

        val detail = "$handedness conf=${"%.2f".format(confidence)} " +
                fingerDetails.joinToString(" ")

        return DetectionDetail(state, ratios, extendedCount, curledCount, hands.size, confidence, detail)
    }

    private fun distance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2
        val dy = y1 - y2
        return sqrt(dx * dx + dy * dy)
    }

    fun close() {
        try {
            handLandmarker?.close()
            handLandmarker = null
            Log.d(TAG, "HandLandmarker closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing HandLandmarker", e)
        }
    }
}
