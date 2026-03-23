// /GrabDrop/app/src/main/java/com/grabdrop/gesture/GestureClassifier.kt
package com.grabdrop.gesture

import android.content.Context
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.*

/**
 * TCN-based gesture classifier using ONNX Runtime.
 *
 * Features (144 dimensions):
 * - Normalized landmarks (63): relative to wrist, divided by palm size
 * - Velocity (63): frame-to-frame difference
 * - Wrist velocity (3): wrist position change
 * - Finger distances (10): distances between fingertips
 * - Finger angles (5): bending angle for each finger
 */
class GestureClassifier(private val context: Context) {

    companion object {
        private const val TAG = "GestureClassifier"
        private const val MODEL_FILE = "gesture_tcn_pruned_quantized.onnx"
        private const val CONFIG_FILE = "config.json"
    }

    // Model configuration
    private var seqLen: Int = 30
    private var featureDim: Int = 144
    private var rawDim: Int = 63
    private var numLandmarks: Int = 21
    private var classNames: List<String> = listOf("grab", "release", "swipe_up", "swipe_down", "noise")

    // Normalization parameters
    private var normalizeMean: FloatArray = FloatArray(0)
    private var normalizeStd: FloatArray = FloatArray(0)

    // Feature computation constants
    private var pairs: List<Pair<Int, Int>> = listOf()  // Finger tip pairs for distance
    private var fingerChains: List<List<Int>> = listOf()  // Finger joint chains for angle

    // ONNX Runtime
    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    var isInitialized = false
        private set

    // Sliding window for sequence (pre-filled with zeros)
    private val window = ArrayDeque<FloatArray>()

    // Track if each frame is real (non-zero) or placeholder
    private val isRealFrame = ArrayDeque<Boolean>()

    // Previous frame for velocity calculation
    private var prevNormLandmarks: FloatArray? = null
    private var prevWrist: FloatArray? = null

    // Landmark indices
    private val WRIST_IDX = 0
    private val MID_FINGER_IDX = 9

    data class ClassificationResult(
        val gesture: String,
        val confidence: Float,
        val classIndex: Int,
        val validFrames: Int  // Number of non-zero frames in window
    )

    // Minimum number of real frames needed to return a result
    private val MIN_VALID_FRAMES = 15

    init {
        try {
            Log.d(TAG, "Initializing GestureClassifier...")
            loadConfig()
            Log.d(TAG, "Config loaded successfully")
            loadModel()
            Log.d(TAG, "Model loaded successfully")
            initializeWindow()
            Log.d(TAG, "Window initialized")
            isInitialized = true
            Log.d(TAG, "GestureClassifier initialized: seqLen=$seqLen, featureDim=$featureDim")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize GestureClassifier: ${e.message}", e)
            isInitialized = false
        }
    }

    private fun loadConfig() {
        try {
            Log.d(TAG, "Loading config from assets: $CONFIG_FILE")

            // Check if file exists
            val assetFiles = context.assets.list("") ?: arrayOf()
            if (!assetFiles.contains(CONFIG_FILE)) {
                throw RuntimeException("Config file not found in assets: $CONFIG_FILE. Available: ${assetFiles.joinToString()}")
            }

            val configJson = context.assets.open(CONFIG_FILE).bufferedReader().use { it.readText() }
            Log.d(TAG, "Config JSON loaded: ${configJson.length} chars")

            val json = JSONObject(configJson)

            seqLen = json.getInt("seq_len")
            featureDim = json.getInt("feature_dim")
            rawDim = json.getInt("raw_dim")
            numLandmarks = json.getInt("num_landmarks")

            // Load class names
            val classNamesArray = json.getJSONArray("class_names")
            classNames = (0 until classNamesArray.length()).map { classNamesArray.getString(it) }

            // Load normalization parameters
            val meanArray = json.getJSONArray("normalize_mean")
            val stdArray = json.getJSONArray("normalize_std")
            normalizeMean = FloatArray(meanArray.length()) { meanArray.getDouble(it).toFloat() }
            normalizeStd = FloatArray(stdArray.length()) { stdArray.getDouble(it).toFloat() }

            // Load pairs for finger distance
            val pairsArray = json.getJSONArray("pairs")
            pairs = (0 until pairsArray.length()).map { i ->
                val pair = pairsArray.getJSONArray(i)
                Pair(pair.getInt(0), pair.getInt(1))
            }

            // Load finger chains for angle calculation
            val chainsArray = json.getJSONArray("finger_chains")
            fingerChains = (0 until chainsArray.length()).map { i ->
                val chain = chainsArray.getJSONArray(i)
                (0 until chain.length()).map { chain.getInt(it) }
            }

            Log.d(TAG, "Config loaded: ${classNames.size} classes, ${pairs.size} pairs, ${fingerChains.size} finger chains")
            Log.d(TAG, "seqLen=$seqLen, featureDim=$featureDim, rawDim=$rawDim, numLandmarks=$numLandmarks")

        } catch (e: Exception) {
            Log.e(TAG, "Error loading config: ${e.message}", e)
            throw e
        }
    }

    private fun loadModel() {
        try {
            Log.d(TAG, "Loading ONNX model from assets: $MODEL_FILE")

            // Check if file exists in assets
            val assetFiles = context.assets.list("") ?: arrayOf()
            if (!assetFiles.contains(MODEL_FILE)) {
                throw RuntimeException("Model file not found in assets: $MODEL_FILE. Available: ${assetFiles.joinToString()}")
            }

            // Create ONNX environment
            env = OrtEnvironment.getEnvironment()
            if (env == null) {
                throw RuntimeException("Failed to create OrtEnvironment")
            }
            Log.d(TAG, "OrtEnvironment created")

            // Load model bytes from assets
            val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            Log.d(TAG, "Model bytes loaded: ${modelBytes.size} bytes")

            // Create session options
            val options = OrtSession.SessionOptions()
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            // Create session
            session = env!!.createSession(modelBytes, options)
            Log.d(TAG, "ONNX session created: inputNames=${session?.inputNames}, outputNames=${session?.outputNames}")

        } catch (e: Exception) {
            Log.e(TAG, "Error loading ONNX model: ${e.message}", e)
            throw e
        }
    }

    private fun initializeWindow() {
        window.clear()
        isRealFrame.clear()
        // Don't pre-fill - wait for real frames
    }

    /**
     * Reset the sliding window and state.
     */
    fun reset() {
        initializeWindow()
        prevNormLandmarks = null
        prevWrist = null
    }

    /**
     * Add a frame of landmarks and get classification result.
     * @param landmarks Raw landmarks as FloatArray of size 63 (21 points x 3 coords)
     * @return ClassificationResult or null if model not initialized or not enough valid frames
     */
    fun addFrameAndClassify(landmarks: FloatArray?): ClassificationResult? {
        if (!isInitialized || session == null) return null

        // Compute features
        val features = computeFeatures(landmarks)

        // Add to window (sliding, remove oldest if full)
        if (window.size >= seqLen) {
            window.removeFirst()
            isRealFrame.removeFirst()
        }
        window.addLast(features)
        // Mark as real frame if we have valid landmarks
        isRealFrame.addLast(landmarks != null && landmarks.size >= rawDim)

        // Count valid (real) frames in window
        val validFrames = isRealFrame.count { it }
        if (validFrames < MIN_VALID_FRAMES) {
            return null  // Not enough real frames yet
        }

        // Run inference
        return classify(validFrames)
    }

    /**
     * Compute 144-dimensional feature vector from raw landmarks.
     */
    private fun computeFeatures(landmarks: FloatArray?): FloatArray {
        val features = FloatArray(featureDim)

        // Handle null or invalid input
        if (landmarks == null || landmarks.size < rawDim) {
            // Use zeros for invalid input, but keep velocity from previous
            if (prevNormLandmarks != null) {
                // Copy previous velocity (indices 63-125)
                System.arraycopy(prevNormLandmarks!!, 0, features, 63, 63)
            }
            return features
        }

        // Reshape landmarks to (21, 3)
        val lms = Array(numLandmarks) { i ->
            floatArrayOf(
                landmarks[i * 3],
                landmarks[i * 3 + 1],
                landmarks[i * 3 + 2]
            )
        }

        // Extract wrist and mid finger
        val wrist = lms[WRIST_IDX]
        val midFinger = lms[MID_FINGER_IDX]

        // Calculate palm size (distance from wrist to middle finger MCP)
        val palmSize = sqrt(
            (midFinger[0] - wrist[0]).pow(2) +
            (midFinger[1] - wrist[1]).pow(2) +
            (midFinger[2] - wrist[2]).pow(2)
        ).coerceAtLeast(1e-6f)

        // 1. Normalized landmarks (63 dims): relative to wrist, divided by palm size
        val normLandmarks = FloatArray(rawDim)
        for (i in 0 until numLandmarks) {
            normLandmarks[i * 3] = (lms[i][0] - wrist[0]) / palmSize
            normLandmarks[i * 3 + 1] = (lms[i][1] - wrist[1]) / palmSize
            normLandmarks[i * 3 + 2] = (lms[i][2] - wrist[2]) / palmSize
        }
        System.arraycopy(normLandmarks, 0, features, 0, rawDim)

        // 2. Velocity (63 dims): frame-to-frame difference of normalized landmarks
        val velocity = if (prevNormLandmarks != null) {
            FloatArray(rawDim) { i ->
                normLandmarks[i] - prevNormLandmarks!![i]
            }
        } else {
            FloatArray(rawDim) { 0f }
        }
        System.arraycopy(velocity, 0, features, rawDim, rawDim)

        // 3. Wrist velocity (3 dims)
        val wristVel = if (prevWrist != null) {
            floatArrayOf(
                wrist[0] - prevWrist!![0],
                wrist[1] - prevWrist!![1],
                wrist[2] - prevWrist!![2]
            )
        } else {
            floatArrayOf(0f, 0f, 0f)
        }
        features[rawDim * 2] = wristVel[0]
        features[rawDim * 2 + 1] = wristVel[1]
        features[rawDim * 2 + 2] = wristVel[2]

        // 4. Finger distances (10 dims): distances between fingertip pairs
        val offset = rawDim * 2 + 3
        for ((idx, pair) in pairs.withIndex()) {
            val (i, j) = pair
            val dx = normLandmarks[i * 3] - normLandmarks[j * 3]
            val dy = normLandmarks[i * 3 + 1] - normLandmarks[j * 3 + 1]
            val dz = normLandmarks[i * 3 + 2] - normLandmarks[j * 3 + 2]
            features[offset + idx] = sqrt(dx * dx + dy * dy + dz * dz)
        }

        // 5. Finger angles (5 dims): bending angle for each finger
        val angleOffset = offset + pairs.size
        for ((idx, chain) in fingerChains.withIndex()) {
            if (chain.size < 3) continue

            // v1: from chain[0] to chain[1]
            // v2: from chain[1] to chain[last] (fingertip)
            val v1 = floatArrayOf(
                lms[chain[1]][0] - lms[chain[0]][0],
                lms[chain[1]][1] - lms[chain[0]][1],
                lms[chain[1]][2] - lms[chain[0]][2]
            )
            val lastIdx = chain[chain.size - 1]
            val v2 = floatArrayOf(
                lms[lastIdx][0] - lms[chain[1]][0],
                lms[lastIdx][1] - lms[chain[1]][1],
                lms[lastIdx][2] - lms[chain[1]][2]
            )

            val n1 = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]) + 1e-8f
            val n2 = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]) + 1e-8f

            val cosAngle = ((v1[0] / n1) * (v2[0] / n2) +
                           (v1[1] / n1) * (v2[1] / n2) +
                           (v1[2] / n1) * (v2[2] / n2)).coerceIn(-1f, 1f)

            features[angleOffset + idx] = acos(cosAngle)
        }

        // Save current state for next frame
        prevNormLandmarks = normLandmarks.copyOf()
        prevWrist = wrist.copyOf()

        // Debug: log some feature values
        if (Log.isLoggable(TAG, Log.DEBUG)) {
            // Log finger angles (last 5 dims) - should be different for FIST vs PALM
            val angles = FloatArray(5) { features[angleOffset + it] }
            Log.d(TAG, "Features: angles=[${angles.map { "%.2f".format(it) }.joinToString(", ")}] " +
                    "palmSize=${String.format("%.4f", palmSize)}")
        }

        return features
    }

    /**
     * Run inference on the current window.
     * @param validFrames Number of real (non-placeholder) frames in window
     */
    private fun classify(validFrames: Int): ClassificationResult? {
        val session = this.session ?: return null
        val env = this.env ?: return null

        try {
            // Prepare input: shape (1, featureDim, seqLen)
            // Real frames at the BEGINNING (time 0), zeros at the END
            val inputArray = Array(1) { Array(featureDim) { FloatArray(seqLen) } }
            val windowList = window.toList()
            val numFrames = minOf(windowList.size, seqLen)

            for (t in 0 until numFrames) {
                val frame = windowList[t]
                for (d in 0 until minOf(frame.size, featureDim)) {
                    // Normalize
                    val normalized = (frame[d] - normalizeMean[d]) / (normalizeStd[d] + 1e-8f)
                    inputArray[0][d][t] = normalized  // Place real frames at the beginning
                }
            }

            // Create ONNX tensor
            val inputName = session.inputNames.iterator().next()
            val inputTensor = OnnxTensor.createTensor(env, inputArray)

            // Run inference
            val output = session.run(mapOf(inputName to inputTensor))

            // Get output
            val outputTensor = output[0] as OnnxTensor
            Log.d(TAG, "Output shape: ${outputTensor.info.shape.joinToString()}")
            val outputBuffer = outputTensor.floatBuffer

            // Read output values from buffer
            val outputArray = FloatArray(classNames.size)
            outputBuffer.rewind()
            for (i in outputArray.indices) {
                outputArray[i] = outputBuffer.get()
            }

            // Debug: log raw outputs
            val outputStr = outputArray.mapIndexed { i, v ->
                "${classNames[i]}=${String.format("%.2f", v)}"
            }.joinToString(" ")
            Log.d(TAG, "Raw output: [$outputStr]")

            // Find max class
            var maxIdx = 0
            var maxVal = outputArray[0]
            for (i in 1 until classNames.size) {
                if (outputArray[i] > maxVal) {
                    maxVal = outputArray[i]
                    maxIdx = i
                }
            }

            // Apply softmax for confidence
            val maxLogit = outputArray.maxOrNull() ?: 0f
            var sumExp = 0f
            val probs = FloatArray(classNames.size) { i ->
                val exp = kotlin.math.exp(outputArray[i] - maxLogit)
                sumExp += exp
                exp
            }
            val confidence = probs[maxIdx] / sumExp

            inputTensor.close()
            output.close()

            return ClassificationResult(
                gesture = classNames[maxIdx],
                confidence = confidence,
                classIndex = maxIdx,
                validFrames = validFrames
            )
        } catch (e: Exception) {
            Log.e(TAG, "Classification error: ${e.message}", e)
            return null
        }
    }

    fun close() {
        try {
            session?.close()
            session = null
            env = null
        } catch (e: Exception) {
            Log.e(TAG, "Error closing classifier", e)
        }
    }
}
