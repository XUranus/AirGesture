package com.grabdrop.util

import android.content.Context
import android.content.SharedPreferences

/**
 * Persistent settings backed by SharedPreferences.
 * Each setting has a default that matches the original compile-time
 * constant in Constants.kt, and typed accessors so the rest of the
 * code can read values at runtime.
 */
object AppSettings {

    private const val PREFS_NAME = "grabdrop_settings"

    // ── Keys ──────────────────────────────────────────────────────
    // Detection Method
    const val KEY_USE_NEURAL_NETWORK        = "use_neural_network"

    // Gesture Timing
    const val KEY_IDLE_FPS                  = "idle_fps"
    const val KEY_WAKEUP_FRAME_INTERVAL_MS  = "wakeup_frame_interval_ms"
    const val KEY_IDLE_WINDOW_SIZE          = "idle_window_size"
    const val KEY_IDLE_TRIGGER_THRESHOLD    = "idle_trigger_threshold"
    const val KEY_WAKEUP_DURATION_MS        = "wakeup_duration_ms"
    const val KEY_WAKEUP_CONFIRM_FRAMES     = "wakeup_confirm_frames"

    // Hand Classification
    const val KEY_FINGER_EXTENDED_THRESHOLD = "finger_extended_threshold"
    const val KEY_FINGER_CURLED_THRESHOLD   = "finger_curled_threshold"
    const val KEY_MIN_FINGERS_FOR_PALM      = "min_fingers_for_palm"
    const val KEY_MIN_FINGERS_FOR_FIST      = "min_fingers_for_fist"

    // Swipe Detection
    const val KEY_SWIPE_DISPLACEMENT        = "swipe_displacement_threshold"
    const val KEY_SWIPE_CONFIRM_FRAMES      = "swipe_confirm_frames"
    const val KEY_SWIPE_MIN_VELOCITY        = "swipe_min_velocity"
    const val KEY_SWIPE_COOLDOWN_MS         = "swipe_cooldown_ms"

    // Network
    const val KEY_UDP_PORT                  = "udp_port"
    const val KEY_MULTICAST_GROUP           = "multicast_group"
    const val KEY_SCREENSHOT_OFFER_TIMEOUT  = "screenshot_offer_timeout_ms"
    const val KEY_GRAB_COOLDOWN_MS          = "grab_cooldown_ms"

    // Sound
    const val KEY_SOUND_ENABLED             = "sound_enabled"

    // ── Defaults (mirror original Constants values) ──────────────
    const val DEF_USE_NEURAL_NETWORK        = true
    const val DEF_IDLE_FPS                  = 10
    const val DEF_WAKEUP_FRAME_INTERVAL_MS  = 33L
    const val DEF_IDLE_WINDOW_SIZE          = 10
    const val DEF_IDLE_TRIGGER_THRESHOLD    = 8
    const val DEF_WAKEUP_DURATION_MS        = 2_000L
    const val DEF_WAKEUP_CONFIRM_FRAMES     = 8
    const val DEF_FINGER_EXTENDED_THRESHOLD = 1.3f
    const val DEF_FINGER_CURLED_THRESHOLD   = 0.9f
    const val DEF_MIN_FINGERS_FOR_PALM      = 3
    const val DEF_MIN_FINGERS_FOR_FIST      = 3
    const val DEF_SWIPE_DISPLACEMENT        = 0.12f
    const val DEF_SWIPE_CONFIRM_FRAMES      = 5
    const val DEF_SWIPE_MIN_VELOCITY        = 0.008f
    const val DEF_SWIPE_COOLDOWN_MS         = 800L
    const val DEF_UDP_PORT                  = 9877
    const val DEF_MULTICAST_GROUP           = "239.255.77.88"
    const val DEF_SCREENSHOT_OFFER_TIMEOUT  = 10_000L
    const val DEF_GRAB_COOLDOWN_MS          = 3_000L
    const val DEF_SOUND_ENABLED             = true

    private var prefs: SharedPreferences? = null

    /** Must be called once from Application.onCreate(). */
    fun init(context: Context) {
        prefs = context.applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    fun p(): SharedPreferences =
        prefs ?: throw IllegalStateException("AppSettings.init() was not called")

    // ── Typed Getters ────────────────────────────────────────────

    // Detection method
    val useNeuralNetwork: Boolean
        get() = p().getBoolean(KEY_USE_NEURAL_NETWORK, DEF_USE_NEURAL_NETWORK)

    // Gesture timing
    val idleFps: Int
        get() = p().getInt(KEY_IDLE_FPS, DEF_IDLE_FPS)
    val idleFrameIntervalMs: Long
        get() = 1000L / idleFps
    val wakeupFrameIntervalMs: Long
        get() = p().getLong(KEY_WAKEUP_FRAME_INTERVAL_MS, DEF_WAKEUP_FRAME_INTERVAL_MS)
    val idleWindowSize: Int
        get() = p().getInt(KEY_IDLE_WINDOW_SIZE, DEF_IDLE_WINDOW_SIZE)
    val idleTriggerThreshold: Int
        get() = p().getInt(KEY_IDLE_TRIGGER_THRESHOLD, DEF_IDLE_TRIGGER_THRESHOLD)
    val wakeupDurationMs: Long
        get() = p().getLong(KEY_WAKEUP_DURATION_MS, DEF_WAKEUP_DURATION_MS)
    val wakeupConfirmFrames: Int
        get() = p().getInt(KEY_WAKEUP_CONFIRM_FRAMES, DEF_WAKEUP_CONFIRM_FRAMES)

    // Hand classification
    val fingerExtendedThreshold: Float
        get() = p().getFloat(KEY_FINGER_EXTENDED_THRESHOLD, DEF_FINGER_EXTENDED_THRESHOLD)
    val fingerCurledThreshold: Float
        get() = p().getFloat(KEY_FINGER_CURLED_THRESHOLD, DEF_FINGER_CURLED_THRESHOLD)
    val minFingersForPalm: Int
        get() = p().getInt(KEY_MIN_FINGERS_FOR_PALM, DEF_MIN_FINGERS_FOR_PALM)
    val minFingersForFist: Int
        get() = p().getInt(KEY_MIN_FINGERS_FOR_FIST, DEF_MIN_FINGERS_FOR_FIST)

    // Swipe detection
    val swipeDisplacementThreshold: Float
        get() = p().getFloat(KEY_SWIPE_DISPLACEMENT, DEF_SWIPE_DISPLACEMENT)
    val swipeConfirmFrames: Int
        get() = p().getInt(KEY_SWIPE_CONFIRM_FRAMES, DEF_SWIPE_CONFIRM_FRAMES)
    val swipeMinVelocity: Float
        get() = p().getFloat(KEY_SWIPE_MIN_VELOCITY, DEF_SWIPE_MIN_VELOCITY)
    val swipeCooldownMs: Long
        get() = p().getLong(KEY_SWIPE_COOLDOWN_MS, DEF_SWIPE_COOLDOWN_MS)

    // Network
    val udpPort: Int
        get() = p().getInt(KEY_UDP_PORT, DEF_UDP_PORT)
    val multicastGroup: String
        get() = p().getString(KEY_MULTICAST_GROUP, DEF_MULTICAST_GROUP) ?: DEF_MULTICAST_GROUP
    val screenshotOfferTimeoutMs: Long
        get() = p().getLong(KEY_SCREENSHOT_OFFER_TIMEOUT, DEF_SCREENSHOT_OFFER_TIMEOUT)
    val grabCooldownMs: Long
        get() = p().getLong(KEY_GRAB_COOLDOWN_MS, DEF_GRAB_COOLDOWN_MS)

    // Sound
    val soundEnabled: Boolean
        get() = p().getBoolean(KEY_SOUND_ENABLED, DEF_SOUND_ENABLED)

    // ── Setters (individual) ─────────────────────────────────────

    fun setInt(key: String, value: Int)       = p().edit().putInt(key, value).apply()
    fun setLong(key: String, value: Long)     = p().edit().putLong(key, value).apply()
    fun setFloat(key: String, value: Float)   = p().edit().putFloat(key, value).apply()
    fun setString(key: String, value: String) = p().edit().putString(key, value).apply()
    fun setBoolean(key: String, value: Boolean) = p().edit().putBoolean(key, value).apply()

    /** Reset all settings to defaults. */
    fun resetAll() {
        p().edit().clear().apply()
    }
}
