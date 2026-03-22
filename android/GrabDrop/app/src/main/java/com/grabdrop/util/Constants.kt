package com.grabdrop.util

import java.util.UUID

/**
 * Central configuration object.
 *
 * The values that are user-configurable are now delegated to [AppSettings]
 * (backed by SharedPreferences), while truly fixed identifiers remain
 * compile-time constants.
 */
object Constants {
    // ── Fixed (not configurable) ─────────────────────────────────
    const val NOTIFICATION_CHANNEL_ID = "grabdrop_service"
    const val NOTIFICATION_ID = 1001
    const val SERVICE_ACTION_STOP = "com.grabdrop.STOP_SERVICE"

    const val BROADCAST_TYPE_SCREENSHOT_READY = "SCREENSHOT_READY"
    const val MOCK_RELEASE_DELAY_MS = 2_500L

    val DEVICE_ID: String = UUID.randomUUID().toString().substring(0, 8)

    // ── Configurable via AppSettings ─────────────────────────────

    // Network
    val UDP_PORT: Int get() = AppSettings.udpPort
    val MULTICAST_GROUP: String get() = AppSettings.multicastGroup
    val SCREENSHOT_OFFER_TIMEOUT_MS: Long get() = AppSettings.screenshotOfferTimeoutMs
    val GRAB_COOLDOWN_MS: Long get() = AppSettings.grabCooldownMs

    // Gesture detection – timing
    val IDLE_FPS: Int get() = AppSettings.idleFps
    val IDLE_FRAME_INTERVAL_MS: Long get() = AppSettings.idleFrameIntervalMs
    val WAKEUP_FRAME_INTERVAL_MS: Long get() = AppSettings.wakeupFrameIntervalMs
    val IDLE_WINDOW_SIZE: Int get() = AppSettings.idleWindowSize
    val IDLE_TRIGGER_THRESHOLD: Int get() = AppSettings.idleTriggerThreshold
    val WAKEUP_DURATION_MS: Long get() = AppSettings.wakeupDurationMs
    val WAKEUP_CONFIRM_FRAMES: Int get() = AppSettings.wakeupConfirmFrames

    // Gesture detection – hand classification
    val FINGER_EXTENDED_THRESHOLD: Float get() = AppSettings.fingerExtendedThreshold
    val FINGER_CURLED_THRESHOLD: Float get() = AppSettings.fingerCurledThreshold
    val MIN_FINGERS_FOR_PALM: Int get() = AppSettings.minFingersForPalm
    val MIN_FINGERS_FOR_FIST: Int get() = AppSettings.minFingersForFist

    // Swipe detection
    val SWIPE_DISPLACEMENT_THRESHOLD: Float get() = AppSettings.swipeDisplacementThreshold
    val SWIPE_CONFIRM_FRAMES: Int get() = AppSettings.swipeConfirmFrames
    val SWIPE_MIN_VELOCITY: Float get() = AppSettings.swipeMinVelocity
    val SWIPE_COOLDOWN_MS: Long get() = AppSettings.swipeCooldownMs
}
