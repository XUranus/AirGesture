// /GrabDrop/app/src/main/java/com/grabdrop/util/Constants.kt
package com.grabdrop.util

import java.util.UUID

object Constants {
    const val NOTIFICATION_CHANNEL_ID = "grabdrop_service"
    const val NOTIFICATION_ID = 1001
    const val SERVICE_ACTION_STOP = "com.grabdrop.STOP_SERVICE"

    const val UDP_PORT = 9877
    const val MULTICAST_GROUP = "239.255.77.88"

    const val BROADCAST_TYPE_SCREENSHOT_READY = "SCREENSHOT_READY"

    const val SCREENSHOT_OFFER_TIMEOUT_MS = 10_000L
    const val GRAB_COOLDOWN_MS = 3_000L
    const val MOCK_RELEASE_DELAY_MS = 2_500L

    // Gesture detection
    const val IDLE_FPS = 10
    const val IDLE_FRAME_INTERVAL_MS = 1000L / IDLE_FPS
    const val WAKEUP_FRAME_INTERVAL_MS = 33L  // ~30fps
    const val IDLE_WINDOW_SIZE = 10
    const val IDLE_TRIGGER_THRESHOLD = 8
    const val WAKEUP_DURATION_MS = 2_000L
    const val WAKEUP_CONFIRM_FRAMES = 8

    const val FINGER_EXTENDED_THRESHOLD = 1.3f
    const val FINGER_CURLED_THRESHOLD = 0.9f
    const val MIN_FINGERS_FOR_PALM = 3
    const val MIN_FINGERS_FOR_FIST = 3

    // Swipe detection
    // Normalized displacement threshold (0-1 range, where 1 = full frame height)
    const val SWIPE_DISPLACEMENT_THRESHOLD = 0.12f
    // Minimum consecutive frames moving in same direction
    const val SWIPE_CONFIRM_FRAMES = 5
    // Minimum velocity (displacement per frame) to count as moving
    const val SWIPE_MIN_VELOCITY = 0.008f
    // Swipe cooldown (shorter than grab cooldown since swipe is repeatable)
    const val SWIPE_COOLDOWN_MS = 800L

    val DEVICE_ID: String = UUID.randomUUID().toString().substring(0, 8)
}
