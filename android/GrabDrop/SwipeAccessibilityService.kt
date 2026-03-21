// /GrabDrop/app/src/main/java/com/grabdrop/service/SwipeAccessibilityService.kt
package com.grabdrop.service

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.os.Build
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent

class SwipeAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "SwipeA11y"

        @Volatile
        var instance: SwipeAccessibilityService? = null
            private set

        val isRunning: Boolean get() = instance != null

        fun performSwipe(direction: SwipeDirection) {
            instance?.doSwipe(direction) ?: Log.w(TAG, "Service not running")
        }
    }

    enum class SwipeDirection { UP, DOWN }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.d(TAG, "Accessibility service connected")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // Not needed — we only use this service for gesture dispatch
    }

    override fun onInterrupt() {
        Log.d(TAG, "Accessibility service interrupted")
    }

    override fun onDestroy() {
        instance = null
        Log.d(TAG, "Accessibility service destroyed")
        super.onDestroy()
    }

    private fun doSwipe(direction: SwipeDirection) {
        val wm = getSystemService(WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)

        val screenWidth = metrics.widthPixels
        val screenHeight = metrics.heightPixels

        val centerX = screenWidth / 2f
        // Swipe in the middle 60% of the screen to avoid edge gestures
        val swipeLength = screenHeight * 0.35f

        val startY: Float
        val endY: Float

        when (direction) {
            SwipeDirection.UP -> {
                // Finger moves up = content scrolls up = swipe from bottom to top
                startY = screenHeight * 0.65f
                endY = startY - swipeLength
            }
            SwipeDirection.DOWN -> {
                // Finger moves down = content scrolls down = swipe from top to bottom
                startY = screenHeight * 0.35f
                endY = startY + swipeLength
            }
        }

        val path = Path().apply {
            moveTo(centerX, startY)
            lineTo(centerX, endY)
        }

        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, 250))
            .build()

        val success = dispatchGesture(gesture, object : GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription?) {
                Log.d(TAG, "Swipe $direction completed")
            }

            override fun onCancelled(gestureDescription: GestureDescription?) {
                Log.w(TAG, "Swipe $direction cancelled")
            }
        }, null)

        Log.d(TAG, "Swipe $direction dispatched: success=$success")
    }
}
