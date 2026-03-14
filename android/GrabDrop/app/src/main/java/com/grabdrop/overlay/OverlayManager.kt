// /GrabDrop/app/src/main/java/com/grabdrop/overlay/OverlayManager.kt
package com.grabdrop.overlay

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import java.util.concurrent.CopyOnWriteArrayList

class OverlayManager(private val context: Context) {

    companion object {
        private const val TAG = "OverlayManager"
        private const val FLASH_DURATION = 250L
        private const val THUMBNAIL_SHOW_DURATION = 1800L
        private const val THUMBNAIL_FADE_DURATION = 250L
        private const val RIPPLE_DURATION = 500L
        private const val MAX_OVERLAY_LIFETIME = 5000L
    }

    private val windowManager =
        context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private val handler = Handler(Looper.getMainLooper())

    private val activeViews = CopyOnWriteArrayList<View>()

    // Persistent wakeup indicator view
    private var wakeupIndicatorView: View? = null

    private fun canDrawOverlay(): Boolean {
        return Settings.canDrawOverlays(context)
    }

    // =====================================================
    // WAKEUP INDICATOR — persistent small icon at top center
    // =====================================================

    fun showWakeupIndicator(emoji: String) {
        if (!canDrawOverlay()) return

        handler.post {
            try {
                // Remove previous indicator if any
                hideWakeupIndicatorInternal()

                val density = context.resources.displayMetrics.density
                val size = (density * 48).toInt()
                val topMargin = (density * 32).toInt()

                val textView = TextView(context).apply {
                    text = emoji
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 28f)
                    gravity = android.view.Gravity.CENTER
                    setBackgroundColor(Color.argb(180, 30, 30, 46))
                    setPadding(
                        (8 * density).toInt(),
                        (4 * density).toInt(),
                        (8 * density).toInt(),
                        (4 * density).toInt()
                    )
                    tag = System.currentTimeMillis()
                }

                // Round corners via outline
                textView.clipToOutline = true
                textView.outlineProvider = object : android.view.ViewOutlineProvider() {
                    override fun getOutline(view: View, outline: android.graphics.Outline) {
                        outline.setRoundRect(
                            0, 0, view.width, view.height,
                            12 * density
                        )
                    }
                }

                val params = WindowManager.LayoutParams(
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                    PixelFormat.TRANSLUCENT
                ).apply {
                    gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                    y = topMargin
                }

                textView.alpha = 0f
                windowManager.addView(textView, params)
                wakeupIndicatorView = textView

                // Fade in with pulse
                ValueAnimator.ofFloat(0f, 1f).apply {
                    duration = 200
                    addUpdateListener { textView.alpha = it.animatedValue as Float }
                    start()
                }

                // Pulse animation
                startPulse(textView)

                Log.d(TAG, "Wakeup indicator shown: $emoji")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to show wakeup indicator", e)
            }
        }
    }

    fun hideWakeupIndicator() {
        handler.post { hideWakeupIndicatorInternal() }
    }

    private fun hideWakeupIndicatorInternal() {
        wakeupIndicatorView?.let { view ->
            try {
                if (view.isAttachedToWindow) {
                    windowManager.removeViewImmediate(view)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to remove wakeup indicator", e)
            }
            wakeupIndicatorView = null
            Log.d(TAG, "Wakeup indicator hidden")
        }
    }

    private fun startPulse(view: View) {
        val pulseAnimator = ValueAnimator.ofFloat(1f, 1.15f, 1f).apply {
            duration = 800
            repeatCount = ValueAnimator.INFINITE
            addUpdateListener {
                if (view.isAttachedToWindow) {
                    val scale = it.animatedValue as Float
                    view.scaleX = scale
                    view.scaleY = scale
                } else {
                    cancel()
                }
            }
            start()
        }
        // Store animator reference for cleanup in view tag
        view.setTag(android.R.id.content, pulseAnimator)
    }

    // =====================================================
    // GRAB ANIMATION — flash + thumbnail
    // =====================================================

    fun showGrabAnimation(screenshotBitmap: Bitmap) {
        if (!canDrawOverlay()) {
            Log.w(TAG, "Cannot draw overlays — permission not granted")
            return
        }
        cleanupStaleOverlays()
        handler.post {
            try {
                showFlash()
            } catch (e: Exception) {
                Log.e(TAG, "Flash failed", e)
            }
        }
        handler.postDelayed({
            try {
                if (!screenshotBitmap.isRecycled) {
                    showThumbnail(screenshotBitmap)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Thumbnail failed", e)
            }
        }, FLASH_DURATION)
    }

    // =====================================================
    // RELEASE ANIMATION — ripple
    // =====================================================

    fun showReleaseAnimation() {
        if (!canDrawOverlay()) {
            Log.w(TAG, "Cannot draw overlays — permission not granted")
            return
        }
        cleanupStaleOverlays()
        handler.post {
            try {
                showRipple()
            } catch (e: Exception) {
                Log.e(TAG, "Ripple failed", e)
            }
        }
    }

    fun removeAll() {
        handler.post {
            hideWakeupIndicatorInternal()
            for (view in activeViews) {
                forceRemoveView(view)
            }
            activeViews.clear()
        }
    }

    // --- Flash Effect ---

    private fun showFlash() {
        val flashView = View(context).apply {
            setBackgroundColor(Color.WHITE)
            alpha = 0.8f
            tag = System.currentTimeMillis()
        }

        val params = createOverlayParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT
        )

        addOverlayView(flashView, params)

        ValueAnimator.ofFloat(0.8f, 0f).apply {
            duration = FLASH_DURATION
            addUpdateListener { flashView.alpha = it.animatedValue as Float }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    removeOverlayView(flashView)
                }
            })
            start()
        }

        scheduleForceRemoval(flashView, FLASH_DURATION + 500)
    }

    // --- Thumbnail Preview ---

    private fun showThumbnail(bitmap: Bitmap) {
        if (bitmap.isRecycled) {
            Log.w(TAG, "Bitmap recycled, skipping thumbnail")
            return
        }

        val density = context.resources.displayMetrics.density
        val thumbSize = (density * 120).toInt()
        val margin = (density * 16).toInt()

        val scaledHeight = (thumbSize * bitmap.height.toFloat() / bitmap.width).toInt()
            .coerceAtLeast(1)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, thumbSize, scaledHeight, true)

        val imageView = ImageView(context).apply {
            setImageBitmap(scaledBitmap)
            scaleType = ImageView.ScaleType.FIT_CENTER
        }

        val container = FrameLayout(context).apply {
            setPadding(4, 4, 4, 4)
            setBackgroundColor(Color.WHITE)
            tag = System.currentTimeMillis()
        }
        container.addView(
            imageView,
            FrameLayout.LayoutParams(thumbSize, FrameLayout.LayoutParams.WRAP_CONTENT)
        )

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.START
            x = margin
            y = margin
        }

        container.alpha = 0f
        container.scaleX = 0.3f
        container.scaleY = 0.3f

        addOverlayView(container, params)

        ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 250
            addUpdateListener {
                val fraction = it.animatedValue as Float
                container.alpha = fraction
                val scale = 0.3f + 0.7f * fraction
                container.scaleX = scale
                container.scaleY = scale
            }
            start()
        }

        handler.postDelayed({
            dismissThumbnail(container, imageView, scaledBitmap)
        }, THUMBNAIL_SHOW_DURATION)

        scheduleForceRemoval(container, THUMBNAIL_SHOW_DURATION + THUMBNAIL_FADE_DURATION + 1000) {
            imageView.setImageDrawable(null)
            recycleSafely(scaledBitmap)
        }
    }

    private fun dismissThumbnail(
        container: FrameLayout,
        imageView: ImageView,
        scaledBitmap: Bitmap
    ) {
        if (!container.isAttachedToWindow) {
            recycleSafely(scaledBitmap)
            return
        }

        try {
            ValueAnimator.ofFloat(1f, 0f).apply {
                duration = THUMBNAIL_FADE_DURATION
                addUpdateListener {
                    val fraction = it.animatedValue as Float
                    container.alpha = fraction
                    val scale = 0.5f + 0.5f * fraction
                    container.scaleX = scale
                    container.scaleY = scale
                }
                addListener(object : AnimatorListenerAdapter() {
                    override fun onAnimationEnd(animation: Animator) {
                        imageView.setImageDrawable(null)
                        removeOverlayView(container)
                        handler.post { recycleSafely(scaledBitmap) }
                    }
                })
                start()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Thumbnail dismiss error", e)
            imageView.setImageDrawable(null)
            removeOverlayView(container)
            recycleSafely(scaledBitmap)
        }
    }

    // --- Ripple Effect ---

    private fun showRipple() {
        val rippleView = RippleView(context).apply {
            tag = System.currentTimeMillis()
        }

        val params = createOverlayParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT
        )

        addOverlayView(rippleView, params)

        rippleView.startRipple(RIPPLE_DURATION) {
            removeOverlayView(rippleView)
        }

        scheduleForceRemoval(rippleView, RIPPLE_DURATION + 1000)
    }

    // --- View Management ---

    private fun addOverlayView(view: View, params: WindowManager.LayoutParams) {
        try {
            windowManager.addView(view, params)
            activeViews.add(view)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add overlay", e)
        }
    }

    private fun removeOverlayView(view: View) {
        try {
            if (view.isAttachedToWindow) {
                windowManager.removeViewImmediate(view)
            }
        } catch (e: Exception) {
            Log.e(TAG, "removeOverlayView error", e)
        }
        activeViews.remove(view)
    }

    private fun forceRemoveView(view: View) {
        try {
            // Cancel any pulse animator
            (view.getTag(android.R.id.content) as? ValueAnimator)?.cancel()

            if (view.isAttachedToWindow) {
                if (view is FrameLayout) {
                    for (i in 0 until view.childCount) {
                        val child = view.getChildAt(i)
                        if (child is ImageView) child.setImageDrawable(null)
                    }
                }
                windowManager.removeViewImmediate(view)
            }
        } catch (e: Exception) {
            Log.e(TAG, "forceRemoveView error", e)
        }
        activeViews.remove(view)
    }

    private fun scheduleForceRemoval(view: View, delayMs: Long, cleanup: (() -> Unit)? = null) {
        handler.postDelayed({
            if (activeViews.contains(view)) {
                Log.w(TAG, "Force removing stuck overlay")
                cleanup?.invoke()
                forceRemoveView(view)
            }
        }, delayMs)
    }

    private fun cleanupStaleOverlays() {
        val now = System.currentTimeMillis()
        val stale = activeViews.filter { view ->
            val created = (view.tag as? Long) ?: 0L
            now - created > MAX_OVERLAY_LIFETIME
        }
        for (view in stale) {
            forceRemoveView(view)
        }
    }

    private fun createOverlayParams(width: Int, height: Int): WindowManager.LayoutParams {
        return WindowManager.LayoutParams(
            width, height,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        )
    }

    private fun recycleSafely(bitmap: Bitmap?) {
        try {
            if (bitmap != null && !bitmap.isRecycled) bitmap.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "recycle error", e)
        }
    }

    // --- Ripple View ---

    private class RippleView(context: Context) : View(context) {
        private val paint1 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(120, 79, 195, 247)
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }
        private val paint2 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(60, 79, 195, 247)
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }

        private var rippleRadius = 0f
        private var fraction = 0f
        private var animator: ValueAnimator? = null

        fun startRipple(durationMs: Long, onComplete: () -> Unit) {
            post {
                val maxRadius = Math.hypot(
                    (width / 2).toDouble(),
                    (height / 2).toDouble()
                ).toFloat().coerceAtLeast(800f)

                animator = ValueAnimator.ofFloat(0f, 1f).apply {
                    duration = durationMs
                    addUpdateListener { anim ->
                        fraction = anim.animatedValue as Float
                        rippleRadius = maxRadius * fraction
                        invalidate()
                    }
                    addListener(object : AnimatorListenerAdapter() {
                        override fun onAnimationEnd(animation: Animator) {
                            try { onComplete() } catch (_: Exception) {}
                        }
                    })
                    start()
                }
            }
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            if (rippleRadius <= 0f) return
            val cx = width / 2f
            val cy = height / 2f
            val alpha1 = (120 * (1f - fraction)).toInt()
            val sw1 = 8f + 20f * fraction
            paint1.alpha = alpha1
            paint1.strokeWidth = sw1
            canvas.drawCircle(cx, cy, rippleRadius, paint1)
            if (rippleRadius > 80f) {
                paint2.alpha = (alpha1 * 0.5f).toInt()
                paint2.strokeWidth = sw1
                canvas.drawCircle(cx, cy, rippleRadius - 80f, paint2)
            }
        }

        override fun onDetachedFromWindow() {
            super.onDetachedFromWindow()
            animator?.cancel()
            animator = null
        }
    }
}
