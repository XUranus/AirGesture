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
import java.util.concurrent.atomic.AtomicBoolean

class OverlayManager(private val context: Context) {

    companion object {
        private const val TAG = "OverlayManager"
        private const val FLASH_DURATION = 250L
        private const val THUMBNAIL_SHOW_DURATION = 1800L
        private const val THUMBNAIL_FADE_DURATION = 250L
        private const val RIPPLE_DURATION = 500L
        private const val LOOPING_RIPPLE_DURATION = 800L
        private const val MAX_OVERLAY_LIFETIME = 5000L
    }

    private val windowManager =
        context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private val handler = Handler(Looper.getMainLooper())

    private val activeViews = CopyOnWriteArrayList<View>()
    private var wakeupIndicatorView: View? = null

    // Looping ripple state
    private val loopingRippleActive = AtomicBoolean(false)
    private var loopingRippleView: LoopingRippleView? = null

    private fun canDrawOverlay(): Boolean {
        return Settings.canDrawOverlays(context)
    }

    // =====================================================
    // WAKEUP INDICATOR
    // =====================================================

    fun showWakeupIndicator(emoji: String) {
        if (!canDrawOverlay()) return
        handler.post {
            try {
                hideWakeupIndicatorInternal()

                val density = context.resources.displayMetrics.density
                val topMargin = (density * 32).toInt()

                val textView = TextView(context).apply {
                    text = emoji
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 28f)
                    gravity = android.view.Gravity.CENTER
                    setBackgroundColor(Color.argb(180, 30, 30, 46))
                    setPadding(
                        (8 * density).toInt(), (4 * density).toInt(),
                        (8 * density).toInt(), (4 * density).toInt()
                    )
                    tag = System.currentTimeMillis()
                }

                textView.clipToOutline = true
                textView.outlineProvider = object : android.view.ViewOutlineProvider() {
                    override fun getOutline(view: View, outline: android.graphics.Outline) {
                        outline.setRoundRect(0, 0, view.width, view.height, 12 * density)
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

                ValueAnimator.ofFloat(0f, 1f).apply {
                    duration = 200
                    addUpdateListener { textView.alpha = it.animatedValue as Float }
                    start()
                }

                startPulse(textView)
            } catch (e: Exception) {
                Log.e(TAG, "Show indicator failed", e)
            }
        }
    }

    fun hideWakeupIndicator() {
        handler.post { hideWakeupIndicatorInternal() }
    }

    private fun hideWakeupIndicatorInternal() {
        wakeupIndicatorView?.let { view ->
            try {
                (view.getTag(android.R.id.content) as? ValueAnimator)?.cancel()
                if (view.isAttachedToWindow) windowManager.removeViewImmediate(view)
            } catch (e: Exception) {
                Log.e(TAG, "Hide indicator failed", e)
            }
            wakeupIndicatorView = null
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
        view.setTag(android.R.id.content, pulseAnimator)
    }

    // =====================================================
    // GRAB ANIMATION — flash + thumbnail
    // =====================================================

    fun showGrabAnimation(screenshotBitmap: Bitmap) {
        if (!canDrawOverlay()) return
        cleanupStaleOverlays()
        handler.post {
            try { showFlash() } catch (e: Exception) { Log.e(TAG, "Flash failed", e) }
        }
        handler.postDelayed({
            try {
                if (!screenshotBitmap.isRecycled) showThumbnail(screenshotBitmap)
            } catch (e: Exception) { Log.e(TAG, "Thumbnail failed", e) }
        }, FLASH_DURATION)
    }

    // =====================================================
    // RELEASE ANIMATION — single ripple
    // =====================================================

    fun showReleaseAnimation() {
        if (!canDrawOverlay()) return
        cleanupStaleOverlays()
        handler.post {
            try { showRipple() } catch (e: Exception) { Log.e(TAG, "Ripple failed", e) }
        }
    }

    // =====================================================
    // LOOPING RIPPLE — repeats until stopLoopingRipple()
    // =====================================================

    fun startLoopingRipple() {
        if (!canDrawOverlay()) return
        if (loopingRippleActive.getAndSet(true)) return  // already running

        handler.post {
            try {
                val rippleView = LoopingRippleView(context)
                rippleView.tag = System.currentTimeMillis()

                val params = createOverlayParams(
                    WindowManager.LayoutParams.MATCH_PARENT,
                    WindowManager.LayoutParams.MATCH_PARENT
                )

                windowManager.addView(rippleView, params)
                loopingRippleView = rippleView
                activeViews.add(rippleView)

                rippleView.startLooping(LOOPING_RIPPLE_DURATION)
                Log.d(TAG, "Looping ripple started")
            } catch (e: Exception) {
                Log.e(TAG, "Looping ripple start failed", e)
                loopingRippleActive.set(false)
            }
        }
    }

    fun stopLoopingRipple() {
        if (!loopingRippleActive.getAndSet(false)) return  // not running

        handler.post {
            loopingRippleView?.let { view ->
                try {
                    view.stopLooping()
                    // Fade out over 300ms
                    ValueAnimator.ofFloat(1f, 0f).apply {
                        duration = 300
                        addUpdateListener { view.alpha = it.animatedValue as Float }
                        addListener(object : AnimatorListenerAdapter() {
                            override fun onAnimationEnd(animation: Animator) {
                                removeOverlayView(view)
                            }
                        })
                        start()
                    }
                } catch (e: Exception) {
                    removeOverlayView(view)
                    Log.e(TAG, "Looping ripple stop failed", e)
                }
                loopingRippleView = null
            }
            Log.d(TAG, "Looping ripple stopped")
        }
    }

    // =====================================================
    // CLEANUP
    // =====================================================

    fun removeAll() {
        handler.post {
            hideWakeupIndicatorInternal()
            loopingRippleActive.set(false)
            loopingRippleView?.stopLooping()
            loopingRippleView = null
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
                override fun onAnimationEnd(animation: Animator) { removeOverlayView(flashView) }
            })
            start()
        }
        scheduleForceRemoval(flashView, FLASH_DURATION + 500)
    }

    // --- Thumbnail ---

    private fun showThumbnail(bitmap: Bitmap) {
        if (bitmap.isRecycled) return

        val density = context.resources.displayMetrics.density
        val thumbSize = (density * 120).toInt()
        val margin = (density * 16).toInt()
        val scaledHeight = (thumbSize * bitmap.height.toFloat() / bitmap.width).toInt().coerceAtLeast(1)
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
        container.addView(imageView, FrameLayout.LayoutParams(thumbSize, FrameLayout.LayoutParams.WRAP_CONTENT))

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
            x = margin; y = margin
        }

        container.alpha = 0f; container.scaleX = 0.3f; container.scaleY = 0.3f
        addOverlayView(container, params)

        ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 250
            addUpdateListener {
                val f = it.animatedValue as Float
                container.alpha = f
                val s = 0.3f + 0.7f * f
                container.scaleX = s; container.scaleY = s
            }
            start()
        }

        handler.postDelayed({ dismissThumbnail(container, imageView, scaledBitmap) }, THUMBNAIL_SHOW_DURATION)
        scheduleForceRemoval(container, THUMBNAIL_SHOW_DURATION + THUMBNAIL_FADE_DURATION + 1000) {
            imageView.setImageDrawable(null)
            recycleSafely(scaledBitmap)
        }
    }

    private fun dismissThumbnail(container: FrameLayout, imageView: ImageView, scaledBitmap: Bitmap) {
        if (!container.isAttachedToWindow) { recycleSafely(scaledBitmap); return }
        try {
            ValueAnimator.ofFloat(1f, 0f).apply {
                duration = THUMBNAIL_FADE_DURATION
                addUpdateListener {
                    val f = it.animatedValue as Float
                    container.alpha = f
                    val s = 0.5f + 0.5f * f
                    container.scaleX = s; container.scaleY = s
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
            imageView.setImageDrawable(null)
            removeOverlayView(container)
            recycleSafely(scaledBitmap)
        }
    }

    // --- Single Ripple ---

    private fun showRipple() {
        val rippleView = SingleRippleView(context).apply { tag = System.currentTimeMillis() }
        val params = createOverlayParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT
        )
        addOverlayView(rippleView, params)
        rippleView.startRipple(RIPPLE_DURATION) { removeOverlayView(rippleView) }
        scheduleForceRemoval(rippleView, RIPPLE_DURATION + 1000)
    }

    // --- View Management ---

    private fun addOverlayView(view: View, params: WindowManager.LayoutParams) {
        try {
            windowManager.addView(view, params)
            activeViews.add(view)
        } catch (e: Exception) { Log.e(TAG, "addOverlayView error", e) }
    }

    private fun removeOverlayView(view: View) {
        try { if (view.isAttachedToWindow) windowManager.removeViewImmediate(view) }
        catch (e: Exception) { Log.e(TAG, "removeOverlayView error", e) }
        activeViews.remove(view)
    }

    private fun forceRemoveView(view: View) {
        try {
            (view.getTag(android.R.id.content) as? ValueAnimator)?.cancel()
            if (view.isAttachedToWindow) {
                if (view is FrameLayout) {
                    for (i in 0 until view.childCount) {
                        (view.getChildAt(i) as? ImageView)?.setImageDrawable(null)
                    }
                }
                windowManager.removeViewImmediate(view)
            }
        } catch (e: Exception) { Log.e(TAG, "forceRemoveView error", e) }
        activeViews.remove(view)
    }

    private fun scheduleForceRemoval(view: View, delayMs: Long, cleanup: (() -> Unit)? = null) {
        handler.postDelayed({
            if (activeViews.contains(view)) {
                cleanup?.invoke()
                forceRemoveView(view)
            }
        }, delayMs)
    }

    private fun cleanupStaleOverlays() {
        val now = System.currentTimeMillis()
        activeViews.filter { v ->
            val created = (v.tag as? Long) ?: 0L
            now - created > MAX_OVERLAY_LIFETIME && v !== loopingRippleView
        }.forEach { forceRemoveView(it) }
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
        try { if (bitmap != null && !bitmap.isRecycled) bitmap.recycle() }
        catch (_: Exception) {}
    }

    // --- Single Ripple View ---

    private class SingleRippleView(context: Context) : View(context) {
        private val paint1 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(120, 79, 195, 247); style = Paint.Style.STROKE; strokeWidth = 8f
        }
        private val paint2 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(60, 79, 195, 247); style = Paint.Style.STROKE; strokeWidth = 8f
        }
        private var rippleRadius = 0f
        private var fraction = 0f
        private var animator: ValueAnimator? = null

        fun startRipple(durationMs: Long, onComplete: () -> Unit) {
            post {
                val maxR = Math.hypot((width / 2).toDouble(), (height / 2).toDouble())
                    .toFloat().coerceAtLeast(800f)
                animator = ValueAnimator.ofFloat(0f, 1f).apply {
                    duration = durationMs
                    addUpdateListener { a ->
                        fraction = a.animatedValue as Float
                        rippleRadius = maxR * fraction
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
            val cx = width / 2f; val cy = height / 2f
            paint1.alpha = (120 * (1f - fraction)).toInt()
            paint1.strokeWidth = 8f + 20f * fraction
            canvas.drawCircle(cx, cy, rippleRadius, paint1)
            if (rippleRadius > 80f) {
                paint2.alpha = (paint1.alpha * 0.5f).toInt()
                paint2.strokeWidth = paint1.strokeWidth
                canvas.drawCircle(cx, cy, rippleRadius - 80f, paint2)
            }
        }

        override fun onDetachedFromWindow() { super.onDetachedFromWindow(); animator?.cancel() }
    }

    // --- Looping Ripple View ---

    private class LoopingRippleView(context: Context) : View(context) {
        private val paint1 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(100, 79, 195, 247); style = Paint.Style.STROKE; strokeWidth = 6f
        }
        private val paint2 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(60, 79, 195, 247); style = Paint.Style.STROKE; strokeWidth = 6f
        }
        private val paint3 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(40, 79, 195, 247); style = Paint.Style.STROKE; strokeWidth = 6f
        }

        // Multiple ripple rings at different phases
        private var fraction1 = 0f
        private var fraction2 = 0f
        private var fraction3 = 0f
        private var animator: ValueAnimator? = null
        private var isLooping = false

        fun startLooping(cycleDurationMs: Long) {
            isLooping = true
            post {
                val maxR = Math.hypot((width / 2).toDouble(), (height / 2).toDouble())
                    .toFloat().coerceAtLeast(800f)

                animator = ValueAnimator.ofFloat(0f, 1f).apply {
                    duration = cycleDurationMs
                    repeatCount = ValueAnimator.INFINITE
                    addUpdateListener { a ->
                        if (!isLooping) { cancel(); return@addUpdateListener }
                        val base = a.animatedValue as Float
                        fraction1 = base
                        fraction2 = (base + 0.33f) % 1f
                        fraction3 = (base + 0.66f) % 1f
                        invalidate()
                    }
                    start()
                }
            }
        }

        fun stopLooping() {
            isLooping = false
            animator?.cancel()
            animator = null
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            if (!isLooping) return

            val cx = width / 2f; val cy = height / 2f
            val maxR = Math.hypot((width / 2).toDouble(), (height / 2).toDouble())
                .toFloat().coerceAtLeast(400f)

            drawRing(canvas, cx, cy, maxR, fraction1, paint1)
            drawRing(canvas, cx, cy, maxR, fraction2, paint2)
            drawRing(canvas, cx, cy, maxR, fraction3, paint3)
        }

        private fun drawRing(canvas: Canvas, cx: Float, cy: Float, maxR: Float, fraction: Float, paint: Paint) {
            val r = maxR * fraction
            if (r <= 0f) return
            paint.alpha = (100 * (1f - fraction)).toInt()
            paint.strokeWidth = 4f + 12f * fraction
            canvas.drawCircle(cx, cy, r, paint)
        }

        override fun onDetachedFromWindow() {
            super.onDetachedFromWindow()
            stopLooping()
        }
    }



    fun showSwipeIndicator(isUp: Boolean) {
        if (!canDrawOverlay()) return
        handler.post {
            try {
                val density = context.resources.displayMetrics.density
                val arrowView = TextView(context).apply {
                    text = if (isUp) "⬆️" else "⬇️"
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 48f)
                    gravity = android.view.Gravity.CENTER
                    alpha = 0.8f
                    tag = System.currentTimeMillis()
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
                    gravity = Gravity.CENTER
                }

                addOverlayView(arrowView, params)

                // Animate: slide in direction + fade out
                val slideDistance = (200 * density).toInt()
                ValueAnimator.ofFloat(0f, 1f).apply {
                    duration = 500
                    addUpdateListener {
                        val f = it.animatedValue as Float
                        arrowView.alpha = 0.8f * (1f - f)
                        arrowView.translationY = if (isUp) -slideDistance * f else slideDistance * f
                        arrowView.scaleX = 1f + 0.3f * f
                        arrowView.scaleY = 1f + 0.3f * f
                    }
                    addListener(object : AnimatorListenerAdapter() {
                        override fun onAnimationEnd(animation: Animator) {
                            removeOverlayView(arrowView)
                        }
                    })
                    start()
                }

                scheduleForceRemoval(arrowView, 1000)
            } catch (e: Exception) {
                Log.e(TAG, "Swipe indicator failed", e)
            }
        }
    }

}
