// /GrabDrop/app/src/main/java/com/grabdrop/capture/ScreenCaptureManager.kt
package com.grabdrop.capture

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class ScreenCaptureManager(private val context: Context) {

    companion object {
        private const val TAG = "ScreenCapture"
    }

    private var mediaProjection: MediaProjection? = null
    private var projectionCallback: MediaProjection.Callback? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null

    private val handlerThread = HandlerThread("ScreenCaptureThread").apply { start() }
    private val handler = Handler(handlerThread.looper)
    private val mainHandler = Handler(Looper.getMainLooper())

    private var screenWidth = 0
    private var screenHeight = 0

    // The latest frame, continuously updated by the listener
    private val latestBitmap = AtomicReference<Bitmap?>(null)
    private val isCapturing = AtomicBoolean(false)
    private val isSetUp = AtomicBoolean(false)

    fun initProjection(resultCode: Int, data: Intent) {
        val mpm = context.getSystemService(Context.MEDIA_PROJECTION_SERVICE)
                as MediaProjectionManager
        mediaProjection = mpm.getMediaProjection(resultCode, data)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            projectionCallback = object : MediaProjection.Callback() {
                override fun onStop() {
                    Log.d(TAG, "MediaProjection stopped via callback")
                    tearDown()
                }
            }
            mediaProjection?.registerCallback(projectionCallback!!, mainHandler)
            Log.d(TAG, "MediaProjection callback registered (API 34+)")
        }

        Log.d(TAG, "MediaProjection initialized")
        setup()
    }

    private fun setup() {
        val projection = mediaProjection ?: return

        val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)

        screenWidth = metrics.widthPixels
        screenHeight = metrics.heightPixels
        val density = metrics.densityDpi

        Log.d(TAG, "Setting up: ${screenWidth}x${screenHeight} @ ${density}dpi")

        val reader = ImageReader.newInstance(
            screenWidth, screenHeight, PixelFormat.RGBA_8888, 2
        )
        imageReader = reader

        // Continuously listen for new frames
        reader.setOnImageAvailableListener({ ir ->
            // Skip if teardown in progress or not fully set up
            if (!isSetUp.get()) {
                ir.acquireLatestImage()?.close()
                return@setOnImageAvailableListener
            }
            
            val image = ir.acquireLatestImage() ?: return@setOnImageAvailableListener
            try {
                val bmp = imageToBitmap(image, screenWidth, screenHeight)
                if (bmp != null) {
                    // Swap in new bitmap, recycle old one
                    val old = latestBitmap.getAndSet(bmp)
                    old?.recycle()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame", e)
            } finally {
                image.close()
            }
        }, handler)

        virtualDisplay = projection.createVirtualDisplay(
            "GrabDropCapture",
            screenWidth, screenHeight, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface,
            object : VirtualDisplay.Callback() {
                override fun onStopped() {
                    Log.d(TAG, "VirtualDisplay stopped")
                    isSetUp.set(false)
                }
            },
            handler
        )

        isSetUp.set(true)
        Log.d(TAG, "Persistent VirtualDisplay created, listener active")
    }

    private fun tearDown() {
        isSetUp.set(false)

        // Clear listener FIRST to prevent new callbacks
        val reader = imageReader
        reader?.setOnImageAvailableListener(null, null)
        
        // Close reader before releasing virtual display to ensure
        // any pending images are properly handled
        reader?.close()
        imageReader = null
        
        virtualDisplay?.release()
        virtualDisplay = null

        latestBitmap.getAndSet(null)?.recycle()
    }

    /**
     * Capture is now instant — just copies the latest buffered frame.
     * No polling, no delays, no race conditions.
     */
    suspend fun capture(): Bitmap? = withContext(Dispatchers.IO) {
        if (!isSetUp.get()) {
            Log.w(TAG, "VirtualDisplay not set up, attempting re-setup")
            withContext(Dispatchers.Main) { setup() }
            // Wait briefly for first frame to arrive
            delay(500)
        }

        // Prevent concurrent captures
        if (!isCapturing.compareAndSet(false, true)) {
            Log.w(TAG, "Capture already in progress, skipping")
            return@withContext null
        }

        try {
            // If we already have a buffered frame, use it immediately
            var frame = latestBitmap.getAndSet(null)

            if (frame != null && !frame.isRecycled) {
                Log.d(TAG, "Instant capture from buffer: ${frame.width}x${frame.height}")
                // Return a copy so the buffer cycle doesn't recycle it under us
                val copy = frame.copy(frame.config, false)
                frame.recycle()
                return@withContext copy
            }

            // Rare case: no buffered frame yet. Wait briefly.
            Log.d(TAG, "No buffered frame, waiting...")
            var retries = 10
            while (retries > 0) {
                delay(100)
                frame = latestBitmap.getAndSet(null)
                if (frame != null && !frame.isRecycled) {
                    Log.d(TAG, "Got frame after ${11 - retries} waits")
                    val copy = frame.copy(frame.config, false)
                    frame.recycle()
                    return@withContext copy
                }
                retries--
            }

            Log.e(TAG, "No frame available after waiting")
            null
        } finally {
            isCapturing.set(false)
        }
    }

    private fun imageToBitmap(image: Image, targetWidth: Int, targetHeight: Int): Bitmap? {
        val planes = image.planes
        if (planes.isEmpty()) return null

        val buffer = planes[0].buffer
        if (!buffer.hasRemaining()) return null
        
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * targetWidth

        val bmpWidth = targetWidth + rowPadding / pixelStride
        val bmp = Bitmap.createBitmap(bmpWidth, targetHeight, Bitmap.Config.ARGB_8888)
        
        try {
            bmp.copyPixelsFromBuffer(buffer)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy pixels from buffer", e)
            bmp.recycle()
            return null
        }

        return if (bmpWidth != targetWidth) {
            Bitmap.createBitmap(bmp, 0, 0, targetWidth, targetHeight).also {
                if (it !== bmp) bmp.recycle()
            }
        } else {
            bmp
        }
    }

    fun release() {
        Log.d(TAG, "Releasing ScreenCaptureManager")

        tearDown()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            projectionCallback?.let {
                try {
                    mediaProjection?.unregisterCallback(it)
                } catch (e: Exception) {
                    Log.e(TAG, "Error unregistering callback", e)
                }
            }
        }
        projectionCallback = null

        mediaProjection?.stop()
        mediaProjection = null

        handlerThread.quitSafely()
    }
}
