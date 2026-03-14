// /GrabDrop/app/src/main/java/com/grabdrop/service/GrabDropService.kt
package com.grabdrop.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.grabdrop.R
import com.grabdrop.capture.MediaStoreHelper
import com.grabdrop.capture.ScreenCaptureManager
import com.grabdrop.gesture.GestureEvent
import com.grabdrop.gesture.RealGestureDetector
import com.grabdrop.network.NetworkManager
import com.grabdrop.network.ScreenshotOffer
import com.grabdrop.overlay.OverlayManager
import com.grabdrop.ui.MainActivity
import com.grabdrop.util.Constants
import com.grabdrop.util.SoundPlayer
import kotlinx.coroutines.*
import java.text.SimpleDateFormat
import java.util.*

class GrabDropService : Service() {

    companion object {
        private const val TAG = "GrabDropService"
    }

    private val serviceScope = CoroutineScope(
        SupervisorJob() + Dispatchers.Main +
                CoroutineExceptionHandler { _, throwable ->
                    Log.e(TAG, "Coroutine exception caught", throwable)
                    ServiceState.addEvent(formatTime() + " ⚠️ Error: ${throwable.message}")
                }
    )

    private lateinit var screenCaptureManager: ScreenCaptureManager
    private lateinit var overlayManager: OverlayManager
    private lateinit var gestureDetector: RealGestureDetector
    private lateinit var networkManager: NetworkManager
    private lateinit var soundPlayer: SoundPlayer

    private var wakeLock: PowerManager.WakeLock? = null
    private var lastGrabTime = 0L
    private var pendingOffer: ScreenshotOffer? = null
    private var defaultUncaughtHandler: Thread.UncaughtExceptionHandler? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "onCreate called")

        defaultUncaughtHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            Log.e(TAG, "UNCAUGHT EXCEPTION on thread ${thread.name}", throwable)
            ServiceState.addEvent(formatTime() + " 💀 CRASH: ${throwable.message}")
            defaultUncaughtHandler?.uncaughtException(thread, throwable)
        }

        screenCaptureManager = ScreenCaptureManager(this)
        overlayManager = OverlayManager(this)
        // Pass overlayManager to gesture detector for wakeup indicator
        gestureDetector = RealGestureDetector(this, overlayManager)
        networkManager = NetworkManager(this)
        soundPlayer = SoundPlayer(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand called, action=${intent?.action}")

        when (intent?.action) {
            Constants.SERVICE_ACTION_STOP -> {
                Log.d(TAG, "Stop action received")
                stopSelf()
                return START_NOT_STICKY
            }
        }

        try {
            val notification = buildNotification()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(
                    Constants.NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or
                            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
                )
            } else {
                startForeground(Constants.NOTIFICATION_ID, notification)
            }
            Log.d(TAG, "startForeground succeeded")
        } catch (e: Exception) {
            Log.e(TAG, "startForeground FAILED", e)
            stopSelf()
            return START_NOT_STICKY
        }

        if (!MediaProjectionHolder.isAvailable) {
            Log.e(TAG, "No MediaProjection data in holder!")
            ServiceState.setRunning(true)
            ServiceState.addEvent(formatTime() + " ⚠️ Started WITHOUT screen capture")
            acquireWakeLock()
            startGestureAndNetwork()
            return START_STICKY
        }

        val (resultCode, resultData) = MediaProjectionHolder.consume()
        Log.d(TAG, "MediaProjectionHolder: resultCode=$resultCode, hasData=${resultData != null}")

        try {
            screenCaptureManager.initProjection(resultCode, resultData!!)
            Log.d(TAG, "MediaProjection initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "MediaProjection init FAILED", e)
            ServiceState.addEvent(formatTime() + " ⚠️ Screen capture init failed: ${e.message}")
        }

        acquireWakeLock()
        startGestureAndNetwork()

        ServiceState.setRunning(true)
        ServiceState.addEvent(formatTime() + " ✅ Service started with screen capture + gesture detection")

        return START_STICKY
    }

    override fun onDestroy() {
        Log.d(TAG, "onDestroy called")
        stopAll()
        defaultUncaughtHandler?.let {
            Thread.setDefaultUncaughtExceptionHandler(it)
        }
        ServiceState.setRunning(false)
        ServiceState.addEvent(formatTime() + " Service stopped")
        super.onDestroy()
    }

    private fun startGestureAndNetwork() {
        Log.d(TAG, "Starting gesture detector and network manager")
        networkManager.start()
        gestureDetector.start()

        serviceScope.launch {
            gestureDetector.events.collect { event ->
                Log.d(TAG, "Gesture event received: $event")
                try {
                    when (event) {
                        is GestureEvent.Grab -> handleGrab()
                        is GestureEvent.Release -> handleRelease()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error handling gesture event", e)
                    ServiceState.addEvent(formatTime() + " ⚠️ Error: ${e.message}")
                }
            }
        }

        serviceScope.launch {
            networkManager.incomingOffers.collect { offer ->
                try { handleIncomingOffer(offer) }
                catch (e: Exception) { Log.e(TAG, "Error handling offer", e) }
            }
        }

        serviceScope.launch {
            networkManager.nearbyDeviceCount.collect { count ->
                Log.d(TAG, "Nearby device count updated: $count")
                ServiceState.setNearbyDevices(count)
            }
        }

        Log.d(TAG, "All components started")
    }

    private fun stopAll() {
        Log.d(TAG, "Stopping all components")
        try { gestureDetector.stop() } catch (e: Exception) { Log.e(TAG, "gesture stop", e) }
        try { networkManager.stop() } catch (e: Exception) { Log.e(TAG, "network stop", e) }
        try { screenCaptureManager.release() } catch (e: Exception) { Log.e(TAG, "capture stop", e) }
        try { soundPlayer.release() } catch (e: Exception) { Log.e(TAG, "sound stop", e) }
        try { overlayManager.removeAll() } catch (e: Exception) { Log.e(TAG, "overlay stop", e) }
        releaseWakeLock()
        serviceScope.cancel()
    }

    private suspend fun handleGrab() {
        val now = System.currentTimeMillis()
        if (now - lastGrabTime < Constants.GRAB_COOLDOWN_MS) {
            Log.d(TAG, "Grab ignored (cooldown)")
            return
        }
        lastGrabTime = now

        ServiceState.setStatus("Taking screenshot...")
        ServiceState.addEvent(formatTime() + " ✊ GRAB detected")

        val bitmap: Bitmap?
        try {
            bitmap = screenCaptureManager.capture()
        } catch (e: Exception) {
            Log.e(TAG, "Screenshot capture exception", e)
            ServiceState.addEvent(formatTime() + " ❌ Screenshot failed: ${e.message}")
            ServiceState.setStatus("Monitoring gestures...")
            return
        }

        if (bitmap == null) {
            Log.e(TAG, "Screenshot capture returned null")
            ServiceState.addEvent(formatTime() + " ❌ Screenshot returned null")
            ServiceState.setStatus("Monitoring gestures...")
            return
        }

        Log.d(TAG, "Screenshot captured: ${bitmap.width}x${bitmap.height}")

        try {
            val uri = MediaStoreHelper.saveBitmap(this@GrabDropService, bitmap, "GrabDrop_Sent")
            Log.d(TAG, "Screenshot saved: $uri")
        } catch (e: Exception) { Log.e(TAG, "Save failed", e) }

        try { overlayManager.showGrabAnimation(bitmap) } catch (e: Exception) { Log.e(TAG, "Overlay failed", e) }
        try { soundPlayer.playShutter() } catch (e: Exception) { Log.e(TAG, "Sound failed", e) }

        try {
            val imageData = MediaStoreHelper.bitmapToByteArray(bitmap)
            networkManager.broadcastScreenshotAvailable(imageData)
        } catch (e: Exception) { Log.e(TAG, "Broadcast failed", e) }

        ServiceState.addEvent(formatTime() + " 📸 Screenshot captured & broadcast sent")
        ServiceState.setStatus("Monitoring gestures...")
    }

    private suspend fun handleRelease() {
        val offer = pendingOffer
        if (offer == null) {
            ServiceState.addEvent(formatTime() + " 🤚 RELEASE detected (no pending offer)")
            return
        }

        if (System.currentTimeMillis() - offer.timestamp > Constants.SCREENSHOT_OFFER_TIMEOUT_MS) {
            pendingOffer = null
            return
        }
        pendingOffer = null

        ServiceState.setStatus("Receiving screenshot...")
        ServiceState.addEvent(formatTime() + " 🤚 RELEASE — downloading from ${offer.senderName}")

        val data = try { networkManager.downloadScreenshot(offer) }
        catch (e: Exception) { Log.e(TAG, "Download exception", e); null }

        if (data == null) {
            ServiceState.addEvent(formatTime() + " ❌ Download failed")
            ServiceState.setStatus("Monitoring gestures...")
            return
        }

        var bitmap: Bitmap? = null
        try {
            bitmap = MediaStoreHelper.byteArrayToBitmap(data)
            val uri = MediaStoreHelper.saveBitmap(this@GrabDropService, bitmap, "GrabDrop_Received")
            try { overlayManager.showReleaseAnimation() } catch (_: Exception) {}
            try { soundPlayer.playReceive() } catch (_: Exception) {}
            if (uri != null) {
                try {
                    startActivity(Intent(Intent.ACTION_VIEW).apply {
                        setDataAndType(uri, "image/png")
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    })
                } catch (_: Exception) {}
            }
            ServiceState.addEvent(formatTime() + " 📥 Screenshot received & saved")
        } catch (e: Exception) {
            ServiceState.addEvent(formatTime() + " ❌ Processing failed: ${e.message}")
        } finally {
            bitmap?.recycle()
        }
        ServiceState.setStatus("Monitoring gestures...")
    }

    private fun handleIncomingOffer(offer: ScreenshotOffer) {
        Log.d(TAG, "Incoming offer from ${offer.senderName}")
        pendingOffer = offer
        ServiceState.addEvent(
            formatTime() + " 📡 Offer from ${offer.senderName} — do RELEASE to receive"
        )
        // On receiving device, auto-trigger release for testing
        // Remove this line for production — the real gesture handles it
        gestureDetector.triggerMockRelease(Constants.MOCK_RELEASE_DELAY_MS)
    }

    private fun buildNotification(): Notification {
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val openPending = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stopIntent = Intent(this, GrabDropService::class.java).apply {
            action = Constants.SERVICE_ACTION_STOP
        }
        val stopPending = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, Constants.NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText("Monitoring gestures • Tap to open")
            .setSmallIcon(R.drawable.ic_notification)
            .setOngoing(true)
            .setContentIntent(openPending)
            .addAction(0, getString(R.string.stop_service), stopPending)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    private fun acquireWakeLock() {
        try {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "GrabDrop::ServiceWakeLock")
                .apply { acquire(60 * 60 * 1000L) }
        } catch (e: Exception) { Log.e(TAG, "WakeLock error", e) }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    private fun formatTime(): String {
        return SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
    }
}
