// /GrabDrop/app/src/main/java/com/grabdrop/service/GrabDropService.kt
package com.grabdrop.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.net.Uri
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
import java.util.concurrent.ConcurrentLinkedDeque

class GrabDropService : Service() {

    companion object {
        private const val TAG = "GrabDropService"
        private const val RETROACTIVE_MATCH_WINDOW_MS = 3_000L
        private const val RECEIVED_NOTIFICATION_ID = 2001
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
    private var defaultUncaughtHandler: Thread.UncaughtExceptionHandler? = null

    private val pendingOffers = ConcurrentLinkedDeque<TimestampedOffer>()
    @Volatile
    private var lastUnmatchedReleaseTime = 0L
    private val offerLock = Any()

    data class TimestampedOffer(
        val offer: ScreenshotOffer,
        val receivedAt: Long = System.currentTimeMillis()
    )

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
        gestureDetector = RealGestureDetector(this, overlayManager)
        networkManager = NetworkManager(this)
        soundPlayer = SoundPlayer(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand called, action=${intent?.action}")

        when (intent?.action) {
            Constants.SERVICE_ACTION_STOP -> {
                stopSelf()
                return START_NOT_STICKY
            }
        }

        try {
            val notification = buildNotification()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(
                    Constants.NOTIFICATION_ID, notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or
                            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
                )
            } else {
                startForeground(Constants.NOTIFICATION_ID, notification)
            }
        } catch (e: Exception) {
            Log.e(TAG, "startForeground FAILED", e)
            stopSelf()
            return START_NOT_STICKY
        }

        if (!MediaProjectionHolder.isAvailable) {
            ServiceState.setRunning(true)
            ServiceState.addEvent(formatTime() + " ⚠️ Started WITHOUT screen capture")
            acquireWakeLock()
            startGestureAndNetwork()
            return START_STICKY
        }

        val (resultCode, resultData) = MediaProjectionHolder.consume()
        try {
            screenCaptureManager.initProjection(resultCode, resultData!!)
        } catch (e: Exception) {
            ServiceState.addEvent(formatTime() + " ⚠️ Screen capture init failed: ${e.message}")
        }

        acquireWakeLock()
        startGestureAndNetwork()
        ServiceState.setRunning(true)
        ServiceState.addEvent(formatTime() + " ✅ Service started")

        return START_STICKY
    }

    override fun onDestroy() {
        stopAll()
        defaultUncaughtHandler?.let { Thread.setDefaultUncaughtExceptionHandler(it) }
        ServiceState.setRunning(false)
        ServiceState.addEvent(formatTime() + " Service stopped")
        super.onDestroy()
    }

    private fun startGestureAndNetwork() {
        networkManager.start()
        gestureDetector.start()

        serviceScope.launch {
            gestureDetector.events.collect { event ->
                try {
                    when (event) {
                        is GestureEvent.Grab -> handleGrab()
                        is GestureEvent.Release -> handleRelease()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error handling gesture", e)
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
                ServiceState.setNearbyDevices(count)
            }
        }
    }

    private fun stopAll() {
        try { gestureDetector.stop() } catch (_: Exception) {}
        try { networkManager.stop() } catch (_: Exception) {}
        try { screenCaptureManager.release() } catch (_: Exception) {}
        try { soundPlayer.release() } catch (_: Exception) {}
        try { overlayManager.removeAll() } catch (_: Exception) {}
        releaseWakeLock()
        serviceScope.cancel()
    }

    // --- Offer Queue ---

    private fun addOffer(offer: ScreenshotOffer) {
        val ts = TimestampedOffer(offer)
        synchronized(offerLock) {
            pendingOffers.addLast(ts)
            cleanupExpiredOffers()
            ServiceState.addEvent(formatTime() +
                    " 📡 Offer queued from ${offer.senderName} (${pendingOffers.size} pending)")
        }
    }

    private fun getBestOffer(): ScreenshotOffer? {
        synchronized(offerLock) {
            cleanupExpiredOffers()
            if (pendingOffers.isEmpty()) return null
            val best = pendingOffers.removeLast()
            pendingOffers.clear()
            return best.offer
        }
    }

    private fun cleanupExpiredOffers() {
        val now = System.currentTimeMillis()
        while (pendingOffers.isNotEmpty()) {
            val oldest = pendingOffers.peekFirst() ?: break
            if (now - oldest.receivedAt > Constants.SCREENSHOT_OFFER_TIMEOUT_MS) {
                pendingOffers.removeFirst()
            } else break
        }
    }

    // --- Core Logic ---

    private suspend fun handleGrab() {
        val now = System.currentTimeMillis()
        if (now - lastGrabTime < Constants.GRAB_COOLDOWN_MS) return
        lastGrabTime = now

        ServiceState.setStatus("Taking screenshot...")
        ServiceState.addEvent(formatTime() + " ✊ GRAB detected")

        val bitmap = try { screenCaptureManager.capture() }
        catch (e: Exception) {
            ServiceState.addEvent(formatTime() + " ❌ Screenshot failed: ${e.message}")
            ServiceState.setStatus("Monitoring gestures...")
            return
        }

        if (bitmap == null) {
            ServiceState.addEvent(formatTime() + " ❌ Screenshot returned null")
            ServiceState.setStatus("Monitoring gestures...")
            return
        }

        try { MediaStoreHelper.saveBitmap(this, bitmap, "GrabDrop_Sent") } catch (_: Exception) {}
        try { overlayManager.showGrabAnimation(bitmap) } catch (_: Exception) {}
        try { soundPlayer.playShutter() } catch (_: Exception) {}

        try {
            val imageData = MediaStoreHelper.bitmapToByteArray(bitmap)
            networkManager.broadcastScreenshotAvailable(imageData)
        } catch (_: Exception) {}

        ServiceState.addEvent(formatTime() + " 📸 Screenshot captured & broadcast sent")
        ServiceState.setStatus("Monitoring gestures...")
    }

    private suspend fun handleRelease() {
        val offer = getBestOffer()

        if (offer == null) {
            ServiceState.addEvent(formatTime() + " 🤚 RELEASE — no offer yet, waiting for retroactive match")
            lastUnmatchedReleaseTime = System.currentTimeMillis()
            return
        }

        ServiceState.addEvent(formatTime() + " 🤚 RELEASE — matched offer from ${offer.senderName}")
        downloadAndSave(offer)
    }

    private fun handleIncomingOffer(offer: ScreenshotOffer) {
        addOffer(offer)

        val now = System.currentTimeMillis()
        val releaseAge = now - lastUnmatchedReleaseTime
        val hadRecentRelease = releaseAge in 1..RETROACTIVE_MATCH_WINDOW_MS

        if (hadRecentRelease) {
            lastUnmatchedReleaseTime = 0L
            ServiceState.addEvent(formatTime() +
                    " 🔄 Retroactive match! RELEASE was ${releaseAge}ms ago")

            val matchedOffer = getBestOffer()
            if (matchedOffer != null) {
                serviceScope.launch { downloadAndSave(matchedOffer) }
            }
        }
    }

    private suspend fun downloadAndSave(offer: ScreenshotOffer) {
        ServiceState.setStatus("Receiving screenshot...")
        ServiceState.addEvent(formatTime() + " 📥 Downloading from ${offer.senderName}...")

        // Start looping ripple while downloading
        overlayManager.startLoopingRipple()

        val data = try {
            networkManager.downloadScreenshot(offer)
        } catch (e: Exception) {
            Log.e(TAG, "Download exception", e)
            null
        }

        // Stop looping ripple
        overlayManager.stopLoopingRipple()

        if (data == null) {
            ServiceState.addEvent(formatTime() + " ❌ Download failed")
            ServiceState.setStatus("Monitoring gestures...")
            return
        }

        var bitmap: Bitmap? = null
        try {
            bitmap = MediaStoreHelper.byteArrayToBitmap(data)
            val uri = MediaStoreHelper.saveBitmap(this, bitmap, "GrabDrop_Received")

            // Final single ripple + sound
            try { overlayManager.showReleaseAnimation() } catch (_: Exception) {}
            try { soundPlayer.playReceive() } catch (_: Exception) {}

            if (uri != null) {
                openReceivedImage(uri)
            }

            ServiceState.addEvent(formatTime() + " 📥 Screenshot received & saved!")
        } catch (e: Exception) {
            ServiceState.addEvent(formatTime() + " ❌ Processing failed: ${e.message}")
        } finally {
            bitmap?.recycle()
        }

        ServiceState.setStatus("Monitoring gestures...")
    }

    /**
     * Open received image — works whether app is foreground or background.
     * Shows a notification with view action, AND tries to open directly.
     */
    private fun openReceivedImage(uri: Uri) {
        // 1. Always post a notification so user can tap to view
        val viewIntent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "image/png")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        val viewPending = PendingIntent.getActivity(
            this, uri.hashCode(), viewIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or
                    PendingIntent.FLAG_IMMUTABLE or
                    PendingIntent.FLAG_ONE_SHOT
        )

        val notification = NotificationCompat.Builder(this, Constants.NOTIFICATION_CHANNEL_ID)
            .setContentTitle("📥 Screenshot Received!")
            .setContentText("Tap to view the received screenshot")
            .setSmallIcon(R.drawable.ic_notification)
            .setAutoCancel(true)
            .setContentIntent(viewPending)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            // Full-screen intent brings it to attention even in background
            .setFullScreenIntent(viewPending, true)
            .setCategory(NotificationCompat.CATEGORY_MESSAGE)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE)
                as android.app.NotificationManager
        notificationManager.notify(
            RECEIVED_NOTIFICATION_ID + (System.currentTimeMillis() % 1000).toInt(),
            notification
        )

        // 2. Also try to open directly (works if app is in foreground)
        try {
            val directIntent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "image/png")
                addFlags(
                    Intent.FLAG_ACTIVITY_NEW_TASK or
                            Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            startActivity(directIntent)
        } catch (e: Exception) {
            Log.d(TAG, "Direct open failed (app may be in background): ${e.message}")
        }
    }

    // --- Notification ---

    private fun buildNotification(): Notification {
        val openPending = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).apply { flags = Intent.FLAG_ACTIVITY_SINGLE_TOP },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stopPending = PendingIntent.getService(
            this, 1,
            Intent(this, GrabDropService::class.java).apply { action = Constants.SERVICE_ACTION_STOP },
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
        } catch (_: Exception) {}
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    private fun formatTime(): String {
        return SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
    }
}
