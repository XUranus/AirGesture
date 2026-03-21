// /GrabDrop/app/src/main/java/com/grabdrop/ui/MainActivity.kt
package com.grabdrop.ui

import android.Manifest
import android.accessibilityservice.AccessibilityServiceInfo
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import androidx.compose.ui.platform.AccessibilityManager
import androidx.core.content.ContextCompat
import com.grabdrop.service.GrabDropService
import com.grabdrop.service.MediaProjectionHolder
import com.grabdrop.service.ServiceState
import com.grabdrop.service.SwipeAccessibilityService
import com.grabdrop.ui.theme.GrabDropTheme
import com.grabdrop.util.Constants

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var cameraPermissionLauncher: ActivityResultLauncher<String>
    private lateinit var notificationPermissionLauncher: ActivityResultLauncher<String>
    private lateinit var mediaProjectionLauncher: ActivityResultLauncher<Intent>
    private lateinit var overlayPermissionLauncher: ActivityResultLauncher<Intent>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate")

        setupLaunchers()

        setContent {
            GrabDropTheme {
                val isRunning by ServiceState.isRunning.collectAsState()
                val statusText by ServiceState.statusText.collectAsState()
                val nearbyDevices by ServiceState.nearbyDevices.collectAsState()
                val eventLog by ServiceState.eventLog.collectAsState()

                MainScreen(
                    isRunning = isRunning,
                    statusText = statusText,
                    nearbyDevices = nearbyDevices,
                    eventLog = eventLog,
                    onStartClicked = { startPermissionFlow() },
                    onStopClicked = { stopService() }
                )
            }
        }
    }

    private fun setupLaunchers() {
        cameraPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            Log.d(TAG, "Camera permission granted=$granted")
            if (granted) {
                continuePermissionFlow(step = 2)
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
            }
        }

        notificationPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            Log.d(TAG, "Notification permission granted=$granted")
            continuePermissionFlow(step = 3)
        }

        overlayPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) {
            val canDraw = Settings.canDrawOverlays(this)
            Log.d(TAG, "Overlay permission canDraw=$canDraw")
            if (canDraw) {
                continuePermissionFlow(step = 4)
            } else {
                Toast.makeText(
                    this,
                    "Overlay permission required for animations",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

        mediaProjectionLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            Log.d(TAG, "MediaProjection result: code=${result.resultCode}, data=${result.data}")
            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                // Store in singleton — avoids parcelable serialization issues
                MediaProjectionHolder.store(result.resultCode, result.data!!)
                Log.d(TAG, "MediaProjection data stored in holder")
                launchService()
            } else {
                Toast.makeText(
                    this,
                    "Screen capture permission required",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun startPermissionFlow() {
        Log.d(TAG, "Starting permission flow")
        continuePermissionFlow(step = 1)
    }

    private fun continuePermissionFlow(step: Int) {
        Log.d(TAG, "Permission flow step=$step")
        when (step) {
            1 -> {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED
                ) continuePermissionFlow(step = 2)
                else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }

            2 -> {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    if (ContextCompat.checkSelfPermission(
                            this, Manifest.permission.POST_NOTIFICATIONS
                        ) == PackageManager.PERMISSION_GRANTED
                    ) continuePermissionFlow(step = 3)
                    else notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                } else continuePermissionFlow(step = 3)
            }

            3 -> {
                if (Settings.canDrawOverlays(this)) {
                    continuePermissionFlow(step = 4)
                } else {
                    val intent = Intent(
                        Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:$packageName")
                    )
                    overlayPermissionLauncher.launch(intent)
                }
            }

            4 -> {
                // Check accessibility service for swipe support
                if (isAccessibilityServiceEnabled()) {
                    Log.d(TAG, "Accessibility service already enabled")
                    continuePermissionFlow(step = 5)
                } else {
                    // Show dialog explaining why, then open settings
                    android.app.AlertDialog.Builder(this)
                        .setTitle("Enable Swipe Gesture")
                        .setMessage(
                            "To simulate screen swipes with hand gestures, " +
                                    "enable GrabDrop in Accessibility settings.\n\n" +
                                    "This is optional — grab/release gestures work without it."
                        )
                        .setPositiveButton("Open Settings") { _, _ ->
                            startActivity(
                                Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                            )
                        }
                        .setNegativeButton("Skip") { _, _ ->
                            continuePermissionFlow(step = 5)
                        }
                        .show()
                }
            }

            5 -> {
                Log.d(TAG, "Requesting MediaProjection")
                val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE)
                        as MediaProjectionManager
                mediaProjectionLauncher.launch(mpm.createScreenCaptureIntent())
            }
        }
    }

    private fun isAccessibilityServiceEnabled(): Boolean {
        val am = getSystemService(Context.ACCESSIBILITY_SERVICE) as android.view.accessij4bility.AccessibilityManager
        val enabled = am.getEnabledAccessibilityServiceList(
            AccessibilityServiceInfo.FEEDBACK_ALL_MASK
        )
        return enabled.any {
            it.resolveInfo.serviceInfo.name == SwipeAccessibilityService::class.java.name
        }
    }

    private fun launchService() {
        Log.d(TAG, "Launching service")
        try {
            val serviceIntent = Intent(this, GrabDropService::class.java)
            ContextCompat.startForegroundService(this, serviceIntent)
            Log.d(TAG, "startForegroundService called successfully")
            Toast.makeText(this, "GrabDrop is now active!", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start service", e)
            Toast.makeText(this, "Failed to start: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun stopService() {
        Log.d(TAG, "Stopping service")
        val stopIntent = Intent(this, GrabDropService::class.java).apply {
            action = Constants.SERVICE_ACTION_STOP
        }
        startService(stopIntent)
    }
}
