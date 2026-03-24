package com.grabdrop

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import com.grabdrop.util.AppSettings
import com.grabdrop.util.Constants

class GrabDropApp : Application() {
    override fun onCreate() {
        super.onCreate()
        AppSettings.init(this)
        createNotificationChannel()
    }

    private fun createNotificationChannel() {
        // Service channel (low importance)
        val channel = NotificationChannel(
            Constants.NOTIFICATION_CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = getString(R.string.notification_channel_description)
            setShowBadge(false)
        }

        // High priority channel for screenshot received notifications
        val highChannel = NotificationChannel(
            Constants.NOTIFICATION_CHANNEL_HIGH_ID,
            "Screenshot Received",
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            description = "Notifications for received screenshots"
            setShowBadge(true)
            enableLights(true)
            enableVibration(true)
        }

        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
        manager.createNotificationChannel(highChannel)
    }
}
