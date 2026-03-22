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
        val channel = NotificationChannel(
            Constants.NOTIFICATION_CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = getString(R.string.notification_channel_description)
            setShowBadge(false)
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }
}
