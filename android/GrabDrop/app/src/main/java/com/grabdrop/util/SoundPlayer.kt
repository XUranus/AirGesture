// /GrabDrop/app/src/main/java/com/grabdrop/util/SoundPlayer.kt
package com.grabdrop.util

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.MediaActionSound
import android.media.ToneGenerator

class SoundPlayer(private val context: Context) {

    private val mediaActionSound = MediaActionSound()

    init {
        mediaActionSound.load(MediaActionSound.SHUTTER_CLICK)
    }

    fun playShutter() {
        if (!AppSettings.soundEnabled) return
        try {
            mediaActionSound.play(MediaActionSound.SHUTTER_CLICK)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun playReceive() {
        if (!AppSettings.soundEnabled) return
        try {
            val toneGen = ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80)
            toneGen.startTone(ToneGenerator.TONE_PROP_ACK, 200)
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                toneGen.release()
            }, 300)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun release() {
        mediaActionSound.release()
    }
}

