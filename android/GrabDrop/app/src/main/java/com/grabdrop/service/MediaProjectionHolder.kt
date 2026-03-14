// /GrabDrop/app/src/main/java/com/grabdrop/service/MediaProjectionHolder.kt
package com.grabdrop.service

import android.content.Intent

object MediaProjectionHolder {
    private const val UNSET = Int.MIN_VALUE

    var resultCode: Int = UNSET
        private set
    var resultData: Intent? = null
        private set

    val isAvailable: Boolean
        get() = resultCode != UNSET && resultData != null

    fun store(code: Int, data: Intent) {
        resultCode = code
        resultData = data
    }

    fun consume(): Pair<Int, Intent?> {
        return Pair(resultCode, resultData)
    }

    fun clear() {
        resultCode = UNSET
        resultData = null
    }
}
