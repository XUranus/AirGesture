// /GrabDrop/app/src/main/java/com/grabdrop/service/ServiceState.kt
package com.grabdrop.service

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

object ServiceState {
    private val _isRunning = MutableStateFlow(false)
    val isRunning: StateFlow<Boolean> = _isRunning

    private val _statusText = MutableStateFlow("Idle")
    val statusText: StateFlow<String> = _statusText

    private val _nearbyDevices = MutableStateFlow(0)
    val nearbyDevices: StateFlow<Int> = _nearbyDevices

    private val _lastEvent = MutableStateFlow("")
    val lastEvent: StateFlow<String> = _lastEvent

    private val _eventLog = MutableStateFlow<List<String>>(emptyList())
    val eventLog: StateFlow<List<String>> = _eventLog

    private val _debugMode = MutableStateFlow(false)
    val debugMode: StateFlow<Boolean> = _debugMode

    fun setRunning(running: Boolean) {
        _isRunning.value = running
        _statusText.value = if (running) "Monitoring gestures..." else "Idle"
    }

    fun setNearbyDevices(count: Int) {
        _nearbyDevices.value = count
    }

    fun addEvent(event: String) {
        _lastEvent.value = event
        _eventLog.value = (listOf(event) + _eventLog.value).take(500)
    }

    fun setStatus(text: String) {
        _statusText.value = text
    }

    fun setDebugMode(enabled: Boolean) {
        _debugMode.value = enabled
    }

    fun isDebug(): Boolean = _debugMode.value

    fun reset() {
        _isRunning.value = false
        _statusText.value = "Idle"
        _nearbyDevices.value = 0
        _lastEvent.value = ""
        _eventLog.value = emptyList()
    }
}
