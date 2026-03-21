// /GrabDrop/app/src/main/java/com/grabdrop/network/NetworkManager.kt
package com.grabdrop.network

import android.content.Context
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import com.grabdrop.util.Constants
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONObject
import java.io.*
import java.net.*
import java.util.concurrent.ConcurrentHashMap

class NetworkManager(private val context: Context) {

    companion object {
        private const val TAG = "NetworkManager"
        private const val HEARTBEAT_INTERVAL = 3_000L
        private const val DEVICE_TIMEOUT = 10_000L
        private const val CLEANUP_INTERVAL = 5_000L

        private const val MSG_TYPE_HEARTBEAT = "HEARTBEAT"
        private const val MSG_TYPE_SCREENSHOT = "SCREENSHOT_READY"
    }

    private val scope = CoroutineScope(
        Dispatchers.IO + SupervisorJob() +
                CoroutineExceptionHandler { _, throwable ->
                    Log.e(TAG, "Coroutine error", throwable)
                }
    )

    private val _incomingOffers = MutableSharedFlow<ScreenshotOffer>(extraBufferCapacity = 10)
    val incomingOffers: SharedFlow<ScreenshotOffer> = _incomingOffers

    private val _nearbyDeviceCount = MutableStateFlow(0)
    val nearbyDeviceCount: StateFlow<Int> = _nearbyDeviceCount

    // deviceId -> last seen timestamp
    private val discoveredDevices = ConcurrentHashMap<String, DeviceInfo>()

    private var multicastLock: WifiManager.MulticastLock? = null
    private var udpListenerJob: Job? = null
    private var heartbeatJob: Job? = null
    private var cleanupJob: Job? = null
    private var tcpServerSocket: ServerSocket? = null

    private var latestScreenshotData: ByteArray? = null

    data class DeviceInfo(
        val deviceId: String,
        val deviceName: String,
        val address: InetAddress,
        var lastSeen: Long
    )

    fun start() {
        acquireMulticastLock()
        startUdpListener()
        startHeartbeat()
        startDeviceCleanup()
        Log.d(TAG, "Started all network components")
    }

    fun stop() {
        heartbeatJob?.cancel()
        cleanupJob?.cancel()
        udpListenerJob?.cancel()
        tcpServerSocket?.close()
        releaseMulticastLock()
        discoveredDevices.clear()
        _nearbyDeviceCount.value = 0
        scope.cancel()
        Log.d(TAG, "Stopped")
    }

    // --- Heartbeat ---

    private fun startHeartbeat() {
        heartbeatJob = scope.launch {
            // Send first heartbeat immediately
            sendHeartbeat()

            while (isActive) {
                delay(HEARTBEAT_INTERVAL)
                sendHeartbeat()
            }
        }
        Log.d(TAG, "Heartbeat started (interval=${HEARTBEAT_INTERVAL}ms)")
    }

    private suspend fun sendHeartbeat() {
        try {
            val json = JSONObject().apply {
                put("type", MSG_TYPE_HEARTBEAT)
                put("sender_id", Constants.DEVICE_ID)
                put("sender_name", Build.MODEL)
                put("timestamp", System.currentTimeMillis())
            }

            val data = json.toString().toByteArray()

            // Send to multicast group
            try {
                val address = InetAddress.getByName(Constants.MULTICAST_GROUP)
                val packet = DatagramPacket(data, data.size, address, Constants.UDP_PORT)
                DatagramSocket().use { socket ->
                    socket.broadcast = true
                    socket.send(packet)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Multicast heartbeat failed", e)
            }

            // Also send to broadcast address
            try {
                val broadcastAddr = InetAddress.getByName("255.255.255.255")
                val packet = DatagramPacket(data, data.size, broadcastAddr, Constants.UDP_PORT)
                DatagramSocket().use { socket ->
                    socket.broadcast = true
                    socket.send(packet)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Broadcast heartbeat failed", e)
            }
        } catch (e: Exception) {
            Log.e(TAG, "sendHeartbeat error", e)
        }
    }

    // --- Device Cleanup ---

    private fun startDeviceCleanup() {
        cleanupJob = scope.launch {
            while (isActive) {
                delay(CLEANUP_INTERVAL)
                cleanupStaleDevices()
            }
        }
    }

    private fun cleanupStaleDevices() {
        val now = System.currentTimeMillis()
        val stale = discoveredDevices.entries.filter { (_, info) ->
            now - info.lastSeen > DEVICE_TIMEOUT
        }
        for ((id, info) in stale) {
            discoveredDevices.remove(id)
            Log.d(TAG, "Device expired: ${info.deviceName} ($id)")
        }
        updateDeviceCount()
    }

    private fun updateDeviceCount() {
        val count = discoveredDevices.size
        if (_nearbyDeviceCount.value != count) {
            _nearbyDeviceCount.value = count
            Log.d(TAG, "Nearby devices: $count")
        }
    }

    // --- Screenshot Broadcast ---

    fun broadcastScreenshotAvailable(screenshotData: ByteArray) {
        latestScreenshotData = screenshotData
        scope.launch {
            val tcpPort = startTcpServer(screenshotData)
            sendScreenshotBroadcast(tcpPort, screenshotData.size)
        }
    }

    private fun sendScreenshotBroadcast(tcpPort: Int, fileSize: Int) {
        scope.launch {
            try {
                val json = JSONObject().apply {
                    put("type", MSG_TYPE_SCREENSHOT)
                    put("sender_id", Constants.DEVICE_ID)
                    put("sender_name", Build.MODEL)
                    put("tcp_port", tcpPort)
                    put("file_size", fileSize)
                    put("timestamp", System.currentTimeMillis())
                }

                val data = json.toString().toByteArray()

                try {
                    val address = InetAddress.getByName(Constants.MULTICAST_GROUP)
                    val packet = DatagramPacket(data, data.size, address, Constants.UDP_PORT)
                    DatagramSocket().use { socket ->
                        socket.broadcast = true
                        socket.send(packet)
                    }
                } catch (_: Exception) {}

                try {
                    val broadcastAddr = InetAddress.getByName("255.255.255.255")
                    val packet = DatagramPacket(data, data.size, broadcastAddr, Constants.UDP_PORT)
                    DatagramSocket().use { socket ->
                        socket.broadcast = true
                        socket.send(packet)
                    }
                } catch (_: Exception) {}

                Log.d(TAG, "Screenshot broadcast sent on port $tcpPort")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send screenshot broadcast", e)
            }
        }
    }

    // --- Download ---

    suspend fun downloadScreenshot(offer: ScreenshotOffer): ByteArray? =
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Connecting to ${offer.senderAddress}:${offer.tcpPort}")
                val socket = Socket()
                socket.connect(
                    InetSocketAddress(offer.senderAddress, offer.tcpPort),
                    5000
                )

                val output = socket.getOutputStream()
                output.write("GET\n".toByteArray())
                output.flush()

                val input = socket.getInputStream()
                val dis = DataInputStream(input)
                val length = dis.readInt()
                Log.d(TAG, "Receiving $length bytes")

                val data = ByteArray(length)
                dis.readFully(data)

                socket.close()
                Log.d(TAG, "Download complete")
                data
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                null
            }
        }

    // --- UDP Listener ---

    private fun startUdpListener() {
        udpListenerJob = scope.launch {
            // Try multicast first, fall back to plain broadcast
            try {
                startMulticastListener()
            } catch (e: Exception) {
                Log.e(TAG, "Multicast listener failed, trying fallback", e)
                startBroadcastListener()
            }
        }
    }

    private suspend fun startMulticastListener() {
        val group = InetAddress.getByName(Constants.MULTICAST_GROUP)
        val socket = MulticastSocket(Constants.UDP_PORT).apply {
            reuseAddress = true
            joinGroup(group)
            soTimeout = 0
        }

        val buffer = ByteArray(4096)
        Log.d(TAG, "Multicast UDP listener started on port ${Constants.UDP_PORT}")

        try {
            while (scope.isActive) {
                try {
                    val packet = DatagramPacket(buffer, buffer.size)
                    socket.receive(packet)
                    val message = String(packet.data, 0, packet.length)
                    handleUdpMessage(message, packet.address)
                } catch (_: SocketTimeoutException) {
                    // continue
                } catch (e: Exception) {
                    if (scope.isActive) {
                        Log.e(TAG, "UDP receive error", e)
                        delay(1000)
                    }
                }
            }
        } finally {
            try {
                socket.leaveGroup(group)
                socket.close()
            } catch (_: Exception) {}
        }
    }

    private suspend fun startBroadcastListener() {
        val socket = DatagramSocket(null).apply {
            reuseAddress = true
            bind(InetSocketAddress(Constants.UDP_PORT))
            broadcast = true
            soTimeout = 0
        }

        val buffer = ByteArray(4096)
        Log.d(TAG, "Broadcast UDP fallback listener started")

        try {
            while (scope.isActive) {
                try {
                    val packet = DatagramPacket(buffer, buffer.size)
                    socket.receive(packet)
                    val message = String(packet.data, 0, packet.length)
                    handleUdpMessage(message, packet.address)
                } catch (_: SocketTimeoutException) {
                    // continue
                } catch (e: Exception) {
                    if (scope.isActive) {
                        Log.e(TAG, "Fallback receive error", e)
                        delay(1000)
                    }
                }
            }
        } finally {
            socket.close()
        }
    }

    private suspend fun handleUdpMessage(message: String, senderAddress: InetAddress) {
        try {
            val json = JSONObject(message)
            val type = json.getString("type")
            val senderId = json.getString("sender_id")

            // Ignore our own messages
            if (senderId == Constants.DEVICE_ID) return

            val senderName = json.optString("sender_name", "Unknown")
            val timestamp = json.optLong("timestamp", System.currentTimeMillis())

            when (type) {
                MSG_TYPE_HEARTBEAT -> {
                    handleHeartbeat(senderId, senderName, senderAddress, timestamp)
                }

                MSG_TYPE_SCREENSHOT -> {
                    // Also treat screenshot broadcast as a heartbeat
                    handleHeartbeat(senderId, senderName, senderAddress, timestamp)

                    val offer = ScreenshotOffer(
                        senderId = senderId,
                        senderName = senderName,
                        senderAddress = senderAddress,
                        tcpPort = json.getInt("tcp_port"),
                        fileSize = json.getInt("file_size"),
                        timestamp = timestamp
                    )
                    Log.d(TAG, "Screenshot offer from $senderName (${senderAddress}:${offer.tcpPort})")
                    _incomingOffers.emit(offer)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse UDP message", e)
        }
    }

    private fun handleHeartbeat(
        senderId: String,
        senderName: String,
        address: InetAddress,
        timestamp: Long
    ) {
        val existing = discoveredDevices[senderId]
        if (existing == null) {
            Log.d(TAG, "New device discovered: $senderName ($senderId) at $address")
        }

        discoveredDevices[senderId] = DeviceInfo(
            deviceId = senderId,
            deviceName = senderName,
            address = address,
            lastSeen = System.currentTimeMillis()
        )

        updateDeviceCount()
    }

    // --- TCP Server ---

    private suspend fun startTcpServer(data: ByteArray): Int = withContext(Dispatchers.IO) {
        tcpServerSocket?.close()

        val server = ServerSocket(0)
        tcpServerSocket = server
        val port = server.localPort

        Log.d(TAG, "TCP server started on port $port")

        scope.launch {
            try {
                server.soTimeout = Constants.SCREENSHOT_OFFER_TIMEOUT_MS.toInt() + 5000
                repeat(5) {
                    try {
                        val client = server.accept()
                        launch { handleTcpClient(client, data) }
                    } catch (_: SocketTimeoutException) {
                        return@launch
                    }
                }
            } catch (e: Exception) {
                if (e !is SocketException) {
                    Log.e(TAG, "TCP server error", e)
                }
            } finally {
                server.close()
            }
        }

        port
    }

    private suspend fun handleTcpClient(client: Socket, data: ByteArray) {
        withContext(Dispatchers.IO) {
            try {
                val reader = BufferedReader(InputStreamReader(client.getInputStream()))
                val request = reader.readLine()

                if (request == "GET") {
                    val dos = DataOutputStream(client.getOutputStream())
                    dos.writeInt(data.size)
                    dos.write(data)
                    dos.flush()
                    Log.d(TAG, "Sent ${data.size} bytes to ${client.inetAddress}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "TCP client handler error", e)
            } finally {
                client.close()
            }
        }
    }

    // --- Multicast Lock ---

    private fun acquireMulticastLock() {
        try {
            val wifiManager = context.applicationContext
                .getSystemService(Context.WIFI_SERVICE) as WifiManager
            multicastLock = wifiManager.createMulticastLock("GrabDrop").apply {
                setReferenceCounted(true)
                acquire()
            }
            Log.d(TAG, "Multicast lock acquired")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to acquire multicast lock", e)
        }
    }

    private fun releaseMulticastLock() {
        multicastLock?.let {
            if (it.isHeld) it.release()
        }
        multicastLock = null
    }
}
