// /GrabDrop/app/src/main/java/com/grabdrop/network/ScreenshotOffer.kt
package com.grabdrop.network

data class ScreenshotOffer(
    val senderId: String,
    val senderName: String,
    val senderAddress: java.net.InetAddress,
    val tcpPort: Int,
    val fileSize: Int,
    val timestamp: Long
)

