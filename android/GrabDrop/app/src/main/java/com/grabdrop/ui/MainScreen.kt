// /GrabDrop/app/src/main/java/com/grabdrop/ui/MainScreen.kt
package com.grabdrop.ui

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.grabdrop.service.ServiceState
import com.grabdrop.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    isRunning: Boolean,
    statusText: String,
    nearbyDevices: Int,
    eventLog: List<String>,
    onStartClicked: () -> Unit,
    onStopClicked: () -> Unit
) {
    val debugMode by ServiceState.debugMode.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.PanTool,
                            contentDescription = null,
                            tint = Blue400,
                            modifier = Modifier.size(28.dp)
                        )
                        Spacer(Modifier.width(10.dp))
                        Text("GrabDrop", fontWeight = FontWeight.Bold)
                    }
                },
                actions = {
                    // Debug toggle
                    IconButton(
                        onClick = { ServiceState.setDebugMode(!debugMode) }
                    ) {
                        Icon(
                            imageVector = Icons.Default.BugReport,
                            contentDescription = "Toggle Debug",
                            tint = if (debugMode) Color(0xFFFF9800) else TextSecondary,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface
                )
            )
        },
        containerColor = DarkSurface
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(12.dp))

            // Debug mode banner
            AnimatedVisibility(visible = debugMode) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp),
                    shape = RoundedCornerShape(10.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF3E2723)
                    )
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 14.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.BugReport,
                            contentDescription = null,
                            tint = Color(0xFFFF9800),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            "Debug Mode ON — verbose logs + debug frames",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color(0xFFFFCC80)
                        )
                    }
                }
            }

            // Status Card
            StatusCard(isRunning = isRunning, statusText = statusText)

            Spacer(Modifier.height(12.dp))

            // Stats Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                StatCard(
                    modifier = Modifier.weight(1f),
                    icon = Icons.Default.Devices,
                    label = "Nearby",
                    value = "$nearbyDevices",
                    color = Cyan400
                )
                StatCard(
                    modifier = Modifier.weight(1f),
                    icon = Icons.Default.Screenshot,
                    label = "Mode",
                    value = if (isRunning) "Active" else "Off",
                    color = if (isRunning) GreenActive else TextSecondary
                )
            }

            Spacer(Modifier.height(16.dp))

            // Start / Stop Button
            GrabDropButton(
                isRunning = isRunning,
                onStartClicked = onStartClicked,
                onStopClicked = onStopClicked
            )

            Spacer(Modifier.height(12.dp))

            // How it works — only when not running and log empty
            AnimatedVisibility(visible = !isRunning && eventLog.isEmpty()) {
                HowItWorksCard()
            }

            // Event Log
            EventLogCard(
                eventLog = eventLog,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(vertical = 4.dp)
            )
        }
    }
}

@Composable
private fun StatusCard(isRunning: Boolean, statusText: String) {
    val pulseAnim = rememberInfiniteTransition(label = "pulse")
    val pulseScale by pulseAnim.animateFloat(
        initialValue = 1f, targetValue = 1.4f,
        animationSpec = infiniteRepeatable(tween(1000), RepeatMode.Reverse),
        label = "pulse_scale"
    )

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(16.dp)
                    .scale(if (isRunning) pulseScale else 1f)
                    .clip(CircleShape)
                    .background(if (isRunning) GreenActive else TextSecondary)
            )
            Spacer(Modifier.width(14.dp))
            Column {
                Text(
                    if (isRunning) "Active" else "Inactive",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = if (isRunning) GreenActive else TextSecondary
                )
                Text(statusText, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
            }
        }
    }
}

@Composable
private fun StatCard(
    modifier: Modifier = Modifier,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String, value: String, color: Color
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(icon, null, tint = color, modifier = Modifier.size(28.dp))
            Spacer(Modifier.height(8.dp))
            Text(value, style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, color = color)
            Text(label, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
        }
    }
}

@Composable
private fun GrabDropButton(isRunning: Boolean, onStartClicked: () -> Unit, onStopClicked: () -> Unit) {
    Button(
        onClick = { if (isRunning) onStopClicked() else onStartClicked() },
        modifier = Modifier.fillMaxWidth().height(56.dp),
        shape = RoundedCornerShape(16.dp),
        colors = ButtonDefaults.buttonColors(containerColor = if (isRunning) RedStop else Blue600)
    ) {
        Icon(
            if (isRunning) Icons.Default.Stop else Icons.Default.PlayArrow,
            null, modifier = Modifier.size(26.dp)
        )
        Spacer(Modifier.width(10.dp))
        Text(
            if (isRunning) "STOP SERVICE" else "START",
            style = MaterialTheme.typography.labelLarge, fontSize = 18.sp
        )
    }
}

@Composable
private fun HowItWorksCard() {
    Card(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text("How it works", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold, color = Blue400)
            Spacer(Modifier.height(12.dp))
            HowItWorksStep("✊", "Grab to capture", "Make a grab gesture (palm → fist) to take a screenshot")
            Spacer(Modifier.height(10.dp))
            HowItWorksStep("📡", "Auto-broadcast", "The screenshot is announced to nearby devices on your WiFi")
            Spacer(Modifier.height(10.dp))
            HowItWorksStep("🤚", "Release to receive", "On another device, open your hand (fist → palm) to receive it")
        }
    }
}

@Composable
private fun HowItWorksStep(emoji: String, title: String, desc: String) {
    Row {
        Text(emoji, fontSize = 22.sp)
        Spacer(Modifier.width(12.dp))
        Column {
            Text(title, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium, color = TextPrimary)
            Text(desc, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
        }
    }
}

@Composable
private fun EventLogCard(eventLog: List<String>, modifier: Modifier = Modifier) {
    val listState = rememberLazyListState()

    LaunchedEffect(eventLog.size) {
        if (eventLog.isNotEmpty()) listState.animateScrollToItem(0)
    }

    Card(
        modifier = modifier,
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Terminal, null, tint = TextSecondary, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text("Live Log", style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold, color = TextSecondary)
                Spacer(Modifier.weight(1f))
                Text("${eventLog.size}", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
            }
            Spacer(Modifier.height(8.dp))

            if (eventLog.isEmpty()) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("Tap START to begin...", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                }
            } else {
                LazyColumn(state = listState, modifier = Modifier.fillMaxSize()) {
                    items(eventLog) { event ->
                        Text(
                            event,
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 15.sp
                            ),
                            color = when {
                                "❌" in event || "💀" in event -> RedStop
                                "✅" in event || "CONFIRMED" in event -> GreenActive
                                "🔔" in event || "WAKEUP" in event -> Cyan400
                                "⏱️" in event -> Blue400
                                "🔄" in event -> Color(0xFFFF9800)
                                "📊" in event || "🐛" in event -> TextSecondary
                                else -> TextPrimary
                            },
                            maxLines = 3,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(vertical = 2.dp)
                        )
                    }
                }
            }
        }
    }
}
