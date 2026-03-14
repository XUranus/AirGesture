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

            // How it works info — only when not running
            AnimatedVisibility(visible = !isRunning && eventLog.isEmpty()) {
                HowItWorksCard()
            }

            // Event Log — takes remaining space
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
        initialValue = 1f,
        targetValue = 1.4f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse_scale"
    )

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
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
                    text = if (isRunning) "Active" else "Inactive",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = if (isRunning) GreenActive else TextSecondary
                )
                Text(
                    text = statusText,
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextSecondary
                )
            }
        }
    }
}

@Composable
private fun StatCard(
    modifier: Modifier = Modifier,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    value: String,
    color: Color
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
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(28.dp)
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
        }
    }
}

@Composable
private fun GrabDropButton(
    isRunning: Boolean,
    onStartClicked: () -> Unit,
    onStopClicked: () -> Unit
) {
    val buttonScale by animateFloatAsState(
        targetValue = 1f,
        animationSpec = spring(dampingRatio = 0.6f),
        label = "btn_scale"
    )

    Button(
        onClick = { if (isRunning) onStopClicked() else onStartClicked() },
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .scale(buttonScale),
        shape = RoundedCornerShape(16.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isRunning) RedStop else Blue600
        )
    ) {
        Icon(
            imageVector = if (isRunning) Icons.Default.Stop else Icons.Default.PlayArrow,
            contentDescription = null,
            modifier = Modifier.size(26.dp)
        )
        Spacer(Modifier.width(10.dp))
        Text(
            text = if (isRunning) "STOP SERVICE" else "START",
            style = MaterialTheme.typography.labelLarge,
            fontSize = 18.sp
        )
    }
}

@Composable
private fun HowItWorksCard() {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                text = "How it works",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = Blue400
            )
            Spacer(Modifier.height(12.dp))

            HowItWorksStep(
                emoji = "✊",
                title = "Grab to capture",
                desc = "Make a grab gesture (palm → fist) to take a screenshot"
            )
            Spacer(Modifier.height(10.dp))
            HowItWorksStep(
                emoji = "📡",
                title = "Auto-broadcast",
                desc = "The screenshot is announced to nearby devices on your WiFi"
            )
            Spacer(Modifier.height(10.dp))
            HowItWorksStep(
                emoji = "🤚",
                title = "Release to receive",
                desc = "On another device, open your hand (fist → palm) to receive it"
            )
        }
    }
}

@Composable
private fun HowItWorksStep(emoji: String, title: String, desc: String) {
    Row {
        Text(text = emoji, fontSize = 22.sp)
        Spacer(Modifier.width(12.dp))
        Column {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
                color = TextPrimary
            )
            Text(
                text = desc,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
        }
    }
}

@Composable
private fun EventLogCard(
    eventLog: List<String>,
    modifier: Modifier = Modifier
) {
    val listState = rememberLazyListState()

    // Auto-scroll to top when new events arrive
    LaunchedEffect(eventLog.size) {
        if (eventLog.isNotEmpty()) {
            listState.animateScrollToItem(0)
        }
    }

    Card(
        modifier = modifier,
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Terminal,
                    contentDescription = null,
                    tint = TextSecondary,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    text = "Live Log",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = TextSecondary
                )
                Spacer(Modifier.weight(1f))
                Text(
                    text = "${eventLog.size} events",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }

            Spacer(Modifier.height(8.dp))

            if (eventLog.isEmpty()) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Tap START to begin gesture detection...",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                }
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize()
                ) {
                    items(eventLog) { event ->
                        Text(
                            text = event,
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                lineHeight = 15.sp
                            ),
                            color = when {
                                "❌" in event || "💀" in event || "CRASH" in event ->
                                    RedStop
                                "✅" in event || "CONFIRMED" in event ->
                                    GreenActive
                                "🔔" in event || "WAKEUP" in event ->
                                    Cyan400
                                "⏱️" in event ->
                                    Blue400
                                "📊" in event ->
                                    TextSecondary
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
