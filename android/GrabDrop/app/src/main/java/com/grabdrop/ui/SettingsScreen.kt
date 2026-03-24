package com.grabdrop.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.grabdrop.ui.theme.*
import com.grabdrop.util.AppSettings

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit
) {
    // Force recomposition after reset
    var resetTrigger by remember { mutableIntStateOf(0) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text("Settings", fontWeight = FontWeight.Bold)
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = TextPrimary
                        )
                    }
                },
                actions = {
                    IconButton(onClick = {
                        AppSettings.resetAll()
                        resetTrigger++
                    }) {
                        Icon(
                            Icons.Default.RestartAlt,
                            contentDescription = "Reset to Defaults",
                            tint = TextSecondary
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
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp)
        ) {
            // Hint card
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1A237E).copy(alpha = 0.3f))
            ) {
                Row(
                    modifier = Modifier.padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Default.Info,
                        contentDescription = null,
                        tint = Blue400,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        "Changes take effect next time the service starts.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Blue400
                    )
                }
            }

            // ── Detection Method ──
            DetectionMethodSection(resetTrigger = resetTrigger)

            Spacer(Modifier.height(8.dp))

            // ── Gesture Timing ──
            SettingsSection(
                title = "Gesture Timing",
                icon = Icons.Default.Timer,
                color = Blue400,
                key = resetTrigger
            ) {
                IntSettingItem(
                    label = "Idle Scan Rate (FPS)",
                    description = "How many frames per second to scan while idle. Higher = more responsive but uses more battery.",
                    key = AppSettings.KEY_IDLE_FPS,
                    default = AppSettings.DEF_IDLE_FPS,
                    min = 1, max = 30,
                    resetTrigger = resetTrigger
                )
                LongSettingItem(
                    label = "Active Tracking Interval (ms)",
                    description = "Milliseconds between frames during active gesture tracking. Lower = smoother tracking.",
                    key = AppSettings.KEY_WAKEUP_FRAME_INTERVAL_MS,
                    default = AppSettings.DEF_WAKEUP_FRAME_INTERVAL_MS,
                    min = 16, max = 200,
                    resetTrigger = resetTrigger
                )
                IntSettingItem(
                    label = "Detection Window Size",
                    description = "Number of recent frames used to detect a hand presence. Larger = more stable but slower.",
                    key = AppSettings.KEY_IDLE_WINDOW_SIZE,
                    default = AppSettings.DEF_IDLE_WINDOW_SIZE,
                    min = 3, max = 30,
                    resetTrigger = resetTrigger
                )
                IntSettingItem(
                    label = "Wake-up Sensitivity",
                    description = "How many frames (out of window) must show a hand to start tracking. Lower = more sensitive.",
                    key = AppSettings.KEY_IDLE_TRIGGER_THRESHOLD,
                    default = AppSettings.DEF_IDLE_TRIGGER_THRESHOLD,
                    min = 1, max = 30,
                    resetTrigger = resetTrigger
                )
                LongSettingItem(
                    label = "Gesture Time Limit (ms)",
                    description = "Maximum time allowed to complete a gesture after wake-up. Longer = more forgiving.",
                    key = AppSettings.KEY_WAKEUP_DURATION_MS,
                    default = AppSettings.DEF_WAKEUP_DURATION_MS,
                    min = 500, max = 10_000,
                    resetTrigger = resetTrigger
                )
                IntSettingItem(
                    label = "Confirmation Frames",
                    description = "Consecutive frames needed to confirm a grab or release gesture (Legacy mode only). Higher = fewer false triggers.",
                    key = AppSettings.KEY_WAKEUP_CONFIRM_FRAMES,
                    default = AppSettings.DEF_WAKEUP_CONFIRM_FRAMES,
                    min = 1, max = 30,
                    resetTrigger = resetTrigger
                )
            }

            Spacer(Modifier.height(8.dp))

            // ── Hand Classification ──
            SettingsSection(
                title = "Hand Recognition",
                icon = Icons.Default.PanTool,
                color = Cyan400,
                key = resetTrigger
            ) {
                FloatSettingItem(
                    label = "Finger Extended Ratio",
                    description = "Minimum tip-to-wrist / knuckle-to-wrist ratio for a finger to count as extended. Higher = stricter open-hand detection.",
                    key = AppSettings.KEY_FINGER_EXTENDED_THRESHOLD,
                    default = AppSettings.DEF_FINGER_EXTENDED_THRESHOLD,
                    min = 0.5f, max = 3.0f,
                    resetTrigger = resetTrigger
                )
                FloatSettingItem(
                    label = "Finger Curled Ratio",
                    description = "Maximum ratio for a finger to count as curled. Lower = stricter fist detection.",
                    key = AppSettings.KEY_FINGER_CURLED_THRESHOLD,
                    default = AppSettings.DEF_FINGER_CURLED_THRESHOLD,
                    min = 0.3f, max = 2.0f,
                    resetTrigger = resetTrigger
                )
                IntSettingItem(
                    label = "Min Fingers for Open Hand",
                    description = "Minimum extended fingers (out of 4, excluding thumb) to recognize an open palm.",
                    key = AppSettings.KEY_MIN_FINGERS_FOR_PALM,
                    default = AppSettings.DEF_MIN_FINGERS_FOR_PALM,
                    min = 1, max = 4,
                    resetTrigger = resetTrigger
                )
                IntSettingItem(
                    label = "Min Fingers for Fist",
                    description = "Minimum curled fingers (out of 4, excluding thumb) to recognize a closed fist.",
                    key = AppSettings.KEY_MIN_FINGERS_FOR_FIST,
                    default = AppSettings.DEF_MIN_FINGERS_FOR_FIST,
                    min = 1, max = 4,
                    resetTrigger = resetTrigger
                )
            }

            Spacer(Modifier.height(8.dp))

            // ── Swipe Detection ──
            SettingsSection(
                title = "Swipe Detection",
                icon = Icons.Default.SwipeVertical,
                color = Color(0xFFFF9800),
                key = resetTrigger
            ) {
                FloatSettingItem(
                    label = "Swipe Distance Threshold",
                    description = "Minimum hand movement (0-1 normalized) to register a swipe. Lower = more sensitive. (Legacy mode only)",
                    key = AppSettings.KEY_SWIPE_DISPLACEMENT,
                    default = AppSettings.DEF_SWIPE_DISPLACEMENT,
                    min = 0.01f, max = 0.5f,
                    resetTrigger = resetTrigger
                )
                IntSettingItem(
                    label = "Swipe Confirmation Frames",
                    description = "Consecutive frames of directional movement needed to confirm a swipe. (Legacy mode only)",
                    key = AppSettings.KEY_SWIPE_CONFIRM_FRAMES,
                    default = AppSettings.DEF_SWIPE_CONFIRM_FRAMES,
                    min = 1, max = 20,
                    resetTrigger = resetTrigger
                )
                FloatSettingItem(
                    label = "Minimum Swipe Speed",
                    description = "Minimum per-frame velocity to count as directional movement. Lower = more sensitive. (Legacy mode only)",
                    key = AppSettings.KEY_SWIPE_MIN_VELOCITY,
                    default = AppSettings.DEF_SWIPE_MIN_VELOCITY,
                    min = 0.001f, max = 0.1f,
                    resetTrigger = resetTrigger
                )
                LongSettingItem(
                    label = "Swipe Cooldown (ms)",
                    description = "Minimum time between consecutive swipe gestures to prevent double-triggers.",
                    key = AppSettings.KEY_SWIPE_COOLDOWN_MS,
                    default = AppSettings.DEF_SWIPE_COOLDOWN_MS,
                    min = 100, max = 5_000,
                    resetTrigger = resetTrigger
                )
            }

            Spacer(Modifier.height(8.dp))

            // ── Network ──
            SettingsSection(
                title = "Network",
                icon = Icons.Default.Wifi,
                color = GreenActive,
                key = resetTrigger
            ) {
                IntSettingItem(
                    label = "Discovery Port",
                    description = "UDP port used for device discovery and screenshot announcements. Must match on all devices.",
                    key = AppSettings.KEY_UDP_PORT,
                    default = AppSettings.DEF_UDP_PORT,
                    min = 1024, max = 65535,
                    resetTrigger = resetTrigger
                )
                StringSettingItem(
                    label = "Multicast Address",
                    description = "Multicast group address for LAN discovery. Must match on all devices.",
                    key = AppSettings.KEY_MULTICAST_GROUP,
                    default = AppSettings.DEF_MULTICAST_GROUP,
                    resetTrigger = resetTrigger
                )
                LongSettingItem(
                    label = "Screenshot Offer Timeout (ms)",
                    description = "How long a received screenshot offer stays valid before expiring.",
                    key = AppSettings.KEY_SCREENSHOT_OFFER_TIMEOUT,
                    default = AppSettings.DEF_SCREENSHOT_OFFER_TIMEOUT,
                    min = 1_000, max = 60_000,
                    resetTrigger = resetTrigger
                )
                LongSettingItem(
                    label = "Grab Cooldown (ms)",
                    description = "Minimum time between consecutive grab gestures to prevent double-captures.",
                    key = AppSettings.KEY_GRAB_COOLDOWN_MS,
                    default = AppSettings.DEF_GRAB_COOLDOWN_MS,
                    min = 500, max = 30_000,
                    resetTrigger = resetTrigger
                )
            }

            Spacer(Modifier.height(24.dp))

            // Reset button at bottom
            OutlinedButton(
                onClick = {
                    AppSettings.resetAll()
                    resetTrigger++
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.outlinedButtonColors(
                    contentColor = RedStop
                )
            ) {
                Icon(Icons.Default.RestartAlt, null, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text("Reset All to Defaults")
            }

            Spacer(Modifier.height(32.dp))
        }
    }
}

// ── Detection Method Section ─────────────────────────────────────

@Composable
private fun DetectionMethodSection(resetTrigger: Int) {
    var useNN by remember(resetTrigger) {
        mutableStateOf(AppSettings.p().getBoolean(AppSettings.KEY_USE_NEURAL_NETWORK, AppSettings.DEF_USE_NEURAL_NETWORK))
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Psychology,
                    null,
                    tint = Color(0xFFAB47BC),
                    modifier = Modifier.size(24.dp)
                )
                Spacer(Modifier.width(12.dp))
                Text(
                    "Detection Method",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = Color(0xFFAB47BC)
                )
            }

            Spacer(Modifier.height(12.dp))

            // Neural Network option
            DetectionMethodOption(
                title = "Neural Network (TCN)",
                description = "Uses a trained deep learning model to classify gestures. More accurate across different hand shapes and lighting conditions.",
                selected = useNN,
                onClick = {
                    useNN = true
                    AppSettings.setBoolean(AppSettings.KEY_USE_NEURAL_NETWORK, true)
                }
            )

            Spacer(Modifier.height(8.dp))

            // Legacy option
            DetectionMethodOption(
                title = "Legacy (Rule-based)",
                description = "Uses hand landmark ratios and heuristic rules to detect gestures. No extra model required. Allows fine-tuning via Hand Recognition and Swipe Detection settings.",
                selected = !useNN,
                onClick = {
                    useNN = false
                    AppSettings.setBoolean(AppSettings.KEY_USE_NEURAL_NETWORK, false)
                }
            )

            Spacer(Modifier.height(8.dp))

            // Note about fallback
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        color = Color(0xFF1B5E20).copy(alpha = 0.2f),
                        shape = RoundedCornerShape(8.dp)
                    )
                    .padding(10.dp),
                verticalAlignment = Alignment.Top
            ) {
                Icon(
                    Icons.Default.Shield,
                    null,
                    tint = GreenActive,
                    modifier = Modifier.size(16.dp).padding(top = 2.dp)
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    "If the neural network model fails to load, the app will automatically fall back to legacy detection.",
                    style = MaterialTheme.typography.bodySmall,
                    color = GreenActive,
                    lineHeight = 16.sp
                )
            }
        }
    }
}

@Composable
private fun DetectionMethodOption(
    title: String,
    description: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    val borderColor = if (selected) Color(0xFFAB47BC) else TextSecondary.copy(alpha = 0.3f)
    val bgColor = if (selected) Color(0xFFAB47BC).copy(alpha = 0.1f) else Color.Transparent

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = bgColor),
        border = CardDefaults.outlinedCardBorder().let {
            androidx.compose.foundation.BorderStroke(
                width = if (selected) 1.5.dp else 0.5.dp,
                color = borderColor
            )
        }
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.Top
        ) {
            RadioButton(
                selected = selected,
                onClick = onClick,
                colors = RadioButtonDefaults.colors(
                    selectedColor = Color(0xFFAB47BC),
                    unselectedColor = TextSecondary
                ),
                modifier = Modifier.size(20.dp)
            )
            Spacer(Modifier.width(10.dp))
            Column {
                Text(
                    title,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = if (selected) TextPrimary else TextSecondary
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    description,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                    lineHeight = 16.sp
                )
            }
        }
    }
}

// ── Collapsible Section ──────────────────────────────────────────

@Composable
private fun SettingsSection(
    title: String,
    icon: ImageVector,
    color: Color,
    key: Int,
    content: @Composable ColumnScope.() -> Unit
) {
    var expanded by remember(key) { mutableStateOf(true) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = DarkCard)
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { expanded = !expanded }
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(icon, null, tint = color, modifier = Modifier.size(24.dp))
                Spacer(Modifier.width(12.dp))
                Text(
                    title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = color,
                    modifier = Modifier.weight(1f)
                )
                Icon(
                    if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    contentDescription = if (expanded) "Collapse" else "Expand",
                    tint = TextSecondary
                )
            }

            AnimatedVisibility(visible = expanded) {
                Column(
                    modifier = Modifier.padding(
                        start = 16.dp, end = 16.dp, bottom = 16.dp
                    )
                ) {
                    content()
                }
            }
        }
    }
}

// ── Int Setting ──────────────────────────────────────────────────

@Composable
private fun IntSettingItem(
    label: String,
    description: String,
    key: String,
    default: Int,
    min: Int,
    max: Int,
    resetTrigger: Int
) {
    var text by remember(resetTrigger) {
        mutableStateOf(AppSettings.p().getInt(key, default).toString())
    }
    var isError by remember { mutableStateOf(false) }

    SettingRow(label = label, description = description) {
        OutlinedTextField(
            value = text,
            onValueChange = { newVal ->
                text = newVal
                val parsed = newVal.toIntOrNull()
                if (parsed != null && parsed in min..max) {
                    isError = false
                    AppSettings.setInt(key, parsed)
                } else {
                    isError = true
                }
            },
            isError = isError,
            supportingText = if (isError) {
                { Text("Range: $min - $max", color = RedStop, fontSize = 10.sp) }
            } else null,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.width(100.dp),
            textStyle = MaterialTheme.typography.bodyMedium.copy(color = TextPrimary),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Blue400,
                unfocusedBorderColor = TextSecondary.copy(alpha = 0.4f),
                cursorColor = Blue400
            )
        )
    }
}

// ── Long Setting ─────────────────────────────────────────────────

@Composable
private fun LongSettingItem(
    label: String,
    description: String,
    key: String,
    default: Long,
    min: Long,
    max: Long,
    resetTrigger: Int
) {
    var text by remember(resetTrigger) {
        mutableStateOf(AppSettings.p().getLong(key, default).toString())
    }
    var isError by remember { mutableStateOf(false) }

    SettingRow(label = label, description = description) {
        OutlinedTextField(
            value = text,
            onValueChange = { newVal ->
                text = newVal
                val parsed = newVal.toLongOrNull()
                if (parsed != null && parsed in min..max) {
                    isError = false
                    AppSettings.setLong(key, parsed)
                } else {
                    isError = true
                }
            },
            isError = isError,
            supportingText = if (isError) {
                { Text("Range: $min - $max", color = RedStop, fontSize = 10.sp) }
            } else null,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.width(100.dp),
            textStyle = MaterialTheme.typography.bodyMedium.copy(color = TextPrimary),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Blue400,
                unfocusedBorderColor = TextSecondary.copy(alpha = 0.4f),
                cursorColor = Blue400
            )
        )
    }
}

// ── Float Setting ────────────────────────────────────────────────

@Composable
private fun FloatSettingItem(
    label: String,
    description: String,
    key: String,
    default: Float,
    min: Float,
    max: Float,
    resetTrigger: Int
) {
    var text by remember(resetTrigger) {
        mutableStateOf(AppSettings.p().getFloat(key, default).toString())
    }
    var isError by remember { mutableStateOf(false) }

    SettingRow(label = label, description = description) {
        OutlinedTextField(
            value = text,
            onValueChange = { newVal ->
                text = newVal
                val parsed = newVal.toFloatOrNull()
                if (parsed != null && parsed in min..max) {
                    isError = false
                    AppSettings.setFloat(key, parsed)
                } else {
                    isError = true
                }
            },
            isError = isError,
            supportingText = if (isError) {
                { Text("Range: $min - $max", color = RedStop, fontSize = 10.sp) }
            } else null,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
            modifier = Modifier.width(100.dp),
            textStyle = MaterialTheme.typography.bodyMedium.copy(color = TextPrimary),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Blue400,
                unfocusedBorderColor = TextSecondary.copy(alpha = 0.4f),
                cursorColor = Blue400
            )
        )
    }
}

// ── String Setting ───────────────────────────────────────────────

@Composable
private fun StringSettingItem(
    label: String,
    description: String,
    key: String,
    default: String,
    resetTrigger: Int
) {
    var text by remember(resetTrigger) {
        mutableStateOf(AppSettings.p().getString(key, default) ?: default)
    }

    SettingRow(label = label, description = description) {
        OutlinedTextField(
            value = text,
            onValueChange = { newVal ->
                text = newVal
                if (newVal.isNotBlank()) {
                    AppSettings.setString(key, newVal)
                }
            },
            modifier = Modifier.width(160.dp),
            textStyle = MaterialTheme.typography.bodyMedium.copy(color = TextPrimary),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Blue400,
                unfocusedBorderColor = TextSecondary.copy(alpha = 0.4f),
                cursorColor = Blue400
            )
        )
    }
}

// ── Shared Row Layout ────────────────────────────────────────────

@Composable
private fun SettingRow(
    label: String,
    description: String,
    input: @Composable () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f).padding(end = 12.dp)) {
                Text(
                    label,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    color = TextPrimary
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    description,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                    lineHeight = 16.sp
                )
            }
            input()
        }
        Spacer(Modifier.height(4.dp))
        HorizontalDivider(
            color = TextSecondary.copy(alpha = 0.15f),
            thickness = 0.5.dp
        )
    }
}
