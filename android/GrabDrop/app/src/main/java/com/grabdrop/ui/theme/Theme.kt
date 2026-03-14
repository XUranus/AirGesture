// /GrabDrop/app/src/main/java/com/grabdrop/ui/theme/Theme.kt
package com.grabdrop.ui.theme

import androidx.compose.material3.*
import androidx.compose.runtime.Composable

private val DarkColorScheme = darkColorScheme(
    primary = Blue600,
    onPrimary = androidx.compose.ui.graphics.Color.White,
    secondary = Cyan400,
    background = DarkSurface,
    surface = DarkCard,
    onBackground = TextPrimary,
    onSurface = TextPrimary,
    error = RedStop,
)

@Composable
fun GrabDropTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        typography = GrabDropTypography,
        content = content
    )
}

