// /GrabDrop/app/src/main/java/com/grabdrop/gesture/GestureEvent.kt
package com.grabdrop.gesture

sealed class GestureEvent {
    data object Grab : GestureEvent()
    data object Release : GestureEvent()
    data object SwipeUp : GestureEvent()
    data object SwipeDown : GestureEvent()
}
