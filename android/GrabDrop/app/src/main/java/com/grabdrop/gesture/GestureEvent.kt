// /GrabDrop/app/src/main/java/com/grabdrop/gesture/GestureEvent.kt
package com.grabdrop.gesture

sealed class GestureEvent {
    data object Grab : GestureEvent()      // palm → fist
    data object Release : GestureEvent()   // fist → palm
}

