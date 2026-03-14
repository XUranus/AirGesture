// /GrabDrop/app/src/main/java/com/grabdrop/gesture/HandState.kt
package com.grabdrop.gesture

enum class HandState {
    NONE,   // no hand detected
    PALM,   // open palm — fingers extended
    FIST,   // closed fist — fingers curled
    UNKNOWN // hand detected but can't classify
}
