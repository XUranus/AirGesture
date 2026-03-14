# GrabDrop — Android

> Share screenshots between devices using hand gestures over LAN.

## Overview

GrabDrop lets you **grab** a screenshot by closing your hand (palm → fist) and **release** it onto another device by opening your hand (fist → palm). Screenshots are transferred instantly over your local WiFi network — no cloud, no accounts, no pairing.

## Demo Flow

```
Device A (Sender)                    Device B (Receiver)
─────────────────                    ──────────────────

1. Show open palm 🖐 to camera       
   → IDLE detects palm (8/10 frames)
   → enters WAKEUP stage

2. Close fist ✊                      
   → GRAB confirmed!
   → Screenshot taken
   → Flash + thumbnail animation
   → UDP broadcast sent ──────────► 3. Offer received
                                       → "Screenshot available"

                                    4. Show closed fist ✊ to camera
                                       → IDLE detects fist
                                       → enters WAKEUP stage

                                    5. Open hand 🖐
                                       → RELEASE confirmed!
                                       → Looping ripple animation
                                       → TCP download starts

   TCP server sends PNG ──────────► 6. Screenshot saved
                                       → Ripple + sound
                                       → Gallery notification
```

## Requirements

- Android 10+ (API 29+)
- Front-facing camera
- WiFi connection (same LAN as other GrabDrop devices)
- ~50MB storage for MediaPipe model

## Permissions

| Permission | Why |
|---|---|
| `CAMERA` | Front camera for hand gesture detection |
| `FOREGROUND_SERVICE_CAMERA` | Camera access from background service |
| `FOREGROUND_SERVICE_MEDIA_PROJECTION` | Screen capture |
| `POST_NOTIFICATIONS` | Service notification + received screenshot alerts |
| `SYSTEM_ALERT_WINDOW` | Overlay animations (flash, ripple, wakeup indicator) |
| `INTERNET` / `ACCESS_WIFI_STATE` | LAN device discovery and file transfer |
| `WAKE_LOCK` | Keep service running while screen is off |

## Build & Install

### Prerequisites

- Android Studio Koala (2024.1+)
- JDK 17
- Kotlin 2.0+

### Steps

```bash
# Clone
git clone <repo-url>
cd GrabDrop

# Download MediaPipe model
mkdir -p app/src/main/assets
cd app/src/main/assets
wget -O hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
cd ../../..

# Build
./gradlew assembleDebug

# Install
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Or open in Android Studio

1. Open Android Studio → File → Open → select `GrabDrop/`
2. Sync Gradle
3. Download model (see above)
4. Run on physical device (emulator has limited camera/overlay support)

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MainActivity                       │
│                  (Jetpack Compose UI)                 │
│                                                      │
│  ┌────────────┐  ┌──────────┐  ┌────────────────┐   │
│  │ MainScreen │  │ StatusBar│  │  EventLogCard  │   │
│  │  Start/Stop│  │  Nearby  │  │  Live debug log│   │
│  │  Debug btn │  │  count   │  │                │   │
│  └────────────┘  └──────────┘  └────────────────┘   │
└──────────────────────┬───────────────────────────────┘
                       │ starts/stops
                       ▼
┌──────────────────────────────────────────────────────┐
│               GrabDropService                         │
│           (Foreground Service)                        │
│                                                      │
│  ┌─────────────────┐  ┌──────────────────────────┐   │
│  │ RealGesture      │  │ ScreenCaptureManager     │   │
│  │ Detector         │  │ (MediaProjection +       │   │
│  │ (CameraX +       │  │  persistent VirtualDisp) │   │
│  │  MediaPipe)      │  │                          │   │
│  │                  │  └──────────────────────────┘   │
│  │ ┌─────────────┐ │                                  │
│  │ │HandLandmark │ │  ┌──────────────────────────┐   │
│  │ │Detector     │ │  │ NetworkManager            │   │
│  │ │(MediaPipe   │ │  │ • UDP heartbeat (3s)     │   │
│  │ │ tasks-vision│ │  │ • UDP broadcast          │   │
│  │ └─────────────┘ │  │ • TCP server/client      │   │
│  └─────────────────┘  └──────────────────────────┘   │
│                                                      │
│  ┌─────────────────┐  ┌──────────────────────────┐   │
│  │ OverlayManager  │  │ SoundPlayer              │   │
│  │ • Wakeup 🔔     │  │ • Shutter click          │   │
│  │ • Flash ⚡      │  │ • Receive tone           │   │
│  │ • Thumbnail 🖼  │  │                          │   │
│  │ • Ripple 🌊     │  └──────────────────────────┘   │
│  │ • Looping ripple│                                  │
│  └─────────────────┘                                  │
└──────────────────────────────────────────────────────┘
```

## Project Structure

```
app/src/main/
├── assets/
│   └── hand_landmarker.task          # MediaPipe model (download separately)
├── java/com/grabdrop/
│   ├── GrabDropApp.kt                # Application class
│   ├── camera/
│   │   └── ServiceLifecycleOwner.kt  # LifecycleOwner for CameraX in Service
│   ├── capture/
│   │   ├── MediaStoreHelper.kt       # Save/load bitmaps to MediaStore
│   │   └── ScreenCaptureManager.kt   # MediaProjection screen capture
│   ├── gesture/
│   │   ├── GestureEvent.kt           # Grab/Release sealed class
│   │   ├── HandLandmarkDetector.kt   # MediaPipe wrapper
│   │   ├── HandState.kt              # NONE/PALM/FIST/UNKNOWN enum
│   │   ├── MockGestureDetector.kt    # Mock for testing
│   │   └── RealGestureDetector.kt    # CameraX + state machine
│   ├── network/
│   │   ├── NetworkManager.kt         # UDP/TCP networking
│   │   └── ScreenshotOffer.kt        # Offer data class
│   ├── overlay/
│   │   └── OverlayManager.kt         # WindowManager overlays
│   ├── service/
│   │   ├── GrabDropService.kt        # Main foreground service
│   │   ├── MediaProjectionHolder.kt  # In-process projection data
│   │   └── ServiceState.kt           # Observable state singleton
│   ├── ui/
│   │   ├── MainActivity.kt           # Entry point + permission flow
│   │   ├── MainScreen.kt             # Compose UI
│   │   └── theme/
│   │       ├── Color.kt
│   │       ├── Theme.kt
│   │       └── Type.kt
│   └── util/
│       ├── Constants.kt              # All configurable parameters
│       └── SoundPlayer.kt            # Audio feedback
└── res/
    ├── drawable/
    ├── values/strings.xml
    └── AndroidManifest.xml
```

## Debug Mode

Tap the 🐛 bug icon in the top bar to toggle debug mode:

| Feature | Debug OFF | Debug ON |
|---|---|---|
| Idle stage logs | Stage changes only | Every 10th frame with finger ratios |
| Wakeup stage logs | Confirm/timeout only | Every 5th frame with progress bar |
| Debug frames | Not saved | Saved to gallery every 30s |
| Stats | Not shown | Camera/processed/dropped counts every 10s |

## Configuration

All tunable parameters are in `Constants.kt`:

```kotlin
// Gesture timing
IDLE_FRAME_INTERVAL_MS = 100L      // ~10fps idle scanning
WAKEUP_FRAME_INTERVAL_MS = 33L    // ~30fps wakeup tracking
IDLE_WINDOW_SIZE = 10              // sliding window size
IDLE_TRIGGER_THRESHOLD = 8        // 8/10 frames to wake up
WAKEUP_DURATION_MS = 2_000L       // 2s wakeup window
WAKEUP_CONFIRM_FRAMES = 8         // 8 consecutive frames to confirm

// Hand classification
FINGER_EXTENDED_THRESHOLD = 1.3f  // tip/wrist vs mcp/wrist ratio
FINGER_CURLED_THRESHOLD = 0.9f
MIN_FINGERS_FOR_PALM = 3          // out of 4 (excl. thumb)
MIN_FINGERS_FOR_FIST = 3

// Network
UDP_PORT = 9877
MULTICAST_GROUP = "239.255.77.88"
HEARTBEAT_INTERVAL = 3s
DEVICE_TIMEOUT = 10s
SCREENSHOT_OFFER_TIMEOUT_MS = 10_000L
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "NO_HAND" on phone but works on tablet | Camera rotation differs | Enable debug mode → check debug frames in gallery |
| Service stops immediately | Missing permissions | Check notification + overlay + camera permissions |
| Screenshot always null | MediaProjection invalid | Restart app, re-grant screen capture consent |
| Nearby shows 0 | Not on same WiFi subnet | Check both devices on same network |
| Gallery won't open from background | Android 10+ background activity restriction | Tap the notification instead |
| Gesture too sensitive | Threshold too low | Increase `IDLE_TRIGGER_THRESHOLD` or `WAKEUP_CONFIRM_FRAMES` |
| Gesture too hard to trigger | Threshold too high | Decrease thresholds, check lighting |

## Cross-Platform Compatibility

GrabDrop Android is fully compatible with:
- **GrabDrop Desktop** (Python) — same network protocol
- Any device implementing the GrabDrop protocol (see `PROTOCOL.md`)

