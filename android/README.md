# GrabDrop -- Android

> Share screenshots between devices using hand gestures over LAN.

## Overview

GrabDrop lets you **grab** a screenshot by closing your hand (palm -> fist) and **release** it onto another device by opening your hand (fist -> palm). Screenshots are transferred instantly over your local WiFi network -- no cloud, no accounts, no pairing.

The app supports two gesture detection backends:

- **Neural Network (TCN)** -- A trained Temporal Convolutional Network classifies gestures from a sliding window of hand landmarks. More accurate across varying hand shapes and lighting.
- **Legacy (Rule-based)** -- Uses hand landmark ratios and velocity heuristics. No extra model required and fully tunable from the Settings screen.

Users choose their preferred method in Settings. If the neural network model fails to load, the app automatically falls back to legacy detection.

## Demo Flow

```
Device A (Sender)                    Device B (Receiver)
-----------------                    ------------------

1. Show hand to camera
   -> IDLE detects hand presence
   -> enters WAKEUP stage

2. Close fist (grab gesture)
   -> GRAB confirmed
   -> Screenshot taken
   -> Flash + thumbnail animation
   -> UDP broadcast sent ----------> 3. Offer received
                                       -> "Screenshot available"

                                    4. Show hand to camera
                                       -> IDLE detects hand
                                       -> enters WAKEUP stage

                                    5. Open hand (release gesture)
                                       -> RELEASE confirmed
                                       -> Looping ripple animation
                                       -> TCP download starts

   TCP server sends PNG ----------> 6. Screenshot saved
                                       -> Ripple + sound
                                       -> Gallery notification
```

## Requirements

- Android 10+ (API 29+)
- Front-facing camera
- WiFi connection (same LAN as other GrabDrop devices)
- ~50 MB storage for MediaPipe model
- ~140 KB additional for TCN model (bundled in assets)

## Permissions

| Permission | Why |
|---|---|
| `CAMERA` | Front camera for hand gesture detection |
| `FOREGROUND_SERVICE_CAMERA` | Camera access from background service |
| `FOREGROUND_SERVICE_MEDIA_PROJECTION` | Screen capture |
| `POST_NOTIFICATIONS` | Service notification + received screenshot alerts |
| `SYSTEM_ALERT_WINDOW` | Overlay animations (flash, ripple, wakeup indicator) |
| `INTERNET` / `ACCESS_WIFI_STATE` | LAN device discovery and file transfer |
| `CHANGE_WIFI_MULTICAST_STATE` | Multicast UDP for device discovery |
| `WAKE_LOCK` | Keep service running while screen is off |
| `READ_MEDIA_IMAGES` | Access saved screenshots in gallery |

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

# The TCN model (gesture_tcn_pruned_quantized.onnx) and config.json
# are already included in assets/.

# Build
./gradlew assembleDebug

# Install
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Or open in Android Studio

1. Open Android Studio -> File -> Open -> select `GrabDrop/`
2. Sync Gradle
3. Download MediaPipe model (see above)
4. Run on physical device (emulator has limited camera/overlay support)

## Architecture

```
+------------------------------------------------------+
|                    MainActivity                       |
|                  (Jetpack Compose UI)                 |
|                                                      |
|  +------------+  +----------+  +----------------+   |
|  | MainScreen |  | Settings |  |  EventLogCard  |   |
|  | Start/Stop |  | Screen   |  |  Live debug log|   |
|  | Debug/Gear |  | (config) |  |                |   |
|  +------------+  +----------+  +----------------+   |
+----------------------+-------------------------------+
                       | starts/stops
                       v
+------------------------------------------------------+
|               GrabDropService                         |
|           (Foreground Service)                        |
|                                                      |
|  +---------------------+  +----------------------+  |
|  | RealGestureDetector  |  | ScreenCaptureManager |  |
|  | (CameraX + MediaPipe)|  | (MediaProjection)    |  |
|  |                      |  +----------------------+  |
|  | +------------------+ |                            |
|  | | HandLandmark     | |  +----------------------+  |
|  | | Detector         | |  | NetworkManager       |  |
|  | | (MediaPipe)      | |  | - UDP heartbeat (3s) |  |
|  | +------------------+ |  | - UDP broadcast      |  |
|  |                      |  | - TCP server/client   |  |
|  | +------------------+ |  +----------------------+  |
|  | | GestureClassifier| |                            |
|  | | (ONNX Runtime)   | |  +----------------------+  |
|  | | [Neural Network] | |  | OverlayManager       |  |
|  | +------------------+ |  | - Wakeup indicator   |  |
|  |                      |  | - Flash / Thumbnail  |  |
|  | Detection modes:     |  | - Ripple animations  |  |
|  |  - Neural Network    |  +----------------------+  |
|  |  - Legacy (fallback) |                            |
|  +---------------------+  +----------------------+  |
|                            | SoundPlayer          |  |
|  +---------------------+  | - Shutter click      |  |
|  | SwipeAccessibility  |  | - Receive tone       |  |
|  | Service             |  +----------------------+  |
|  +---------------------+                            |
+------------------------------------------------------+
        ^
        |  reads at startup
+------------------------------------------------------+
|               AppSettings                             |
|         (SharedPreferences singleton)                 |
|  - Detection method (NN vs Legacy)                   |
|  - Gesture timing / Hand recognition                 |
|  - Swipe detection / Network config                  |
+------------------------------------------------------+
```

## Project Structure

```
app/src/main/
+-- assets/
|   +-- hand_landmarker.task              # MediaPipe model (download separately)
|   +-- gesture_tcn_pruned_quantized.onnx # TCN gesture classification model
|   +-- config.json                       # TCN model config (normalization, class names)
+-- java/com/grabdrop/
|   +-- GrabDropApp.kt                    # Application class (initializes AppSettings)
|   +-- camera/
|   |   +-- ServiceLifecycleOwner.kt      # LifecycleOwner for CameraX in Service
|   +-- capture/
|   |   +-- MediaStoreHelper.kt           # Save/load bitmaps to MediaStore
|   |   +-- ScreenCaptureManager.kt       # MediaProjection screen capture
|   +-- gesture/
|   |   +-- GestureClassifier.kt          # TCN model wrapper (ONNX Runtime)
|   |   +-- GestureEvent.kt               # Grab/Release/SwipeUp/SwipeDown sealed class
|   |   +-- HandLandmarkDetector.kt       # MediaPipe wrapper + raw landmark extraction
|   |   +-- HandState.kt                  # NONE/PALM/FIST/UNKNOWN enum
|   |   +-- RealGestureDetector.kt        # Dual-mode detector (NN + Legacy)
|   +-- network/
|   |   +-- NetworkManager.kt             # UDP/TCP networking
|   |   +-- ScreenshotOffer.kt            # Offer data class
|   +-- overlay/
|   |   +-- OverlayManager.kt             # WindowManager overlays
|   +-- service/
|   |   +-- GrabDropService.kt            # Main foreground service
|   |   +-- MediaProjectionHolder.kt      # In-process projection data
|   |   +-- ServiceState.kt               # Observable state singleton
|   |   +-- SwipeAccessibilityService.kt  # Accessibility service for screen swipes
|   +-- ui/
|   |   +-- MainActivity.kt               # Entry point + permission flow
|   |   +-- MainScreen.kt                 # Compose UI (main dashboard)
|   |   +-- SettingsScreen.kt             # Compose UI (configuration page)
|   |   +-- theme/
|   |       +-- Color.kt
|   |       +-- Theme.kt
|   |       +-- Type.kt
|   +-- util/
|       +-- AppSettings.kt                # SharedPreferences wrapper (all config)
|       +-- Constants.kt                  # Runtime-configurable parameters
|       +-- SoundPlayer.kt                # Audio feedback
+-- res/
    +-- drawable/
    +-- values/strings.xml
    +-- values/ids.xml                     # Custom view IDs
    +-- xml/accessibility_service_config.xml
    +-- AndroidManifest.xml
```

## Gesture Detection

### Detection Modes

The app supports two detection backends, selectable in Settings:

#### Neural Network (TCN) -- default

Uses a pruned and quantized Temporal Convolutional Network running via ONNX Runtime.

- **Input**: 30-frame sliding window of 144-dimensional feature vectors (normalized landmarks, velocity, wrist velocity, finger distances, finger angles)
- **Output**: 5-class classification -- `grab`, `release`, `swipe_up`, `swipe_down`, `noise`
- **Confidence threshold**: 0.5 (predictions below this are ignored)
- **IDLE stage**: Wakes up when any hand is detected in enough frames
- **WAKEUP stage**: Feeds raw landmarks to the TCN; emits gesture event when a non-noise class exceeds the confidence threshold

#### Legacy (Rule-based)

Uses hand landmark geometry computed by MediaPipe.

- **Hand classification**: Compares fingertip-to-wrist / knuckle-to-wrist distance ratios against configurable thresholds to classify each frame as PALM, FIST, UNKNOWN, or NONE
- **Grab/Release**: IDLE detects consistent palm or fist presence, then WAKEUP looks for the opposite state over consecutive frames
- **Swipe detection**: Tracks frame-to-frame Y-velocity and cumulative displacement against configurable thresholds
- **IDLE stage**: Requires `IDLE_TRIGGER_THRESHOLD` frames of a specific hand state (palm or fist) within the sliding window
- **WAKEUP stage**: Requires `WAKEUP_CONFIRM_FRAMES` consecutive frames of the target state to confirm

#### Automatic Fallback

If the user selects Neural Network but the ONNX model fails to load (missing file, incompatible runtime, etc.), the detector automatically falls back to Legacy mode and logs the fallback event. The service continues running without interruption.

## Settings

All parameters are configurable from the in-app Settings screen (tap the gear icon on the main screen). Settings are persisted via SharedPreferences and take effect the next time the service starts.

### Detection Method

| Option | Description |
|---|---|
| Neural Network (TCN) | Deep learning model for gesture classification (default) |
| Legacy (Rule-based) | Heuristic detection using landmark ratios; fully tunable below |

### Gesture Timing

| Setting | Default | Description |
|---|---|---|
| Idle Scan Rate | 10 FPS | Frame rate while scanning for hand presence |
| Active Tracking Interval | 33 ms | Frame interval during active gesture tracking (~30 FPS) |
| Detection Window Size | 10 | Number of recent frames for hand presence detection |
| Wake-up Sensitivity | 8 | Frames (out of window) needed to trigger wakeup |
| Gesture Time Limit | 2000 ms | Maximum time to complete a gesture after wakeup |
| Confirmation Frames | 8 | Consecutive frames to confirm gesture (Legacy only) |

### Hand Recognition

| Setting | Default | Description |
|---|---|---|
| Finger Extended Ratio | 1.3 | Min tip/knuckle ratio for extended finger |
| Finger Curled Ratio | 0.9 | Max ratio for curled finger |
| Min Fingers for Open Hand | 3 | Extended fingers needed to recognize a palm (out of 4) |
| Min Fingers for Fist | 3 | Curled fingers needed to recognize a fist (out of 4) |

### Swipe Detection

| Setting | Default | Description |
|---|---|---|
| Swipe Distance Threshold | 0.12 | Min normalized displacement for a swipe (Legacy only) |
| Swipe Confirmation Frames | 5 | Consecutive directional frames needed (Legacy only) |
| Minimum Swipe Speed | 0.008 | Min per-frame velocity for directional movement (Legacy only) |
| Swipe Cooldown | 800 ms | Min time between consecutive swipes |

### Network

| Setting | Default | Description |
|---|---|---|
| Discovery Port | 9877 | UDP port for device discovery (must match all devices) |
| Multicast Address | 239.255.77.88 | Multicast group for LAN discovery |
| Screenshot Offer Timeout | 10000 ms | How long a received offer stays valid |
| Grab Cooldown | 3000 ms | Min time between consecutive grabs |

### Reset

Tap "Reset All to Defaults" to restore every setting to its original value.

## Debug Mode

Tap the bug icon in the top bar to toggle debug mode:

| Feature | Debug OFF | Debug ON |
|---|---|---|
| Idle stage logs | Stage changes only | Every 10th frame with hand state details |
| Wakeup stage logs | Confirm/timeout only | Every 5th frame with classification progress |
| Debug frames | Not saved | Saved to gallery every 30s |
| Stats | Not shown | Camera/processed/dropped counts every 10s |

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| Jetpack Compose (BOM) | 2024.06.00 | UI framework (Material 3) |
| CameraX | 1.3.4 | Front camera access |
| MediaPipe tasks-vision | 0.10.14 | Hand landmark detection |
| ONNX Runtime Android | 1.20.0 | TCN model inference |
| Kotlin Coroutines | 1.8.1 | Async operations |
| AndroidX Lifecycle | 2.8.3 | Service lifecycle, Compose integration |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "NO_HAND" on phone but works on tablet | Camera rotation differs | Enable debug mode, check debug frames in gallery |
| Service stops immediately | Missing permissions | Check notification + overlay + camera permissions |
| Screenshot always null | MediaProjection invalid | Restart app, re-grant screen capture consent |
| Nearby shows 0 | Not on same WiFi subnet | Check both devices on same network |
| Gallery won't open from background | Android 10+ restriction | Tap the notification instead |
| Gesture too sensitive | Thresholds too low | Open Settings, increase Wake-up Sensitivity or Confirmation Frames |
| Gesture too hard to trigger | Thresholds too high | Open Settings, decrease thresholds or check lighting |
| TCN model not loading | Missing/corrupt ONNX file | App falls back to Legacy automatically; check logs |
| Swipe direction inverted | Expected behavior | Swipe maps to page scroll direction (swipe up = scroll content up) |

## Cross-Platform Compatibility

GrabDrop Android is fully compatible with:
- **GrabDrop Desktop** (Python) -- same network protocol
- Any device implementing the GrabDrop protocol (see `network.md`)
