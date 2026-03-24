# GrabDrop

> Share screenshots between devices using hand gestures over LAN -- no cloud, no accounts, no pairing.

## What It Does

GrabDrop lets you **grab** a screenshot by closing your hand (palm -> fist) and **release** it onto another device by opening your hand (fist -> palm). Screenshots transfer instantly over your local WiFi network. Swipe gestures (up/down) simulate page scrolling.

```
Device A (Sender)                    Device B (Receiver)
-----------------                    ------------------

1. Show hand to camera               
   -> hand detected                  
                                     
2. Close fist (GRAB)                 
   -> screenshot captured            
   -> UDP broadcast sent ----------> 3. Offer received & queued
                                     
                                     4. Show hand to camera
                                        -> hand detected
                                     
                                     5. Open hand (RELEASE)
                                        -> TCP download
                                        -> image saved & opened
```

## Gesture Detection

GrabDrop supports two detection backends, selectable per device:

| Mode | How It Works | Pros |
|---|---|---|
| **Neural Network (TCN)** | Temporal Convolutional Network classifies gestures from a 30-frame sliding window of 144-dim hand landmark features via ONNX Runtime | More accurate; handles varied hand shapes and lighting |
| **Legacy (Rule-based)** | Finger curl ratios classify palm/fist; frame-to-frame velocity detects swipes | No extra model needed; fully tunable thresholds |

Both modes use the same two-stage detection pipeline:

1. **IDLE** (~10 FPS) -- low-power scanning for hand presence
2. **WAKEUP** (~30 FPS) -- high-precision gesture classification for up to 2 seconds

If the neural network model fails to load, the app automatically falls back to legacy detection.

See [docs/algorithms.md](docs/algorithms.md) for full algorithm details.

## Platforms

### Android (`./android/`)

Jetpack Compose app with foreground service. Requires Android 10+ (API 29).

```bash
cd android/GrabDrop
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Configuration is done through an in-app Settings screen (tap the gear icon). See [android/README.md](android/README.md) for full documentation.

### Desktop (`./desktop/`)

Python CLI client. Runs on Linux, macOS, and Windows.

```bash
cd desktop
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Configuration is done by editing `config.py`. See [desktop/README.md](desktop/README.md) for full documentation.

## Cross-Platform Compatibility

Android and Desktop clients auto-discover each other on the same LAN and transfer screenshots seamlessly. They share the same:

- **UDP discovery** -- multicast (`239.255.77.88`) + broadcast on port `9877`
- **Screenshot offer** -- JSON message with TCP port and file size
- **TCP transfer** -- 4-byte big-endian length prefix + raw PNG data
- **Retroactive matching** -- RELEASE before offer is matched within a 3-second window

See [docs/network.md](docs/network.md) for the full protocol specification.

## Project Structure

```
.
+-- android/                    # Android app (Kotlin + Jetpack Compose)
|   +-- GrabDrop/               # Android Studio project
|   +-- README.md               # Android-specific documentation
+-- desktop/                    # Desktop client (Python)
|   +-- main.py                 # Entry point
|   +-- config.py               # All configurable parameters
|   +-- gesture_detector.py     # Dual-mode detector (NN + Legacy)
|   +-- gesture_classifier.py   # TCN model wrapper (ONNX Runtime)
|   +-- hand_landmark.py        # MediaPipe wrapper
|   +-- assets/                 # ONNX model + config.json
|   +-- README.md               # Desktop-specific documentation
+-- models/                     # ML training notebooks
|   +-- preprocess.ipynb        # Dataset preprocessing
|   +-- train.ipynb             # TCN model training
|   +-- deploy.ipynb            # Model export + quantization
+-- docs/                       # Shared documentation
|   +-- algorithms.md           # Gesture detection algorithms
|   +-- network.md              # Network protocol specification
+-- utils/                      # Utility tools
+-- slides/                     # Presentation materials
+-- report/                     # Project report
```

## Key Dependencies

| Component | Android | Desktop |
|---|---|---|
| Hand detection | MediaPipe tasks-vision 0.10.14 | mediapipe 0.10.14 |
| TCN inference | ONNX Runtime Android 1.20.0 | onnxruntime >= 1.17.0 |
| Camera | CameraX 1.3.4 | OpenCV 4.9+ |
| UI framework | Jetpack Compose (Material 3) | Tkinter (overlay) |
| Screen capture | MediaProjection | spectacle/grim/scrot/mss |

## Documentation

- [Gesture Detection Algorithms](docs/algorithms.md) -- TCN and legacy detection pipelines, feature engineering, tuning guide
- [Network Protocol](docs/network.md) -- UDP discovery, TCP transfer, wire format, implementation guide
- [Android README](android/README.md) -- Build, permissions, settings, architecture
- [Desktop README](desktop/README.md) -- Quick start, config reference, troubleshooting
