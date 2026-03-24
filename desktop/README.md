# GrabDrop -- Desktop (Python)

> Command-line screenshot sharing client using hand gestures.
> Compatible with GrabDrop Android.

## Overview

GrabDrop Desktop is a Python client that runs on Linux, macOS, and Windows. It uses your webcam to detect hand gestures and shares screenshots over your local network.

The app supports two gesture detection backends:

- **Neural Network (TCN)** -- A trained Temporal Convolutional Network classifies gestures from a sliding window of hand landmarks via ONNX Runtime. More accurate across varying hand shapes and lighting.
- **Legacy (Rule-based)** -- Uses hand landmark ratios and velocity heuristics. No extra model required and fully tunable via `config.py`.

Set `DETECTION_METHOD` in `config.py` to choose. If the neural network model fails to load, the app automatically falls back to legacy detection.

## Quick Start

```bash
# Enter directory
cd GrabDrop-Desktop

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

## Requirements

### Python

- Python 3.9+
- pip packages: `mediapipe`, `opencv-python`, `mss`, `Pillow`, `numpy`, `onnxruntime`

### System

| Platform | Screenshot Tool | Install |
|---|---|---|
| KDE (Wayland) | `spectacle` | Usually pre-installed; `sudo pacman -S spectacle` |
| GNOME (Wayland) | `gnome-screenshot` | `sudo apt install gnome-screenshot` |
| Wayland (generic) | `grim` | `sudo pacman -S grim` |
| X11 | `scrot` | `sudo apt install scrot` |
| macOS | Built-in `screencapture` | Pre-installed |
| Windows | `mss` (Python) | Included in pip deps |

### Hardware

- Webcam (built-in or USB)
- WiFi connection on the same LAN as other GrabDrop devices

## Usage

```
$ python main.py

18:00:00 | Log file: logs/grabdrop_20240313_180000.log
18:00:00 | ============================================================
18:00:00 |   GrabDrop Desktop Client Starting
18:00:00 |   Device ID: a3f8b2c1
18:00:00 |   Device Name: MyLaptop
18:00:00 |   Screenshots: /home/user/Pictures/GrabDrop
18:00:00 | ============================================================
18:00:01 | Camera opened: 640x480 @ 30fps
18:00:01 | TCN model loaded -- using Neural Network detection
18:00:01 | IDLE -- scanning at ~10fps (Neural Network)
```

### Performing a GRAB (send screenshot)

1. Hold your hand in front of the webcam for ~1 second
2. The wakeup indicator appears at top of screen
3. Make a grab gesture (close your fist) within 2 seconds
4. Screenshot is captured, saved, and broadcast to nearby devices

### Performing a RELEASE (receive screenshot)

1. When another device broadcasts a screenshot, the log shows the offer
2. Hold your hand in front of the webcam for ~1 second
3. Make a release gesture (open your hand) within 2 seconds
4. Screenshot is downloaded, saved, and opened in your image viewer

### Stop

Press `Ctrl+C` to stop cleanly.

## Architecture

```
+--------------------------------------------------+
|                    main.py                        |
|               (GrabDropDesktop)                   |
|                                                   |
|  +---------------+    +----------------------+   |
|  | Gesture        |    | ScreenCapture        |   |
|  | Detector       |    | (spectacle/grim/     |   |
|  | (OpenCV +      |    |  scrot/mss)          |   |
|  |  MediaPipe)    |    +----------------------+   |
|  |                |                               |
|  | +------------+ |    +----------------------+   |
|  | | Hand       | |    | NetworkManager       |   |
|  | | Landmark   | |    | - UDP heartbeat      |   |
|  | | Detector   | |    | - UDP broadcast      |   |
|  | +------------+ |    | - TCP transfer       |   |
|  |                |    +----------------------+   |
|  | +------------+ |                               |
|  | | Gesture    | |    +----------------------+   |
|  | | Classifier | |    | SoundPlayer          |   |
|  | | (ONNX)     | |    | (system sounds)      |   |
|  | | [NN mode]  | |    +----------------------+   |
|  | +------------+ |                               |
|  |                |    +----------------------+   |
|  | Modes:         |    | Overlay (Tkinter)    |   |
|  |  - Neural Net  |    | - Indicator / Flash  |   |
|  |  - Legacy      |    | - Thumbnail / Ripple |   |
|  +---------------+    +----------------------+   |
+--------------------------------------------------+
        ^
        | reads at startup
+--------------------------------------------------+
|                  config.py                        |
|  - DETECTION_METHOD ("neural_network" / "legacy") |
|  - Gesture timing / Hand recognition             |
|  - Swipe detection / Network config              |
|  - Camera / Paths / Overlay                      |
+--------------------------------------------------+
```

## Project Structure

```
GrabDrop-Desktop/
+-- main.py                 # Entry point + orchestration
+-- config.py               # All configurable parameters
+-- gesture_detector.py     # Dual-mode detector (NN + Legacy)
+-- gesture_classifier.py   # TCN model wrapper (ONNX Runtime)
+-- hand_landmark.py        # MediaPipe Hands wrapper + raw landmarks
+-- screen_capture.py       # Multi-backend screen capture
+-- network_manager.py      # UDP discovery + TCP transfer
+-- overlay.py              # Tkinter overlay windows
+-- sound_player.py         # Cross-platform sound effects
+-- requirements.txt        # Python dependencies
+-- README.md               # This file
+-- assets/
|   +-- gesture_tcn_pruned_quantized.onnx  # TCN model (~140 KB)
|   +-- config.json                        # TCN normalization + class config
+-- logs/                   # Runtime logs (created automatically)
    +-- grabdrop_YYYYMMDD_HHMMSS.log
```

## Gesture Detection

### Detection Modes

Set `DETECTION_METHOD` in `config.py`:

```python
# "neural_network" -- use TCN model via ONNX Runtime (default, more accurate)
# "legacy"         -- use rule-based landmark ratio heuristics (no model needed)
DETECTION_METHOD = "neural_network"
```

#### Neural Network (TCN) -- default

Uses a pruned and quantized Temporal Convolutional Network via ONNX Runtime.

- **Input**: 30-frame sliding window of 144-dimensional feature vectors (normalized landmarks, velocity, wrist velocity, finger distances, finger angles)
- **Output**: 5-class classification -- `grab`, `release`, `swipe_up`, `swipe_down`, `noise`
- **Confidence threshold**: 0.5 (configurable via `TCN_CONFIDENCE_THRESHOLD`)
- **IDLE stage**: Wakes up when any hand is present in enough frames
- **WAKEUP stage**: Feeds raw landmarks to TCN; emits gesture event when a non-noise class exceeds confidence

#### Legacy (Rule-based)

- **Hand classification**: Compares fingertip-to-wrist / knuckle-to-wrist ratios against thresholds
- **Grab/Release**: IDLE detects consistent palm or fist, WAKEUP confirms opposite state over consecutive frames
- **Swipe**: Tracks frame-to-frame Y-velocity and cumulative displacement

#### Automatic Fallback

If `DETECTION_METHOD = "neural_network"` but the ONNX model fails to load (missing file, missing `onnxruntime` package, etc.), the detector falls back to legacy mode automatically with a warning log. The service continues without interruption.

## Configuration

Edit `config.py` to tune parameters:

```python
# Detection method
DETECTION_METHOD = "neural_network"  # or "legacy"

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Gesture detection
IDLE_FPS = 10
WAKEUP_FPS = 30
IDLE_WINDOW_SIZE = 10
IDLE_TRIGGER_THRESHOLD = 8
WAKEUP_DURATION_S = 2.0
WAKEUP_CONFIRM_FRAMES = 8       # Legacy mode only

# Hand classification (Legacy mode)
FINGER_EXTENDED_THRESHOLD = 1.3
FINGER_CURLED_THRESHOLD = 0.9
MIN_FINGERS_FOR_PALM = 3
MIN_FINGERS_FOR_FIST = 3

# Swipe detection (Legacy mode)
SWIPE_DISPLACEMENT_THRESHOLD = 0.12
SWIPE_CONFIRM_FRAMES = 5
SWIPE_MIN_VELOCITY = 0.008
SWIPE_COOLDOWN_S = 0.8

# Neural network
TCN_CONFIDENCE_THRESHOLD = 0.5

# Network
UDP_PORT = 9877
MULTICAST_GROUP = "239.255.77.88"

# Paths
SCREENSHOT_DIR = ~/Pictures/GrabDrop
```

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| mediapipe | >= 0.10.14 | Hand landmark detection |
| opencv-python | >= 4.9.0 | Camera capture + image processing |
| onnxruntime | >= 1.17.0 | TCN model inference |
| mss | >= 9.0.1 | Cross-platform screen capture |
| Pillow | >= 10.0.0 | Image processing |
| numpy | >= 1.24.0 | Numerical operations |

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No camera found" | Check `config.CAMERA_INDEX`, try 0/1/2 |
| "XGetImage() failed" | Install `spectacle` or `grim` for Wayland |
| Camera opens but NO_HAND | Check lighting, hold hand 30-60cm from camera |
| "Failed to open image" | Install `xdg-open` (Linux) or check default image viewer |
| Ctrl+C doesn't work | Should be fixed; if not, use `kill -9 <pid>` |
| "Address already in use" port 9877 | Another GrabDrop instance running; kill it |
| TCN model failed to load | App falls back to Legacy automatically; install `onnxruntime` |
| Gesture too sensitive | Increase `IDLE_TRIGGER_THRESHOLD` or `WAKEUP_CONFIRM_FRAMES` in config.py |
| Gesture too hard to trigger | Decrease thresholds, check lighting |

## Cross-Platform Compatibility

Uses the same network protocol as GrabDrop Android. Both auto-discover each other on the same LAN via UDP heartbeats and transfer screenshots over TCP.
