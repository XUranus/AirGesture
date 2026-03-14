# GrabDrop — Desktop (Python)

> Command-line screenshot sharing client using hand gestures. 
> Compatible with GrabDrop Android.

## Overview

GrabDrop Desktop is a Python client that runs on Linux, macOS, and Windows. It uses your webcam to detect hand gestures and shares screenshots over your local network.

## Quick Start

```bash
# Clone and enter directory
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
- pip packages: `mediapipe`, `opencv-python`, `mss`, `Pillow`, `numpy`

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

18:00:00 │ Log file: logs/grabdrop_20240313_180000.log
18:00:00 │ ============================================================
18:00:00 │   GrabDrop Desktop Client Starting
18:00:00 │   Device ID: a3f8b2c1
18:00:00 │   Device Name: MyLaptop
18:00:00 │   Screenshots: /home/user/Pictures/GrabDrop
18:00:00 │ ============================================================
18:00:01 │ 📷 Camera opened: 640x480 @ 30fps
18:00:01 │ ✅ Using screenshot backend: spectacle
18:00:01 │ 👁️ IDLE stage — scanning for hand at ~10fps
18:00:05 │ 🔍 New device: Pixel_8 (192.168.1.42) — total nearby: 1
```

### Performing a GRAB (send screenshot)

1. Hold your **open palm 🖐** in front of the webcam for ~1 second
2. The indicator `✊` appears at top of screen
3. **Close your fist ✊** within 2 seconds
4. Screenshot is captured, saved, and broadcast to nearby devices

### Performing a RELEASE (receive screenshot)

1. When another device broadcasts a screenshot, the log shows the offer
2. Hold your **closed fist ✊** in front of the webcam for ~1 second
3. The indicator `🤚` appears at top of screen
4. **Open your hand 🖐** within 2 seconds
5. Screenshot is downloaded, saved, and opened in your image viewer

### Stop

Press `Ctrl+C` to stop cleanly.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    main.py                        │
│               (GrabDropDesktop)                   │
│                                                  │
│  ┌─────────────┐    ┌────────────────────────┐   │
│  │   Gesture    │    │   ScreenCapture        │   │
│  │   Detector   │    │   (spectacle/grim/     │   │
│  │   (OpenCV +  │    │    scrot/mss)          │   │
│  │    MediaPipe)│    │                        │   │
│  │              │    └────────────────────────┘   │
│  │  ┌────────┐  │                                 │
│  │  │ Hand   │  │    ┌────────────────────────┐   │
│  │  │Landmark│  │    │   NetworkManager       │   │
│  │  │Detector│  │    │   • UDP heartbeat      │   │
│  │  └────────┘  │    │   • UDP broadcast      │   │
│  └─────────────┘    │   • TCP transfer        │   │
│                      └────────────────────────┘   │
│  ┌─────────────┐    ┌────────────────────────┐   │
│  │  Overlay     │    │   SoundPlayer          │   │
│  │  (Tkinter)   │    │   (system sounds)      │   │
│  │  • Indicator │    │                        │   │
│  │  • Flash     │    └────────────────────────┘   │
│  │  • Thumbnail │                                 │
│  │  • Ripple    │                                 │
│  └─────────────┘                                  │
└──────────────────────────────────────────────────┘
```

## Project Structure

```
GrabDrop-Desktop/
├── main.py                 # Entry point + orchestration
├── config.py               # All configurable parameters
├── gesture_detector.py     # Camera + state machine (IDLE/WAKEUP)
├── hand_landmark.py        # MediaPipe Hands wrapper
├── screen_capture.py       # Multi-backend screen capture
├── network_manager.py      # UDP discovery + TCP transfer
├── overlay.py              # Tkinter overlay windows
├── sound_player.py         # Cross-platform sound effects
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── logs/                   # Runtime logs (created automatically)
    └── grabdrop_YYYYMMDD_HHMMSS.log
```

## Configuration

Edit `config.py` to tune parameters:

```python
# Camera
CAMERA_INDEX = 0          # 0 = default webcam, 1 = secondary
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Gesture detection
IDLE_FPS = 10             # Low-power scanning rate
WAKEUP_FPS = 30           # High-precision tracking rate
IDLE_WINDOW_SIZE = 10     # Frames in sliding window
IDLE_TRIGGER_THRESHOLD = 8  # 8/10 to trigger wakeup
WAKEUP_DURATION_S = 2.0   # Seconds to complete gesture
WAKEUP_CONFIRM_FRAMES = 8 # Consecutive frames to confirm

# Hand classification
FINGER_EXTENDED_THRESHOLD = 1.3
FINGER_CURLED_THRESHOLD = 0.9
MIN_FINGERS_FOR_PALM = 3
MIN_FINGERS_FOR_FIST = 3

# Network
UDP_PORT = 9877
MULTICAST_GROUP = "239.255.77.88"

# Paths
SCREENSHOT_DIR = ~/Pictures/GrabDrop
```

## Logs

Logs are written to both console (INFO+) and file (DEBUG+):

```
logs/grabdrop_20240313_180000.log
```

Log format:
```
18:00:05.123 [INFO ] GestureDetector    │ 🔔 WAKEUP! Detected 🖐 — watching 2s for 🖐→✊ (GRAB)
18:00:05.456 [DEBUG] HandLandmark       │ PALM e=4 c=0 conf=0.95 [IDX:1.52 MID:1.61 RNG:1.45 PNK:1.38]
```

## Screenshot Backend Auto-Detection

```
Session type detected → ordered preference list:

KDE Wayland:    spectacle → grim → scrot → mss → import
GNOME Wayland:  gnome-screenshot → grim → scrot → mss → import
X11:            scrot → mss → import
macOS:          mss (uses native screencapture internally)
Windows:        mss (uses native APIs)

If primary fails → automatically tries next backend
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No camera found" | Check `config.CAMERA_INDEX`, try 0/1/2 |
| "XGetImage() failed" | Install `spectacle` or `grim` for Wayland |
| Camera opens but NO_HAND | Check lighting, hold hand 30-60cm from camera |
| "Failed to open image" | Install `xdg-open` (Linux) or check default image viewer |
| Ctrl+C doesn't work | Should be fixed; if not, use `kill -9 <pid>` |
| "Address already in use" port 9877 | Another GrabDrop instance running; kill it |

## Cross-Platform Compatibility

Uses the same network protocol as GrabDrop Android. Both auto-discover each other on the same LAN via UDP heartbeats and transfer screenshots over TCP.

See `PROTOCOL.md` for protocol specification.

