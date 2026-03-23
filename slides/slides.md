---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: #ffffff;
  }
  section.title-slide {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title-slide h1 {
    color: #e94560;
    font-size: 2em;
    margin-bottom: 0.1em;
  }
  section.title-slide h2 {
    color: #e8e8e8;
    font-size: 1em;
    font-weight: 400;
  }
  section.title-slide p {
    color: #b0b0b0;
    font-size: 0.9em;
  }
  section.section-title {
    background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  section.section-title h1 {
    color: #e94560;
    font-size: 2em;
  }
  section.section-title h2 {
    color: #cccccc;
    font-weight: 400;
    font-size: 1em;
  }
  h1 { color: #0f3460; font-size: 1.5em; }
  h2 { color: #e94560; font-size: 1.2em; }
  h3 { color: #16213e; font-size: 1.0em; }
  table { font-size: 0.7em; }
  th { background: #0f3460; color: white; }
  code { font-size: 0.5em; }
  pre { font-size: 0.6em; }
  img { max-height: 60%; }
  footer { font-size: 0.6em; color: #888; }
  blockquote {
    border-left: 4px solid #e94560;
    background: #f8f8f8;
    padding: 0.5em 1em;
    font-size: 0.85em;
  }
---

<!-- _class: title-slide -->

# GrabDrop
## Cross-Device Screenshot Transfer via Air Gesture Recognition

DSAI5201 — AI and Big Data Computing in Practice
Spring 2026

Speaker_A · Speaker_B · Speaker_C · Speaker_D

---

<!-- _class: section-title -->

# Part 1: Motivation & Overview
## Speaker_A

---

# The Problem: Cross-Device Screenshot Sharing

**Current pain points** when sharing screenshots between phone and laptop:

| Method | Steps | Issues |
|--------|-------|--------|
| Chat apps (WeChat, Telegram) | Screenshot → Open app → Select contact → Send → Open on other device | 5+ steps, needs internet |
| Email | Screenshot → Compose → Attach → Send → Download | Slow, cumbersome |
| Cloud storage | Screenshot → Upload → Switch device → Download | Requires account setup |
| USB cable | Plug in → Find file → Copy | Physical tether |

> **Key insight:** All existing methods require multiple manual steps and break the user's workflow.

---

# Inspiration: Huawei Air Gesture

Huawei introduced **Air Gesture** on select devices — grab content from screen and "drop" it to another device using hand gestures.

**Limitations of Huawei's approach:**
- Only works between Huawei devices (proprietary ecosystem)
- Requires specific Huawei hardware
- Not available on other Android OEMs or desktops

**Our goal — GrabDrop:**
- Make it work on **any Android device** (tested on Xiaomi)
- Bridge the gap between **Android phones and Linux/Mac/Windows laptops**
- Fully **open-source** implementation
- No Internet required — works over **local network (LAN) only**

---

# GrabDrop: System Overview

**Core idea:** Use hand gestures captured by front camera to grab and drop screenshots across devices.

```
  Device A (Sender)                              Device B (Receiver)
  ┌──────────────┐                               ┌──────────────┐
  │  Open Palm    │   GRAB = take screenshot      │              │
  │   → Fist      │                               │              │
  │               │ ── UDP broadcast offer ──►    │              │
  │               │                               │  Fist        │
  │               │                               │   → Open Palm│
  │               │   ◄── TCP download ────       │  RELEASE =   │
  │               │                               │  receive it  │
  └──────────────┘                               └──────────────┘
```

**No pairing. No cloud. No wires. Just gesture.**

---

# Demo Flow

```
Step 1: Both devices running GrabDrop on the same Wi-Fi network

Step 2: GRAB gesture on Device A (palm → fist)
        → Screenshot captured instantly
        → Broadcast availability to all nearby devices

Step 3: RELEASE gesture on Device B (fist → palm)
        → Download screenshot from Device A
        → Save to gallery / open in viewer

Total time: ~3 seconds
```

**Supported platforms:**
- Android (any device with Android 10+, tested on Xiaomi)
- Linux desktop (primary), macOS, Windows

---

<!-- _class: section-title -->

# Part 2: AI Algorithm Design
## Speaker_B

---

# Two-Stage Gesture Detection Pipeline

We designed a **two-stage pipeline** balancing power efficiency and detection accuracy:

```
┌───────────────────┐         ┌────────────────────┐
│   IDLE STAGE      │         │   WAKEUP STAGE     │
│                   │ trigger │                    │
│  Low power        ├────────►│  High precision    │
│  ~10 fps          │         │  ~30 fps           │
│  Detect presence  │         │  Track motion      │
│  of hand          │◄────────┤  Confirm gesture   │
│                   │ timeout │                    │
│                   │ or done │                    │
└───────────────────┘         └────────────────────┘
```

| Stage | FPS | Purpose | Duration |
|-------|-----|---------|----------|
| IDLE | 10 | Hand presence detection | Continuous |
| WAKEUP | 30 | Motion tracking & confirmation | ≤ 2 sec |

---

# Step 1: Hand Landmark Detection (MediaPipe)

We use **Google MediaPipe Hand Landmarker** (float16) to detect 21 hand landmarks:

```
        WRIST (0)
          │
    ┌─────┼─────┬──────┬──────┬──────┐
  THUMB  INDEX  MIDDLE  RING  PINKY
   (1)   (5)    (9)   (13)  (17)
    │     │      │      │      │
   (2)   (6)   (10)   (14)   (18)  ← PIP
    │     │      │      │      │
   (3)   (7)   (11)   (15)   (19)
    │     │      │      │      │
   (4)   (8)   (12)   (16)   (20)  ← TIP
```

| Property | Value |
|----------|-------|
| Model | MediaPipe Hand Landmarker (float16, ~12 MB) |
| Running mode | VIDEO (stateful temporal tracking) |
| Detection confidence | 0.3 (lowered for robustness) |
| Output | 21 normalized (x, y, z) landmarks per frame |

---

# Step 2: Hand State Classification

**Finger Curl Ratio** — a geometric metric to classify finger state:

$$\text{ratio} = \frac{d(\text{fingertip},\ \text{wrist})}{d(\text{finger\_mcp},\ \text{wrist})}$$

where $d$ is 2D Euclidean distance on normalized (x, y) coordinates.

| Condition | Interpretation |
|-----------|---------------|
| ratio > 1.3 | Finger **EXTENDED** (open) |
| ratio < 0.9 | Finger **CURLED** (closed) |
| otherwise | **AMBIGUOUS** (ignored) |

**Hand state rules** (4 fingers: index, middle, ring, pinky; thumb excluded):

| Condition | State |
|-----------|-------|
| >= 3 fingers extended | **PALM** (open hand) |
| >= 3 fingers curled | **FIST** (closed hand) |
| Otherwise | **UNKNOWN** |

---

# Step 3: Idle Stage — Low-Power Hand Detection

**Goal:** Detect when a hand is steadily present before entering high-FPS tracking.

- Maintains a **sliding window** of last 10 frames (~1 second at 10 fps)
- Counts occurrences of each hand state in the window

```
Window: [PALM, PALM, ?, PALM, PALM, PALM, PALM, PALM, ?, PALM]
         PALM count = 8 / 10  >=  threshold (8)
         → Enter WAKEUP stage, looking for FIST (GRAB gesture)
```

| Trigger | Wakeup Looks For | Gesture |
|---------|-----------------|---------|
| 8/10 frames = PALM | PALM → FIST transition | **GRAB** |
| 8/10 frames = FIST | FIST → PALM transition | **RELEASE** |

> Only ~5-8% CPU usage during IDLE — suitable for always-on background service.

---

# Step 4: Wakeup Stage — Gesture Confirmation

**Goal:** Confirm gesture with high precision using **consecutive-frame counting**.

```
Entry: PALM detected → looking for FIST (GRAB)

Frame  1:  PALM  streak=0  [--------]
Frame  4:  FIST  streak=1  [#-------]   ← fist appears
Frame  5:  FIST  streak=2  [##------]
Frame  6:  FIST  streak=3  [###-----]
Frame  7:  ???   streak=0  [--------]   ← broken! reset
Frame  8:  FIST  streak=1  [#-------]   ← restart
  ...
Frame 15:  FIST  streak=8  [########]   ← CONFIRMED
→ Emit GestureEvent.Grab
```

**Key design decisions:**
- **Consecutive** frames (not majority vote) — resets on any non-target frame
- Requires **8 consecutive frames** (~267 ms at 30 fps)
- **2-second timeout** — returns to IDLE if gesture not completed
- **3-second cooldown** after confirmed gesture prevents re-triggering

---

# Gesture Motion Classifier (Time-Series CNN)

*[Planned enhancement — currently using the rule-based pipeline]*

**Architecture concept:** A lightweight 1D CNN trained on temporal sequences of hand landmarks.

- **Input**: Time series of 21 landmark coordinates across N frames
- **Dataset**: Self-collected using a custom PyQt5 recording tool

| Label | Clips Recorded | Duration Range |
|-------|---------------|----------------|
| grab | 12 | 0.4 - 4.8 s |
| release | 9 | 0.4 - 4.8 s |
| swipe-up | 15 | 0.4 - 4.8 s |
| swipe-down | 6 | 0.4 - 4.8 s |

> We built a custom data capture tool that records labeled video clips with live camera preview and generates training metadata in CSV format. The heuristic pipeline already achieves reliable results; the CNN is planned for improved robustness.

---

<!-- _class: section-title -->

# Part 3: System Architecture & Implementation
## Speaker_C

---

# Overall System Architecture

```
┌──────────────────── DEVICE ────────────────────────┐
│                                                    │
│  ┌──────────┐   ┌───────────────┐   ┌──────────┐ │
│  │ Camera   │──►│ MediaPipe     │──►│ Two-Stage│ │
│  │ (CameraX │   │ Hand Landmark │   │ State    │ │
│  │ /OpenCV) │   │ Detector      │   │ Machine  │ │
│  └──────────┘   └───────────────┘   └────┬─────┘ │
│                                          │        │
│              GRAB ◄── GestureEvent ─────►│        │
│                │                   RELEASE│        │
│                ▼                         ▼        │
│  ┌──────────────────┐   ┌────────────────────┐   │
│  │ Screen Capture   │   │ Network Manager    │   │
│  │ (MediaProjection │   │ (UDP discovery +   │   │
│  │  / spectacle)    │   │  TCP transfer)     │   │
│  └──────────────────┘   └────────────────────┘   │
│                                                    │
│  ┌──────────────────┐   ┌────────────────────┐   │
│  │ Overlay Manager  │   │ Sound Player       │   │
│  │ (Visual feedback)│   │ (Audio feedback)   │   │
│  └──────────────────┘   └────────────────────┘   │
└────────────────────────────────────────────────────┘
```

---

# Android Implementation

**Foreground Service Architecture** — runs independently of the Activity lifecycle:

| Component | Technology | Role |
|-----------|-----------|------|
| GrabDropService | Foreground Service | Main orchestrator |
| RealGestureDetector | CameraX ImageAnalysis | Frame capture + state machine |
| HandLandmarkDetector | MediaPipe tasks-vision | 21-landmark detection |
| ScreenCaptureManager | MediaProjection + VirtualDisplay | Screenshot capture |
| NetworkManager | UDP multicast + TCP | Device discovery + transfer |
| OverlayManager | WindowManager overlay | Visual feedback animations |
| SwipeAccessibilityService | AccessibilityService | Programmatic swipe dispatch |

**Key engineering challenges solved:**
- Running CameraX inside a Service (custom ServiceLifecycleOwner)
- Persistent VirtualDisplay with frame buffering via AtomicReference
- Managing 12 Android permissions in correct sequence

---

# Desktop Implementation (Python)

**Modular Python client** — mirrors Android architecture:

| Module | Role | Platform Handling |
|--------|------|-------------------|
| main.py | Orchestrator (GrabDropDesktop) | — |
| gesture_detector.py | Camera + state machine | OpenCV VideoCapture |
| hand_landmark.py | MediaPipe wrapper + classifier | mediapipe Python SDK |
| screen_capture.py | Multi-backend screenshot | Auto-detect per desktop env |
| network_manager.py | UDP + TCP networking | Same protocol as Android |
| overlay.py | Visual feedback | Tkinter transparent windows |

**Screen capture auto-detection chain:**
```
spectacle (KDE) → grim (Wayland) → gnome-screenshot (GNOME)
→ scrot (X11) → mss (Python fallback) → import (ImageMagick)
```

> **Same gesture detection logic, same network protocol — full cross-platform interoperability.**

---

# Network Protocol Design

**Zero-configuration LAN protocol** — no server, no internet, no pairing:

| Phase | Transport | Details |
|-------|-----------|---------|
| **Discovery** | UDP multicast + broadcast (port 9877) | Heartbeat every 3s; timeout 10s |
| **Screenshot offer** | UDP broadcast | SCREENSHOT_READY with TCP port + file size |
| **Transfer** | TCP (dynamic port) | GET request → 4-byte length header → PNG data |

**Wire format:**
```
┌────────────────────┬──────────────────────────────┐
│ 4 bytes (uint32 BE)│  PNG image data              │
│ = file length      │  (variable)                  │
└────────────────────┴──────────────────────────────┘
```

**Retroactive matching:** If RELEASE occurs before the offer arrives due to network latency, the system records the timestamp and auto-matches within a 3-second window.

---

# Swipe Detection (Bonus Feature)

Beyond grab/release, the system also detects **vertical hand swipes** for remote page navigation:

**Detection algorithm** (during WAKEUP stage):
- Track hand center Y-position across frames
- Require >= 5 consecutive frames moving in same direction
- Displacement threshold: 0.12 (normalized screen height)
- Minimum velocity per frame: 0.008

| Gesture | Android Action | Desktop Action |
|---------|----------------|----------------|
| Swipe Up | AccessibilityService dispatch | PageUp key simulation |
| Swipe Down | AccessibilityService dispatch | PageDown key simulation |

Shorter cooldown (0.8s vs 3.0s) allows rapid consecutive swipes for scrolling through documents.

---

<!-- _class: section-title -->

# Part 4: Results, Limitations & Future Work
## Speaker_D

---

# Performance Results

### Latency Breakdown

| Phase | Duration |
|-------|----------|
| IDLE detection (trigger) | ~1.0 s (10 frames at 10 fps) |
| WAKEUP confirmation | ~0.3 - 1.5 s (8 frames at 30 fps) |
| Network transfer (LAN) | ~50 - 500 ms |
| **End-to-end (grab to receive)** | **~2 - 4 seconds** |

### Resource Usage

| Metric | IDLE Stage | WAKEUP Stage |
|--------|-----------|-------------|
| Camera FPS | 10 | 30 |
| ML inference / sec | 10 | 30 |
| CPU usage | ~5 - 8% | ~15 - 25% |
| WAKEUP duration | — | <= 2 seconds |

> The two-stage design keeps battery impact minimal for always-on operation.

---

# Strengths of Our Approach

1. **Cross-platform and cross-vendor**
   - Works on any Android 10+ device — not limited to one OEM
   - Desktop client supports Linux, macOS, Windows
   - Android and Desktop interoperability via shared protocol

2. **Zero-configuration networking**
   - No pairing, no cloud account, no internet required
   - Dual multicast + broadcast for maximum LAN compatibility

3. **Power-efficient design**
   - Two-stage pipeline: 10 fps idle, 30 fps wakeup
   - Wakeup lasts at most 2 seconds, then returns to low-power idle

4. **Robust gesture detection**
   - Geometric finger-curl ratio — not dependent on skin tone or lighting
   - Consecutive-frame confirmation prevents false positives
   - Retroactive matching handles network latency gracefully

---

# Limitations & Challenges

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Lighting sensitivity** | Low light reduces MediaPipe detection rate | Lowered confidence thresholds to 0.3 |
| **No encryption** | All data travels in plaintext on LAN | Suitable for trusted networks; TLS planned |
| **Single hand only** | One hand detected at a time | Sufficient for grab/release use case |
| **Gesture vocabulary** | Limited to grab, release, swipe | Extensible architecture for new gestures |
| **CNN not yet integrated** | Time-series classifier in development | Rule-based pipeline works reliably |
| **Camera angle** | Oblique angles distort finger ratios | Front camera + mirroring recommended |

---

# Future Work

1. **Time-series CNN classifier**
   - Train on self-collected dataset (42 clips and growing)
   - Replace rule-based heuristics for better edge-case handling
   - Explore temporal convolutional networks on landmark sequences

2. **Security enhancements**
   - TLS encryption for TCP transfer
   - Device pairing via QR code scan
   - Allow / deny list by device ID

3. **Extended gesture vocabulary**
   - Two-finger pinch, rotation, multi-hand gestures
   - Custom user-defined gesture bindings

4. **Other improvements**
   - Personalized calibration (per-user threshold tuning)
   - 3D depth from MediaPipe z-coordinates
   - Temporal smoothing (Kalman filter) on finger ratios
   - iOS client for full ecosystem coverage

---

# Summary

| Aspect | Our Contribution |
|--------|-----------------|
| **Problem** | Cross-device screenshot sharing requires too many manual steps |
| **Inspiration** | Huawei Air Gesture — limited to Huawei ecosystem |
| **Solution** | GrabDrop: open-source, cross-platform air gesture screenshot transfer |
| **AI Model** | MediaPipe Hand Landmarker + geometric finger-curl classifier |
| **Architecture** | Two-stage detection pipeline + zero-config LAN protocol |
| **Platforms** | Android (any device) + Linux / macOS / Windows desktop |
| **Result** | End-to-end transfer in ~3 seconds with minimal battery impact |

**All code is open source and available in the project repository.**

---

<!-- _class: title-slide -->

# Thank You
## Questions?

GrabDrop — Cross-Device Screenshot Transfer via Air Gesture

Speaker_A · Speaker_B · Speaker_C · Speaker_D
