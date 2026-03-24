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
  .columns { display: flex; gap: 1em; }
  .columns > div { flex: 1; }
  .note { font-size: 0.75em; color: #666; }
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

# Project Foundation

**Topic:** Quantization and Pruning of Lightweight Vision Models

**Task:** Optimize models like YOLOv8 for efficient edge deployment without significant accuracy loss.

**Our approach:** Apply pruning and quantization techniques to a lightweight TCN gesture classifier.

---

# GrabDrop: System Overview

<div class="columns">
<div>

**Screenshot Transfer**
```
Device A (Sender)
┌──────────────────┐
│ Show hand        │
│   → Close fist   │
│                  │
│   GRAB           │
│   = screenshot   │
└──────────────────┘
         ▲
         │ UDP broadcast offer
         │ TCP download
         ▼
┌──────────────────┐
│ Device B         │
│ (Receiver)       │
│                  │
│ Show hand        │
│   → Open hand    │
│                  │
│   RELEASE        │
│   = TCP download │
│   → PNG saved    │
└──────────────────┘
```

</div>
<div>

**Page Navigation**
```
Device (Controller)
┌──────────────────┐
│ Show hand        │
│   → Move up      │
│                  │
│   SWIPE UP       │
│   = PageUp key   │
└──────────────────┘

┌──────────────────┐
│ Show hand        │
│   → Move down    │
│                  │
│   SWIPE DOWN     │
│   = PageDown key │
└──────────────────┘
```

</div>
</div>

---

# Demo Flow

<div class="columns">
<div>

**Screenshot Transfer**
```
1. Both devices on same Wi-Fi

2. Device A: Show hand ~1s
   → Close fist (GRAB)
   → Screenshot captured
   → UDP broadcast sent

3. Device B: Show hand ~1s
   → Open hand (RELEASE)
   → TCP download
   → Image saved & opened

Total: ~3 seconds
```

</div>
<div>

**Page Navigation**
```
1. Show hand to camera
   → IDLE detects hand
   → enters WAKEUP

2. Move hand upward
   → SWIPE UP detected
   → PageUp key sent

3. Move hand downward
   → SWIPE DOWN detected
   → PageDown key sent

Cooldown: 0.8s
```

</div>
</div>

**Supported platforms:** Android 10+ | Linux | macOS | Windows

---

<!-- _class: section-title -->

# Part 2: AI Algorithm Design
## Speaker_B

---

# Two-Stage Detection Pipeline

<div class="columns">
<div>

**IDLE Stage (10 FPS)**
- Low-power scanning
- Detect hand presence
- 8/10 frames with hand → WAKEUP

</div>
<div>

**WAKEUP Stage (30 FPS)**
- High-precision classification
- TCN runs on 30-frame window
- Timeout: **2s** → return to IDLE

</div>
</div>

```
┌───────────────────┐  hand   ┌────────────────────┐
│   IDLE            │ detected│   WAKEUP           │
│   ~10 fps         ├────────►│   ~30 fps          │
│   Scan for hand   │         │   Classify gesture │
│                   │◄────────┤   Emit event       │
└───────────────────┘ timeout └────────────────────┘
```

| Stage | FPS | CPU (est.) | Duration |
|-------|-----|------------|----------|
| IDLE | 10 | 5-8% | Continuous |
| WAKEUP | 30 | 15-25% | ≤2 sec |

---

# Hand Landmark Detection (MediaPipe)

**MediaPipe Hand Landmarker** — 21 3D landmarks per frame

```
        WRIST (0)
          │
    ┌─────┼─────┬──────┬──────┬──────┐
  THUMB  INDEX  MIDDLE  RING  PINKY
   (1)   (5)    (9)   (13)  (17)
```

| Property | Value |
|----------|-------|
| Model | MediaPipe Hand Landmarker (float16) |
| Size | ~12 MB |
| Confidence | 0.3 (lowered for robustness) |
| Output | 21 × (x, y, z) = 63 dims/frame |

---

# TCN Gesture Classifier

**Why GestureTCN?**
- Lightweight: 87K params vs YOLOv8's 3M-68M
- Real-time: <2ms inference on edge devices
- Demonstrates optimization principles applicable to larger models

**Key Properties:**

| Property | Value |
|----------|-------|
| Dataset | Self-collected: 127 train / 33 test videos (PyQt5 tool) |
| Model | Temporal Convolutional Network |
| File | gesture_tcn_pruned_quantized.onnx |
| Size | 0.17 MB (~170 KB) |
| Classes | grab, release, swipe_up, swipe_down, noise |
| Input | (1, 144, 30) — batch, features, time |
| Output | (1, 5) — logits for 5 classes |
| Runtime | ONNX Runtime |

---

# TCN Architecture

**Causal dilated convolutions** for real-time streaming:

```
Input (144 features × 30 frames)
        │
        ▼
┌───────────────┐    ┌───────────────┐    ┌─────────┐
│ Stem Conv1D   │───►│ TCN Blocks    │───►│ Head    │
│ 144 → 32 ch   │    │ Dilation 1,2,4│    │ 48 → 5  │
└───────────────┘    │ RF = 19 frames│    └─────────┘
                     └───────────────┘
```

**Key Design Choices:**
- **Causal convolutions:** No future information (real-time streaming)
- **Dilated convolutions:** Large receptive field with few parameters
- **Residual connections:** Stable training, gradient flow

**Receptive Field Analysis:**
- Block 1 (d=1): 3 frames
- Block 2 (d=2): 7 frames
- Block 3 (d=4): 15 frames
- Block 4 (d=1): **19 frames** (~0.6s at 30 FPS)

---

# Feature Engineering (144 dims)

<div class="columns">
<div>

| Feature Group | Dims | Purpose |
|---------------|------|---------|
| Normalized landmarks | 63 | Position invariant |
| Velocity | 63 | Motion direction |
| Wrist velocity | 3 | Global movement |
| Finger distances | 10 | Open vs closed |
| Finger angles | 5 | Curl state |

</div>
<div>

**Normalization Pipeline:**
```
Raw (63) → Wrist-relative
        → Palm-size normalized
        → Z-score
```

**Velocity:**
```
velocity[t] = landmarks[t] - landmarks[t-1]
```

</div>
</div>

---

# Model Optimization Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Original   │    │   Pruned    │    │  Quantized  │
│   FP32      │───►│    FP32     │───►│    INT8     │
│  87K params │    │  46K params │    │  46K params │
│  0.34 MB    │    │  0.18 MB    │    │  0.17 MB    │
└─────────────┘    └─────────────┘    └─────────────┘
                        │
                        ▼
              Fine-tune 100 epochs
```

---

# Structured Pruning

**Channel pruning with fine-tuning:**

| Config | Original | Pruned |
|--------|----------|--------|
| stem | 48 | 32 |
| mid | 48 | 32 |
| out | 64 | 48 |
| head | 32 | 24 |
| **Total params** | 87,077 | 45,877 |

**Design choices:**
- Round channels to multiple of 8 for SIMD efficiency
- Remove entire channels (structured), not individual weights
- Fine-tune 100 epochs with lower LR (1e-3)

---

# INT8 Quantization

**Post-training static quantization (PTQ) with calibration:**

**Affine quantization:** $q = \text{round}(r/s + z)$

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

quantize_static(
    model_input="gesture_tcn_pruned.onnx",
    model_output="gesture_tcn_pruned_quantized.onnx",
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
)
```

**Calibration:** Use representative data to determine scale $s$ and zero point $z$ for optimal quantization range.

---

<!-- _class: section-title -->

# Part 3: System Architecture
## Speaker_C

---

# Overall System Architecture

```
┌──────────────────── DEVICE ────────────────────────┐
│                                                    │
│  ┌──────────┐   ┌───────────────┐   ┌──────────┐ │
│  │ Camera   │──►│ MediaPipe     │──►│ Two-Stage│ │
│  │ CameraX/ │   │ Hand Landmark │   │ Pipeline │ │
│  │ OpenCV   │   │ Detector      │   │ IDLE→WAKE│ │
│  └──────────┘   └───────────────┘   └────┬─────┘ │
│                                          │        │
│     GRAB/RELEASE/SWIPE_UP/SWIPE_DOWN ◄───┤        │
│                                          ▼        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│
│  │Screen Capture│  │Network Mgr   │  │Input Mgr ││
│  │MediaProjection│  │UDP discovery│  │PageUp/Down││
│  │ /spectacle   │  │TCP transfer │  │Keys      ││
│  └──────────────┘  └──────────────┘  └──────────┘│
│                                                    │
│  ┌──────────────┐  ┌──────────────┐              │
│  │Overlay Mgr   │  │Sound Player  │              │
│  │Visual fb     │  │Audio fb      │              │
│  └──────────────┘  └──────────────┘              │
└────────────────────────────────────────────────────┘
```

---

# Android Implementation

| Component | Technology | Role |
|-----------|-----------|------|
| GrabDropService | Foreground Service | Main orchestrator |
| RealGestureDetector | CameraX ImageAnalysis | Frame capture + pipeline |
| HandLandmarkDetector | MediaPipe tasks-vision | 21-landmark detection |
| GestureClassifier | ONNX Runtime Android | TCN inference |
| ScreenCaptureManager | MediaProjection | Screenshot capture |
| NetworkManager | UDP multicast + TCP | Discovery + transfer |
| OverlayManager | WindowManager | Visual feedback |
| SwipeAccessibilityService | AccessibilityService | PageUp/Down dispatch |

**Key challenges:** CameraX in Service, 12 permissions, VirtualDisplay buffering

---

# Desktop Implementation (Python)

| Module | Role |
|--------|------|
| main.py | Orchestrator |
| gesture_detector.py | Camera + two-stage pipeline |
| gesture_classifier.py | TCN model wrapper (ONNX) |
| hand_landmark.py | MediaPipe wrapper |
| screen_capture.py | Multi-backend (spectacle/grim/scrot/mss) |
| network_manager.py | UDP + TCP |
| overlay.py | Tkinter visual feedback |

```
Screen capture chain:
spectacle(KDE) → grim(Wayland) → gnome-screenshot → scrot(X11) → mss
```

---

# Network Protocol

**Zero-configuration LAN** — no pairing, no cloud

| Phase | Transport | Details |
|-------|-----------|---------|
| Discovery | UDP multicast (239.255.77.88:9877) | Heartbeat 3s, timeout 10s |
| Screenshot offer | UDP broadcast | TCP port + file size |
| Transfer | TCP | 4-byte length header + PNG |

**Retroactive matching:** RELEASE before offer → matched within 3s window

---

<!-- _class: section-title -->

# Part 4: Results & Future Work
## Speaker_D

---

# Optimization Results

| Metric | Original | Pruned | Pruned+INT8 |
|--------|----------|--------|-------------|
| **Params** | 87,077 | 45,877 | 45,877 |
| **Size** | 0.34 MB | 0.18 MB | 0.17 MB |
| **Compression** | 1.0× | 1.9× | 2.0× |
| **Accuracy** | 88.89% | 92.59% | 92.59% |
| **F1-Score** | 0.888 | 0.929 | 0.929 |
| **Latency (CPU)** | 0.92 ms | 0.79 ms | 1.23 ms |
| **Throughput** | 1087/s | 1271/s | 816/s |

> **Surprising result:** Pruning improved accuracy by +3.7%!

---

# Why Did Pruning Improve Accuracy?

**Hypothesis: Pruning acts as implicit regularization**

```
Original model (87K params):
┌────────────────────────────────────────┐
│ • Overfitting to training distribution │
│ • Memorizing noise in training data    │
│ • Redundant paths dilute features      │
└────────────────────────────────────────┘
              │
              ▼ Pruning removes weak connections
┌────────────────────────────────────────┐
│ • Forced to learn robust features      │
│ • Smaller capacity = better generalization│
│ • Focus on most discriminative patterns │
└────────────────────────────────────────┘
```

**Similar findings:** Lottery Ticket Hypothesis (Frankle & Carbin, 2019); Pruned ResNets often generalize better

---

# Per-Class Performance

| True \ Pred | grab | release | swipe_up | swipe_down | noise |
|-------------|------|---------|----------|------------|-------|
| **grab** | 94% | 4% | 0% | 0% | 2% |
| **release** | 3% | 95% | 0% | 0% | 2% |
| **swipe_up** | 0% | 0% | 91% | 5% | 4% |
| **swipe_down** | 0% | 0% | 6% | 90% | 4% |
| **noise** | 2% | 1% | 3% | 2% | 92% |

**Observations:**
- grab/release: Similar motion, reversed in time (~3-4% confusion)
- swipe_up/down: Motion direction confusion (5-6%)

---

# Strengths

1. **Cross-platform** — Android 10+ + Linux/macOS/Windows
2. **Zero-config** — No pairing, no cloud, no internet
3. **Power-efficient** — Two-stage: 10fps idle, 30fps wakeup (≤2s)
4. **Optimized model** — Pruned + quantized TCN: 2× smaller, +3.7% accuracy
5. **Robust detection** — TCN handles varied hand shapes and lighting

---

# Limitations

| Limitation | Mitigation |
|------------|------------|
| Lighting sensitivity | Lowered confidence (0.3) |
| No encryption | TLS planned |
| Single hand only | Sufficient for use case |
| Camera angle | Front camera recommended |

---

# Future Work

1. **Apply to larger vision models** — YOLOv8 object detection optimization
2. **Advanced quantization** — QAT, mixed-precision (FP16 + INT8)
3. **Security** — TLS encryption, QR pairing
4. **Extended gestures** — Pinch, rotation, multi-hand
5. **iOS client** — Full ecosystem coverage

---

# Summary

| Aspect | Contribution |
|--------|--------------|
| **Problem** | Cross-device screenshot sharing — too many steps |
| **Solution** | GrabDrop: open-source, cross-platform air gesture |
| **AI Model** | MediaPipe + TCN (pruned & quantized) |
| **Optimization** | 30% pruning + INT8 PTQ: 2× smaller, +3.7% accuracy |
| **Platforms** | Android + Linux/macOS/Windows |
| **Result** | ~3s transfer, <2ms inference, 0.17MB model |

**All code is open source.**

---

<!-- _class: title-slide -->

# Thank You
## Questions?

GrabDrop — Cross-Device Screenshot Transfer via Air Gesture

Speaker_A · Speaker_B · Speaker_C · Speaker_D
