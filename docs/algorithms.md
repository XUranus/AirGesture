# GrabDrop Gesture Detection — Model & Algorithm Documentation

**Version:** 1.0  
**Model:** MediaPipe Hand Landmarker (float16)  
**Detection Pipeline:** Two-stage (Idle → Wakeup)

## 1. Overview

GrabDrop detects two dynamic hand gestures using a front-facing camera:

| Gesture | Motion | Emits |
|---|---|---|
| **GRAB** | Open palm 🖐 → Closed fist ✊ | `GestureEvent.Grab` → take screenshot |
| **RELEASE** | Closed fist ✊ → Open palm 🖐 | `GestureEvent.Release` → receive screenshot |

The system uses a **two-stage detection pipeline** to balance battery life and accuracy:

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

## 2. Model: MediaPipe Hand Landmarker

### 2.1 Model Specification

| Property | Value |
|---|---|
| Model | MediaPipe Hand Landmarker |
| Variant | float16 (lite) |
| File | `hand_landmarker.task` |
| Size | ~12MB |
| Framework | Android: `tasks-vision` 0.10.14; Desktop: `mediapipe` 0.10.14 |
| Running Mode | `VIDEO` (stateful, uses temporal tracking) |
| Hands | 1 (single hand detection) |

### 2.2 Model Output

The model outputs **21 3D hand landmarks** per detected hand:

```
        WRIST (0)
          │
    ┌─────┼─────┬─────────┬──────────┬──────────┐
    │     │     │         │          │          │
  THUMB  INDEX  MIDDLE    RING      PINKY
   (1)   (5)    (9)      (13)      (17)
    │     │     │         │          │
   (2)   (6)   (10)      (14)      (18)   ← PIP joints
    │     │     │         │          │
   (3)   (7)   (11)      (15)      (19)
    │     │     │         │          │
   (4)   (8)   (12)      (16)      (20)   ← TIP joints
```

Each landmark has:
- `x`: normalized [0, 1] horizontal position
- `y`: normalized [0, 1] vertical position
- `z`: depth relative to wrist (not used in classification)

### 2.3 Confidence Thresholds

```python
# Android (HandLandmarker.HandLandmarkerOptions)
min_hand_detection_confidence  = 0.3    # lower = more sensitive
min_hand_presence_confidence   = 0.3
min_tracking_confidence        = 0.3

# Desktop (mp.solutions.hands.Hands)
min_detection_confidence       = 0.3
min_tracking_confidence        = 0.3
model_complexity               = 0      # 0=lite, 1=full
```

Lower thresholds (0.3 vs default 0.5) improve detection in poor lighting and at wider angles, at the cost of more false positives (handled by the two-stage pipeline).

## 3. Hand State Classification

### 3.1 Finger Curl Ratio

For each of the 4 fingers (index, middle, ring, pinky), we compute:

```
                tip (8,12,16,20)
                 ╲
                  ╲ d_tip = distance(tip, wrist)
                   ╲
    mcp (5,9,13,17) ╲
     ╲               ╲
      ╲ d_mcp         ╲
       ╲               ╲
        wrist (0) ──────╳

    ratio = d_tip / d_mcp
```

Where `distance()` is 2D Euclidean distance using normalized x,y coordinates:

```
distance(a, b) = sqrt((a.x - b.x)² + (a.y - b.y)²)
```

### 3.2 Classification Rules

```
ratio = dist(fingertip, wrist) / dist(finger_mcp, wrist)

If ratio > 1.3  → finger is EXTENDED  (tip is further from wrist than knuckle)
If ratio < 0.9  → finger is CURLED    (tip is closer to wrist than knuckle)
Otherwise       → finger is AMBIGUOUS (mid-position, not counted)
```

#### Extended finger (ratio > 1.3):
```
    tip
     ╲
      ╲  ← finger stretched out
       ╲
       mcp
        │
      wrist    →  d_tip >> d_mcp  →  ratio ≈ 1.4–1.8
```

#### Curled finger (ratio < 0.9):
```
       ╭─╮
    tip│  │mcp  ← finger folded in
       ╰──│
         wrist  →  d_tip << d_mcp  →  ratio ≈ 0.5–0.8
```

### 3.3 Hand State Determination

```python
extended_count = count(finger for finger in [INDEX, MIDDLE, RING, PINKY]
                       if ratio[finger] > 1.3)

curled_count   = count(finger for finger in [INDEX, MIDDLE, RING, PINKY]
                       if ratio[finger] < 0.9)

if extended_count >= 3:
    state = PALM        # open hand
elif curled_count >= 3:
    state = FIST        # closed hand
else:
    state = UNKNOWN     # ambiguous
```

**Note:** The thumb is excluded from classification because its movement axis is perpendicular to the other fingers, making the ratio metric less reliable.

### 3.4 Example Classifications

```
Open Palm 🖐:
  INDEX:  ratio=1.52  → EXTENDED ✓
  MIDDLE: ratio=1.61  → EXTENDED ✓
  RING:   ratio=1.45  → EXTENDED ✓
  PINKY:  ratio=1.38  → EXTENDED ✓
  → extended=4 ≥ 3 → PALM ✅

Closed Fist ✊:
  INDEX:  ratio=0.65  → CURLED ✓
  MIDDLE: ratio=0.58  → CURLED ✓
  RING:   ratio=0.62  → CURLED ✓
  PINKY:  ratio=0.55  → CURLED ✓
  → curled=4 ≥ 3 → FIST ✅

Mid-transition (ambiguous):
  INDEX:  ratio=1.10  → AMBIGUOUS
  MIDDLE: ratio=0.95  → AMBIGUOUS
  RING:   ratio=0.88  → CURLED ✓
  PINKY:  ratio=0.72  → CURLED ✓
  → curled=2 < 3, extended=0 < 3 → UNKNOWN
```

## 4. Two-Stage Detection Pipeline

### 4.1 Stage 1: IDLE

**Purpose:** Low-power hand presence detection  
**Frame Rate:** ~10 fps  
**Resolution:** 640×480

```
┌──────────────────────────────────────────┐
│  IDLE STAGE STATE MACHINE                 │
│                                          │
│  Sliding window: last 10 frames          │
│  [·, ·, 🖐,🖐,🖐, ·, 🖐,🖐,🖐,🖐]        │
│                                          │
│  Count PALM states in window:            │
│    palm_count = 8                        │
│    threshold  = 8                        │
│    8 ≥ 8 → TRIGGER WAKEUP!              │
│                                          │
│  Similarly for FIST:                     │
│    fist_count ≥ 8 → TRIGGER WAKEUP      │
│                                          │
│  If neither threshold met:              │
│    → continue scanning                   │
└──────────────────────────────────────────┘
```

#### Idle Parameters

| Parameter | Value | Justification |
|---|---|---|
| Window size | 10 frames | 1 second at 10fps — enough for stable detection |
| Trigger threshold | 8/10 | Allows 2 dropped/uncertain frames per second |
| Frame rate | 10 fps | Low enough for battery, high enough for responsiveness |

#### Idle → Wakeup Transition Rules

| Detected State | Wakeup Looks For | Gesture Name |
|---|---|---|
| PALM (8/10) | PALM → FIST transition | **GRAB** |
| FIST (8/10) | FIST → PALM transition | **RELEASE** |

### 4.2 Stage 2: WAKEUP

**Purpose:** High-precision motion tracking  
**Frame Rate:** ~30 fps  
**Duration:** 2 seconds (then timeout back to IDLE)

```
┌──────────────────────────────────────────────────────┐
│  WAKEUP STAGE STATE MACHINE                           │
│                                                      │
│  Entry: PALM detected → looking for FIST (GRAB)     │
│                                                      │
│  Frame 1:  🖐 streak=0  [░░░░░░░░]                   │
│  Frame 2:  🖐 streak=0  [░░░░░░░░]                   │
│  Frame 3:  ❓ streak=0  [░░░░░░░░]  ← ambiguous      │
│  Frame 4:  ✊ streak=1  [█░░░░░░░]  ← fist appears   │
│  Frame 5:  ✊ streak=2  [██░░░░░░]                    │
│  Frame 6:  ✊ streak=3  [███░░░░░]                    │
│  Frame 7:  ❓ streak=0  [░░░░░░░░]  ← broken!        │
│  Frame 8:  ✊ streak=1  [█░░░░░░░]  ← restart         │
│  Frame 9:  ✊ streak=2  [██░░░░░░]                    │
│  Frame 10: ✊ streak=3  [███░░░░░]                    │
│  Frame 11: ✊ streak=4  [████░░░░]                    │
│  Frame 12: ✊ streak=5  [█████░░░]                    │
│  Frame 13: ✊ streak=6  [██████░░]                    │
│  Frame 14: ✊ streak=7  [███████░]                    │
│  Frame 15: ✊ streak=8  [████████]  ← CONFIRMED! ✅   │
│                                                      │
│  → emit GestureEvent.Grab                            │
│  → return to IDLE                                    │
└──────────────────────────────────────────────────────┘
```

#### Wakeup Parameters

| Parameter | Value | Justification |
|---|---|---|
| Duration | 2 seconds | Natural gesture completion time |
| Frame rate | 30 fps | Smooth motion tracking, ~60 frames in window |
| Confirm threshold | 8 consecutive frames | ~267ms continuous — prevents accidental triggers |
| Timeout behavior | Return to IDLE | Saves battery, avoids stuck wakeup |

#### Confirmation Logic

The wakeup stage uses **consecutive-frame counting**, NOT majority voting:

```python
if current_frame_state == target_state:
    consecutive_count += 1
else:
    consecutive_count = 0  # reset on any non-target frame

if consecutive_count >= 8:
    gesture_confirmed = True
```

This requires the user to **hold** the target pose for ~267ms (8 frames at 30fps), which:
- Prevents flicker-based false positives
- Naturally matches human gesture timing
- Is more robust than majority voting (which could sum up non-consecutive detections)

### 4.3 Cooldown

After any confirmed gesture, a 3-second cooldown prevents re-triggering:

```
t=0.0  GRAB confirmed → cooldown starts
t=0.5  PALM detected → ignored (cooldown)
t=1.0  WAKEUP trigger → suppressed (cooldown)
t=3.0  Cooldown expires → normal detection resumes
```

## 5. Camera Input Processing

### 5.1 Frame Pipeline

```
Camera sensor
    │
    ▼
ImageProxy (YUV/RGBA)
    │
    ├── Check throttle (skip if too soon)
    │
    ▼
Convert to Bitmap (ARGB_8888)
    │
    ├── Handle row padding (rowStride > width × pixelStride)
    │
    ▼
Apply rotation (0°/90°/180°/270°)
    │
    ├── ImageProxy.imageInfo.rotationDegrees
    │
    ▼
Mirror horizontally (front camera)
    │
    ├── postScale(-1, 1) — flip X axis
    │
    ▼
MediaPipe detection
    │
    ▼
Hand landmarks (21 points)
    │
    ▼
State classification (PALM/FIST/NONE/UNKNOWN)
    │
    ▼
Stage handler (IDLE or WAKEUP)
```

### 5.2 Platform Differences

| Aspect | Android | Desktop |
|---|---|---|
| Camera API | CameraX `ImageAnalysis` | OpenCV `VideoCapture` |
| Frame format | RGBA_8888 via `ImageProxy` | BGR via `cv2.read()` |
| Color conversion | None (RGBA → MediaPipe) | `cv2.cvtColor(BGR2RGB)` |
| Rotation | `ImageProxy.imageInfo.rotationDegrees` | None (webcam auto-rotates) |
| Mirror | Manual `Matrix.postScale(-1,1)` | `cv2.flip(frame, 1)` |
| Model format | `hand_landmarker.task` (TFLite bundle) | MediaPipe Python (auto-downloads) |
| Running mode | `VIDEO` (stateful) | `process()` per frame |

## 6. Performance Characteristics

### 6.1 Latency Breakdown

```
Gesture Detection Latency:
──────────────────────────
IDLE stage:
  10 frames × 100ms = ~1.0s to trigger wakeup

WAKEUP stage:
  Transition + 8 frames × 33ms = ~0.3-1.5s to confirm

Total GRAB latency: 1.3 - 2.5 seconds

Total RELEASE latency: 1.3 - 2.5 seconds

Network transfer:
  Screenshot broadcast: ~10ms (UDP)
  TCP download: 50-500ms (depends on image size and LAN speed)
```

### 6.2 Power Consumption

```
IDLE stage (most of the time):
  Camera: 640×480 at 10fps
  ML inference: 10 per second
  CPU: ~5-8% (single core)
  
WAKEUP stage (max 2 seconds):
  Camera: 640×480 at 30fps
  ML inference: 30 per second
  CPU: ~15-25% (single core)
  Duration: ≤ 2 seconds → negligible battery impact
```

### 6.3 Accuracy Considerations

| Factor | Impact | Mitigation |
|---|---|---|
| Lighting | Low light → fewer detections | Lower confidence thresholds (0.3) |
| Camera angle | Oblique angle → distorted ratios | Front camera + mirroring |
| Hand size | Small hands → lower ratios | Adjustable thresholds per user |
| Skin tone | MediaPipe trained on diverse dataset | Generally robust |
| Background | Busy background → false detections | Two-stage pipeline filters noise |
| Speed of gesture | Too fast → skipped frames | 30fps wakeup captures ~267ms transitions |

## 7. Tuning Guide

### Making detection MORE sensitive:

```python
# Lower the idle trigger
IDLE_TRIGGER_THRESHOLD = 6      # was 8 (6 out of 10 frames)

# Fewer consecutive frames needed
WAKEUP_CONFIRM_FRAMES = 5       # was 8

# Wider classification bands
FINGER_EXTENDED_THRESHOLD = 1.2  # was 1.3
FINGER_CURLED_THRESHOLD = 1.0    # was 0.9

# Fewer fingers needed
MIN_FINGERS_FOR_PALM = 2         # was 3
MIN_FINGERS_FOR_FIST = 2         # was 3
```

### Making detection LESS sensitive (fewer false positives):

```python
IDLE_TRIGGER_THRESHOLD = 9       # was 8
WAKEUP_CONFIRM_FRAMES = 12      # was 8
FINGER_EXTENDED_THRESHOLD = 1.5  # was 1.3
FINGER_CURLED_THRESHOLD = 0.7    # was 0.9
MIN_FINGERS_FOR_PALM = 4         # was 3
MIN_FINGERS_FOR_FIST = 4         # was 3
```

## 8. Future Improvements

### 8.1 Thumb Analysis

Currently excluded. Could improve accuracy:
```python
# Thumb uses different landmarks and movement axis
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP  = 3
THUMB_TIP = 4

# Thumb extended: tip far from index MCP
# Thumb curled: tip near palm center
```

### 8.2 3D Depth

MediaPipe provides z-coordinates that could improve curl detection:
```python
# z < 0 means closer to camera (towards viewer)
# Curled fingers have tips with z > mcp.z
depth_ratio = tip.z / mcp.z  # could augment 2D ratio
```

### 8.3 Temporal Smoothing

Kalman filter or exponential moving average on finger ratios:
```python
smoothed_ratio = α × current_ratio + (1 - α) × previous_ratio
# α = 0.7 balances responsiveness and stability
```

### 8.4 Personalized Calibration

Run a calibration step where user performs PALM and FIST:
```python
user_palm_ratios = measure(user, pose="PALM")   # e.g., [1.4, 1.5, 1.3, 1.2]
user_fist_ratios = measure(user, pose="FIST")   # e.g., [0.6, 0.5, 0.7, 0.6]

# Set thresholds as midpoint ±margin
EXTENDED_THRESHOLD = mean(user_palm_ratios) - 0.1
CURLED_THRESHOLD = mean(user_fist_ratios) + 0.1
```

### 8.5 Alternative Models

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| MediaPipe Hands (current) | 12MB | 15ms/frame | Good |
| MediaPipe Hand Landmarker full | 18MB | 25ms/frame | Better |
| Custom TFLite gesture classifier | 2-5MB | 5ms/frame | Trainable |
| Google ML Kit Pose | 3MB | 10ms/frame | Body only |

A custom-trained classifier on top of hand landmarks could reduce false positives:
```
Hand Landmarks (21×3) → Dense(64) → Dense(32) → Softmax(PALM/FIST/NONE)
```

## 9. Algorithm Pseudocode Summary

```python
# Main loop
while running:
    frame = camera.read()
    frame = mirror(rotate(frame))
    
    landmarks = mediapipe.detect(frame)
    state = classify(landmarks)  # PALM / FIST / NONE / UNKNOWN
    
    if stage == IDLE:
        window.append(state)
        if count(window, PALM) >= 8:
            enter_wakeup(looking_for=FIST)
        elif count(window, FIST) >= 8:
            enter_wakeup(looking_for=PALM)
    
    elif stage == WAKEUP:
        if state == target:
            streak += 1
        else:
            streak = 0
        
        if streak >= 8:
            emit(GRAB if target==FIST else RELEASE)
            enter_idle()
        elif elapsed > 2.0:
            enter_idle()  # timeout
```

