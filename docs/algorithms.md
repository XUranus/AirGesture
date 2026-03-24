# GrabDrop Gesture Detection -- Algorithm Documentation

**Version:** 2.0
**Detection Backends:** Neural Network (TCN) + Legacy (Rule-based)
**Detection Pipeline:** Two-stage (Idle -> Wakeup)

## 1. Overview

GrabDrop detects four hand gestures using a front-facing camera:

| Gesture | Motion | Emits |
|---|---|---|
| **GRAB** | Open palm -> Closed fist | `GestureEvent.Grab` -- take screenshot |
| **RELEASE** | Closed fist -> Open palm | `GestureEvent.Release` -- receive screenshot |
| **SWIPE UP** | Hand moves upward | `GestureEvent.SwipeUp` -- scroll page up |
| **SWIPE DOWN** | Hand moves downward | `GestureEvent.SwipeDown` -- scroll page down |

The system supports two detection backends:

| Mode | Description |
|---|---|
| **Neural Network (TCN)** | A Temporal Convolutional Network classifies gestures from a sliding window of hand landmark features via ONNX Runtime. Default on both platforms. |
| **Legacy (Rule-based)** | Finger curl ratios classify palm/fist; frame-to-frame velocity detects swipes. No extra model required. |

Both modes share the same **two-stage detection pipeline**:

```
+-------------------+         +--------------------+
|   IDLE STAGE      |         |   WAKEUP STAGE     |
|                   | trigger |                    |
|  Low power        +-------->|  High precision    |
|  ~10 fps          |         |  ~30 fps           |
|  Detect presence  |         |  Classify gesture  |
|  of hand          |<--------+  Confirm + emit    |
|                   | timeout |                    |
|                   | or done |                    |
+-------------------+         +--------------------+
```

### 1.1 Automatic Fallback

If Neural Network mode is selected but the ONNX model fails to load (missing file, missing runtime, etc.), the detector falls back to Legacy mode automatically with a warning log. The service continues without interruption.

### 1.2 Configuration

| Platform | How to select mode |
|---|---|
| Android | Settings screen -> Detection Method -> Neural Network or Legacy |
| Desktop | `config.py` -> `DETECTION_METHOD = "neural_network"` or `"legacy"` |

## 2. Hand Landmark Model: MediaPipe

Both detection backends use MediaPipe Hand Landmarker to extract 21 3D hand landmarks per frame.

### 2.1 Model Specification

| Property | Value |
|---|---|
| Model | MediaPipe Hand Landmarker |
| Variant | float16 (lite) |
| File | `hand_landmarker.task` (Android), auto-downloaded (Desktop) |
| Size | ~12 MB |
| Framework | Android: `tasks-vision` 0.10.14; Desktop: `mediapipe` 0.10.14 |
| Running Mode | `VIDEO` (stateful, uses temporal tracking) |
| Hands | 1 (single hand detection) |

### 2.2 Landmark Layout

```
        WRIST (0)
          |
    +-----+-----+---------+----------+----------+
    |     |     |         |          |          |
  THUMB  INDEX  MIDDLE    RING      PINKY
   (1)   (5)    (9)      (13)      (17)
    |     |     |         |          |
   (2)   (6)   (10)      (14)      (18)   <- PIP joints
    |     |     |         |          |
   (3)   (7)   (11)      (15)      (19)
    |     |     |         |          |
   (4)   (8)   (12)      (16)      (20)   <- TIP joints
```

Each landmark has:
- `x`: normalized [0, 1] horizontal position
- `y`: normalized [0, 1] vertical position
- `z`: depth relative to wrist

### 2.3 Confidence Thresholds

```
min_detection_confidence  = 0.3   # lower = more sensitive
min_tracking_confidence   = 0.3
model_complexity          = 0     # 0=lite, 1=full
```

Lower thresholds (0.3 vs default 0.5) improve detection in poor lighting at the cost of more false positives (handled by the two-stage pipeline).

## 3. Neural Network Detection (TCN)

### 3.1 Model Architecture

The gesture classifier is a pruned and quantized **Temporal Convolutional Network (TCN)** exported to ONNX format.

| Property | Value |
|---|---|
| File | `gesture_tcn_pruned_quantized.onnx` |
| Size | ~140 KB |
| Runtime | ONNX Runtime (Android 1.20.0, Desktop >= 1.17.0) |
| Input shape | `(1, 144, 30)` -- (batch, features, time) |
| Output shape | `(1, 5)` -- logits for 5 classes |
| Classes | `grab`, `release`, `swipe_up`, `swipe_down`, `noise` |

### 3.2 Feature Engineering (144 dimensions)

Each frame of raw landmarks (21 points x 3 coords = 63 values) is transformed into a 144-dimensional feature vector:

| Feature Group | Dimensions | Description |
|---|---|---|
| Normalized landmarks | 63 | Landmarks relative to wrist, divided by palm size |
| Velocity | 63 | Frame-to-frame difference of normalized landmarks |
| Wrist velocity | 3 | Wrist position change between frames |
| Finger distances | 10 | Euclidean distances between all fingertip pairs |
| Finger angles | 5 | Bending angle (acos of joint vectors) per finger |

#### Normalization

```
wrist = landmarks[0]
mid_finger = landmarks[9]  # middle finger MCP
palm_size = distance(mid_finger, wrist)

normalized_landmark[i] = (landmark[i] - wrist) / palm_size
```

This makes features invariant to hand position and scale.

#### Velocity

```
velocity[t] = normalized_landmarks[t] - normalized_landmarks[t-1]
wrist_velocity[t] = wrist[t] - wrist[t-1]
```

Captures motion direction and speed, critical for swipe detection.

#### Finger Distances

For each pair of fingertips (10 pairs from 5 tips):

```
distance = ||norm_tip_i - norm_tip_j||
```

Distinguishes open hand (large distances) from fist (small distances).

#### Finger Angles

For each of the 5 fingers, the bending angle between two joint vectors:

```
v1 = joint[1] - joint[0]   # base to middle
v2 = joint[-1] - joint[1]  # middle to tip
angle = acos(dot(v1/|v1|, v2/|v2|))
```

Extended fingers have small angles (~0); curled fingers have large angles (~pi).

### 3.3 Input Normalization

Each of the 144 feature dimensions is z-score normalized using pre-computed statistics from the training set:

```
normalized_value = (raw_value - mean[d]) / (std[d] + 1e-8)
```

Mean and std arrays are stored in `config.json` alongside the model.

### 3.4 Sliding Window

```
+-- window (30 frames) ----------------------------------------+
| frame_0 | frame_1 | ... | frame_14 | frame_15 | ... | zeros |
| (real)  | (real)  |     | (real)   | (real)   |     | (pad) |
+--------------------------------------------------------------|

- Real frames are placed at the beginning
- Zeros pad the end until the window fills
- Classification starts after MIN_VALID_FRAMES = 15 real frames
- Oldest frames are dropped as new ones arrive (FIFO)
```

### 3.5 Classification

1. Build input tensor: shape `(1, 144, 30)` with normalized features
2. Run ONNX inference -> raw logits of shape `(1, 5)`
3. Apply softmax -> class probabilities
4. Select argmax class; if confidence >= 0.5 and class != `noise`, emit gesture event

### 3.6 IDLE Stage (TCN Mode)

In TCN mode, the IDLE stage wakes up when **any hand is detected** in enough frames, regardless of hand pose:

```
idle_window = [detected, detected, ..., none, detected, ...]
                                                 (10 frames)

if count(detected) >= IDLE_TRIGGER_THRESHOLD (8):
    -> enter WAKEUP
```

This differs from Legacy mode, which requires a specific pose (palm or fist).

### 3.7 WAKEUP Stage (TCN Mode)

1. Reset the classifier's sliding window
2. Feed each frame's raw landmarks to the classifier
3. Wait for the classifier to accumulate >= 15 frames
4. When a non-noise class exceeds the confidence threshold -> emit event
5. If 2 seconds pass without a confident prediction -> timeout back to IDLE

## 4. Legacy Detection (Rule-based)

### 4.1 Finger Curl Ratio

For each of the 4 fingers (index, middle, ring, pinky):

```
ratio = distance(fingertip, wrist) / distance(finger_mcp, wrist)

If ratio > 1.3  -> finger is EXTENDED
If ratio < 0.9  -> finger is CURLED
Otherwise       -> finger is AMBIGUOUS
```

The thumb is excluded because its movement axis is perpendicular to the other fingers.

### 4.2 Hand State Classification

```
extended_count = count of fingers with ratio > FINGER_EXTENDED_THRESHOLD
curled_count   = count of fingers with ratio < FINGER_CURLED_THRESHOLD

if extended_count >= MIN_FINGERS_FOR_PALM (3):  state = PALM
elif curled_count >= MIN_FINGERS_FOR_FIST (3):  state = FIST
else:                                           state = UNKNOWN
```

### 4.3 Example Classifications

```
Open Palm:
  INDEX: 1.52  MIDDLE: 1.61  RING: 1.45  PINKY: 1.38
  -> extended=4 >= 3 -> PALM

Closed Fist:
  INDEX: 0.65  MIDDLE: 0.58  RING: 0.62  PINKY: 0.55
  -> curled=4 >= 3 -> FIST

Ambiguous:
  INDEX: 1.10  MIDDLE: 0.95  RING: 0.88  PINKY: 0.72
  -> extended=0, curled=2 -> UNKNOWN
```

### 4.4 IDLE Stage (Legacy Mode)

The IDLE stage detects a **specific hand pose** consistently:

```
idle_window = [PALM, PALM, NONE, PALM, PALM, PALM, PALM, PALM, PALM, PALM]

palm_count = 8 >= IDLE_TRIGGER_THRESHOLD (8)
-> enter WAKEUP looking for FIST (GRAB gesture)

Similarly:
fist_count >= 8 -> enter WAKEUP looking for PALM (RELEASE gesture)
```

### 4.5 WAKEUP Stage (Legacy Mode)

Uses **consecutive-frame counting** to confirm a gesture:

```
Frame 1:  P  streak=0  [--------]
Frame 2:  P  streak=0  [--------]
Frame 3:  F  streak=1  [#-------]
Frame 4:  F  streak=2  [##------]
Frame 5:  ?  streak=0  [--------]  <- broken
Frame 6:  F  streak=1  [#-------]  <- restart
...
Frame 13: F  streak=8  [########]  <- CONFIRMED!

-> emit GestureEvent.Grab
-> return to IDLE
```

This requires the user to hold the target pose for ~267ms (8 frames at 30fps).

### 4.6 Swipe Detection (Legacy Mode)

Swipe detection runs concurrently with grab/release detection during the WAKEUP stage:

```
For each frame with a detected hand:
    1. Track center_y position
    2. Compute frame-to-frame velocity
    3. Count consecutive directional frames
    4. Compute cumulative displacement from start

Confirm swipe when:
    - consecutive_frames >= SWIPE_CONFIRM_FRAMES (5)  AND
    - total_displacement >= SWIPE_DISPLACEMENT_THRESHOLD (0.12)

    OR

    - trend analysis over recent frames shows displacement >= threshold * 1.2
```

Swipe has its own cooldown (`SWIPE_COOLDOWN_S = 0.8s`) separate from grab/release cooldown.

## 5. Two-Stage Pipeline Parameters

### 5.1 IDLE Stage

| Parameter | Default | Description |
|---|---|---|
| Frame rate | 10 FPS | Low-power scanning |
| Window size | 10 frames | 1 second at 10 FPS |
| Trigger threshold | 8/10 | Frames needed to trigger wakeup |

### 5.2 WAKEUP Stage

| Parameter | Default | Description |
|---|---|---|
| Frame rate | 30 FPS | High-precision tracking |
| Duration | 2 seconds | Max time to complete gesture |
| Confirm frames | 8 consecutive | Legacy mode: frames to confirm (267ms) |
| Confidence threshold | 0.5 | TCN mode: minimum softmax probability |

### 5.3 Cooldowns

| Parameter | Default | Description |
|---|---|---|
| Grab cooldown | 3 seconds | Minimum time between grabs |
| Swipe cooldown | 0.8 seconds | Minimum time between swipes (Legacy) |

## 6. Camera Input Processing

### 6.1 Frame Pipeline

```
Camera sensor
    |
    v
Raw frame (YUV/BGR/RGBA)
    |
    +-- Throttle (skip if too soon for current stage)
    |
    v
Color conversion (to RGB)
    |
    v
Rotation correction (if needed)
    |
    v
Horizontal mirror (front camera)
    |
    v
MediaPipe hand detection
    |
    v
21 landmarks (x, y, z per point)
    |
    +-- Extract raw_landmarks (63-dim float array)
    |
    v
Hand state classification (PALM/FIST/NONE/UNKNOWN)
    |
    v
Stage handler (IDLE or WAKEUP)
    |
    +-- [TCN mode] -> GestureClassifier.add_frame_and_classify()
    +-- [Legacy mode] -> consecutive frame counting + swipe velocity
```

### 6.2 Platform Differences

| Aspect | Android | Desktop |
|---|---|---|
| Camera API | CameraX `ImageAnalysis` | OpenCV `VideoCapture` |
| Frame format | RGBA_8888 via `ImageProxy` | BGR via `cv2.read()` |
| Color conversion | None (RGBA -> MediaPipe) | `cv2.cvtColor(BGR2RGB)` |
| Rotation | `ImageProxy.imageInfo.rotationDegrees` | None (webcam auto-rotates) |
| Mirror | Manual `Matrix.postScale(-1,1)` | `cv2.flip(frame, 1)` |
| Model format | `hand_landmarker.task` (TFLite bundle) | MediaPipe Python (auto-downloads) |
| TCN runtime | ONNX Runtime Android | onnxruntime (Python) |
| Configuration | In-app Settings screen (SharedPreferences) | `config.py` file |

## 7. Performance Characteristics

### 7.1 Latency Breakdown

```
IDLE stage:
  10 frames x 100ms = ~1.0s to trigger wakeup

WAKEUP stage (Legacy):
  Transition + 8 frames x 33ms = ~0.3-1.5s to confirm

WAKEUP stage (TCN):
  15 frames minimum x 33ms = ~0.5s before first classification
  Additional frames until confident prediction = ~0.5-1.5s

Total gesture latency: 1.3 - 2.5 seconds

Network transfer:
  Screenshot broadcast: ~10ms (UDP)
  TCP download: 50-500ms (depends on image size)
```

### 7.2 Resource Usage

```
IDLE stage (most of the time):
  Camera: 640x480 at 10fps
  ML inference: 10x MediaPipe per second
  CPU: ~5-8% (single core)

WAKEUP stage (max 2 seconds):
  Camera: 640x480 at 30fps
  ML inference: 30x MediaPipe + 30x ONNX (TCN mode) per second
  CPU: ~15-30% (single core)
  Duration: <= 2 seconds -> negligible battery impact
```

### 7.3 TCN Model Size

```
gesture_tcn_pruned_quantized.onnx: ~140 KB
config.json:                       ~8 KB
Total:                             ~148 KB
```

## 8. Tuning Guide

### 8.1 Making Detection MORE Sensitive

```python
# Lower the idle trigger
IDLE_TRIGGER_THRESHOLD = 6      # was 8 (6 out of 10 frames)

# Fewer consecutive frames needed (Legacy)
WAKEUP_CONFIRM_FRAMES = 5       # was 8

# Lower confidence threshold (TCN)
TCN_CONFIDENCE_THRESHOLD = 0.3  # was 0.5

# Wider classification bands (Legacy)
FINGER_EXTENDED_THRESHOLD = 1.2  # was 1.3
FINGER_CURLED_THRESHOLD = 1.0    # was 0.9
MIN_FINGERS_FOR_PALM = 2         # was 3
MIN_FINGERS_FOR_FIST = 2         # was 3
```

### 8.2 Making Detection LESS Sensitive

```python
IDLE_TRIGGER_THRESHOLD = 9       # was 8
WAKEUP_CONFIRM_FRAMES = 12      # was 8
TCN_CONFIDENCE_THRESHOLD = 0.7  # was 0.5
FINGER_EXTENDED_THRESHOLD = 1.5  # was 1.3
FINGER_CURLED_THRESHOLD = 0.7    # was 0.9
```

### 8.3 Choosing Between Modes

| Scenario | Recommended Mode |
|---|---|
| General use | Neural Network (default) |
| Model file missing / won't load | Legacy (automatic fallback) |
| Need fine-grained threshold control | Legacy |
| Poor lighting / unusual hand shapes | Neural Network |
| Minimal dependencies | Legacy (no ONNX runtime needed) |

## 9. TCN Training Pipeline

The TCN model was trained from recorded hand gesture data:

```
1. Record gestures     -> utils/recorder/
   (webcam + MediaPipe -> CSV of landmark sequences)

2. Preprocess          -> models/preprocess.ipynb
   (segment, label, extract 144-dim features, normalize)

3. Train               -> models/train.ipynb
   (TCN architecture, cross-entropy loss, 5-class classification)

4. Deploy              -> models/deploy.ipynb
   (prune, quantize, export to ONNX, generate config.json)
```

The exported model and config are placed in:
- Android: `app/src/main/assets/`
- Desktop: `assets/`

## 10. Algorithm Pseudocode Summary

### TCN Mode

```python
while running:
    frame = camera.read()
    frame = mirror(rotate(frame))
    landmarks = mediapipe.detect(frame)
    raw_63 = extract_raw_landmarks(landmarks)

    if stage == IDLE:
        hand_detected = (landmarks is not None)
        window.append(hand_detected)
        if count(window, True) >= 8:
            classifier.reset()
            enter_wakeup()

    elif stage == WAKEUP:
        result = classifier.add_frame_and_classify(raw_63)
        if result and result.confidence >= 0.5 and result.gesture != "noise":
            emit(result.gesture)
            enter_idle()
        elif elapsed > 2.0:
            enter_idle()  # timeout
```

### Legacy Mode

```python
while running:
    frame = camera.read()
    frame = mirror(rotate(frame))
    landmarks = mediapipe.detect(frame)
    state = classify_hand_state(landmarks)  # PALM / FIST / NONE / UNKNOWN

    if stage == IDLE:
        window.append(state)
        if count(window, PALM) >= 8:
            enter_wakeup(looking_for=FIST)
        elif count(window, FIST) >= 8:
            enter_wakeup(looking_for=PALM)

    elif stage == WAKEUP:
        # Check swipe first
        swipe = check_swipe_velocity(landmarks)
        if swipe:
            emit(swipe)
            enter_idle()
            return

        # Check grab/release
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
