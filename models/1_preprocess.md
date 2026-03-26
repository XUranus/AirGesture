# Stage 1: Data Preprocessing (1.preprocess.ipynb)

This document walks you through every section of the preprocessing notebook, explaining **what** each part does, **why** it is needed, and **how** the code works.

---

## Table of Contents

1. [Big Picture: What This Stage Does](#1-big-picture)
2. [Section 1-3: Environment Setup and Constants](#2-environment-setup-and-constants)
3. [Section 4: Preview Video Frames](#3-preview-video-frames)
4. [Section 5: Hand Landmark Detection (MediaPipe)](#4-hand-landmark-detection)
5. [Section 6: Missing Landmark Interpolation](#5-missing-landmark-interpolation)
6. [Section 7: Data Augmentation](#6-data-augmentation)
7. [Section 8: Video Processing and Sample Generation](#7-video-processing-and-sample-generation)
8. [Section 9-10: Dataset Loading with Caching](#8-dataset-loading-with-caching)
9. [Section 11: Normalization Statistics](#9-normalization-statistics)
10. [Section 12: Save Preprocessed Data](#10-save-preprocessed-data)
11. [Output Files Summary](#11-output-files-summary)

---

## 1. Big Picture

The preprocessing stage converts **raw gesture video clips** into a **numerical dataset** that a neural network can learn from. Here is the data flow:

```
Video (.mp4/.avi)
    |
    | [MediaPipe hand detection]
    v
Raw Landmarks per frame: shape (T, 63)
    |            T = number of frames
    |            63 = 21 landmarks x 3 coordinates (x, y, z)
    |
    | [Interpolation of missing frames]
    | [Resampling to fixed 30 frames]
    | [Data augmentation (training only)]
    v
Augmented Samples: shape (30, 63) each
    |
    | [Feature computation]
    v
Feature Vectors: shape (30, 144) each
    |
    | [Save to .npz cache]
    v
Ready for Stage 2 (Training)
```

---

## 2. Environment Setup and Constants

### Sections 1-2: Imports and Dataset Path

```python
# Key imports
import cv2          # OpenCV: reading video files frame by frame
import numpy as np  # NumPy: numerical array operations
from tqdm import tqdm  # Progress bars

from common.constants import *  # All project-wide constants
from common.utils import to_scalar, resample, to_raw_sequence, compute_features
```

The notebook can run either:
- **On Google Colab** (mounting Google Drive for the dataset)
- **Locally** (pointing to a local directory)

You must set `DATASET_PATH` to point to the folder containing `Train/` and `Test/` subdirectories.

### Section 3: Constants (from `common/constants.py`)

These are the core constants that define the data format:

```python
SEQ_LEN = 30          # Every gesture is resampled to exactly 30 frames
NUM_LANDMARKS = 21    # MediaPipe detects 21 hand key points
NUM_COORDS = 3        # Each landmark has x, y, z coordinates
RAW_DIM = 63          # = 21 * 3, total raw values per frame
FEATURE_DIM = 144     # Total engineered features per frame
NUM_CLASSES = 5       # grab, release, swipe_up, swipe_down, noise
MIN_VALID_RATIO = 0.10  # At least 10% of frames must have detected hands
```

**Why 30 frames?** Different videos have different lengths (some gestures are fast, some are slow). By resampling every video to exactly 30 frames, we ensure all inputs to the neural network have the same size. 30 frames is enough to capture gesture dynamics without being too large.

**Why 144 features?** See the [Feature Computation](#feature-computation-in-detail) section below.

---

## 3. Preview Video Frames

### Section 4: `preview_video_frames()`

This utility function displays evenly-spaced frames from a video for visual inspection:

```python
def preview_video_frames(video_path, num_frames=6, title=None):
    cap = cv2.VideoCapture(str(video_path))   # Open video file
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Pick 6 evenly-spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Read and display each frame
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Jump to frame
        ret, frame = cap.read()                 # Read it
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB for display
```

**Why this matters**: Before training a model, it's always good practice to **visually inspect your data**. This helps you confirm the videos are correctly organized and labeled.

---

## 4. Hand Landmark Detection

### Section 5: `HandDetector` class

This is the bridge between raw video pixels and the numerical data our model needs.

```python
class HandDetector:
    def __init__(self, static_mode=True, max_hands=1, min_conf=0.5):
        # Try MediaPipe Tasks API first (newer), fall back to Solutions API
        ...
    
    def detect(self, frame_bgr):
        # Input: one BGR frame from OpenCV
        # Output: numpy array of shape (63,) or None if no hand detected
```

**How it works step-by-step:**

1. **Receives a single video frame** (a 2D image in BGR format)
2. **Converts BGR to RGB** (MediaPipe expects RGB)
3. **Runs MediaPipe's hand landmark detector** on the frame
4. **Extracts 21 landmarks**, each with (x, y, z) coordinates
5. **Returns a flat array of 63 values** `[x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]`
6. **Returns `None`** if no hand is detected in the frame

**Key parameters:**
- `static_mode=True`: Treats each frame independently (better for offline processing)
- `max_hands=1`: Only detect one hand per frame
- `min_conf=0.5`: Confidence threshold -- ignore detections below 50% confidence

**What are the coordinate values?**
- `x, y`: Normalized to [0, 1] relative to image width and height
- `z`: Depth relative to wrist (negative = closer to camera)

---

## 5. Missing Landmark Interpolation

### Section 6: `interpolate_missing()`

In real videos, MediaPipe sometimes **fails to detect the hand** in certain frames (due to blur, occlusion, etc.). This function fills in those gaps:

```python
def interpolate_missing(lm_list):
    # Input: list of landmarks per frame, some may be None
    # Output: numpy array of shape (T, 63) with all gaps filled
```

**How interpolation works:**

Imagine you have landmark data for frames 1, 2, _, _, 5, 6, _, 8:

```
Frame:    1    2    3    4    5    6    7    8
Data:    [ok] [ok] [??] [??] [ok] [ok] [??] [ok]
```

The algorithm:
1. **Finds all valid frames** (where landmarks were detected)
2. **For gaps between valid frames**: Uses **linear interpolation** -- draws a straight line between the known values and fills in the missing points
3. **For gaps at the start/end**: Uses **linear extrapolation** -- extends the trend from the nearest two known points

For example, if frame 2 has x=0.3 and frame 5 has x=0.6:
- Frame 3 would get x=0.4
- Frame 4 would get x=0.5

**Why this is important**: The neural network expects a complete sequence. Missing frames would create discontinuities that confuse the model.

---

## 6. Data Augmentation

### Section 7: Augmentation Functions

**Data augmentation** artificially increases the size and diversity of the training set by creating modified copies of existing samples. This helps the model generalize better and avoid **overfitting** (memorizing the training data instead of learning general patterns).

Here are all the augmentation techniques used:

### 6.1 Mirror (`mirror_x`)

```python
def mirror_x(raw_seq):
    s[:, :, 0] = 1.0 - s[:, :, 0]   # Flip all x-coordinates
```

Mirrors the hand horizontally as if looking in a mirror. This is **only applied to non-swipe gestures** because mirroring swipe_up/swipe_down doesn't change their direction.

### 6.2 Rotation (`rotate_2d`)

```python
def rotate_2d(raw_seq, angle_deg):
    # Rotate all landmarks around the wrist by the given angle
```

Rotates the hand position by a small angle (-15 to +15 degrees) around the wrist. This simulates different camera angles or wrist orientations.

**The math**: Uses a 2D rotation matrix:
```
x' = cos(angle) * x - sin(angle) * y
y' = sin(angle) * x + cos(angle) * y
```

### 6.3 Scale (`scale_landmarks`)

```python
def scale_landmarks(raw_seq, factor):
    # Scale hand size by factor (0.85 to 1.15) around the wrist
```

Makes the hand appear slightly larger or smaller. This handles variations in distance from the camera.

### 6.4 Jitter (`add_jitter`)

```python
def add_jitter(raw_seq, sigma=0.003):
    noise = np.random.randn(*raw.shape) * sigma  # Small Gaussian noise
    return raw + noise
```

Adds tiny random noise to each coordinate. This simulates measurement uncertainty in the landmark detection.

### 6.5 Time Warp (`time_warp`)

```python
def time_warp(raw_seq):
    # Non-linearly stretch/compress time
```

Warps the time axis non-uniformly -- some parts of the gesture are sped up, others slowed down. This simulates natural variations in gesture speed.

### 6.6 Speed Change (`speed_change`)

```python
def speed_change(raw_seq):
    factor = np.random.uniform(0.8, 1.2)  # 80% to 120% speed
```

Uniformly speeds up or slows down the entire gesture by a random factor.

### 6.7 Time Reversal

```python
rev = raw[::-1].copy()  # Reverse the frame order
```

This is a clever augmentation: reversing a **grab** gesture in time looks like a **release**, and vice versa. Similarly, reversing **swipe_up** gives **swipe_down**. The code assigns the correct reversed class label.

### Summary of Augmentations Per Video

For each training video, the code generates:

| Augmentation | Copies | Description |
|-------------|--------|-------------|
| Base sample | 1 | Original resampled sequence |
| Time crops | up to 9 | Different start/end positions |
| Jitter | 3 | Random noise variants |
| Rotation | 6 | Angles: -15, -10, -5, +5, +10, +15 degrees |
| Scale | 4 | Factors: 0.85, 0.9, 1.1, 1.15 |
| Time warp | 2 | Non-linear time distortion |
| Speed change | 2 | Random speed modification |
| Mirror | 0-1 | Only for non-swipe classes |
| Time reversal | 1 | With corrected label |

This results in roughly **20-30 samples per video**, expanding the dataset significantly.

---

## 7. Video Processing and Sample Generation

### Section 8: `extract_video()` and `make_samples()`

These two functions tie everything together:

#### `extract_video(video_path, detector)`

```python
def extract_video(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret: break
        lms.append(detector.detect(frame))  # Detect hands in each frame
    return lms, total_frames
```

Opens a video and runs hand detection on every frame. Returns a list where each element is either a 63-dim landmark array or `None`.

#### `make_samples(lm_list, total_frames, class_name, is_train)`

This is the **core sample generation function**:

1. **Quality check**: At least 10% of frames must have detected hands (`MIN_VALID_RATIO = 0.10`), otherwise the video is rejected
2. **Interpolation**: Fill in missing frames using `interpolate_missing()`
3. **Resampling**: Convert variable-length sequence to exactly 30 frames using `resample()`
4. **Augmentation** (training only): Apply all augmentation techniques described above
5. **Returns**: Lists of samples and their corresponding class labels

---

## 8. Dataset Loading with Caching

### Sections 9-10: Cache System

Processing hundreds of videos through MediaPipe is slow (minutes to hours). The caching system avoids reprocessing:

```python
def load_dataset(data_dir, detector, cache_path=None, is_train=True):
    # 1. Try loading from cache first
    cached = try_load_cache(cache_path, is_train)
    if cached is not None:
        return cached        # Fast path: seconds
    
    # 2. If no cache, process all videos (slow path: minutes)
    for class_name in CLASS_NAMES:
        for video_file in class_dir.iterdir():
            lm_list, total = extract_video(video_file, detector)
            samples, labels = make_samples(lm_list, total, class_name, is_train)
            ...
    
    # 3. Save to cache for next time
    save_cache(cache_path, all_samples, all_labels, is_train)
```

**Cache format**: `.npz` files (NumPy compressed archives) containing:
- `samples`: array of shape `(N, 30, 63)` -- N samples, each with 30 frames of 63 raw landmarks
- `labels`: array of shape `(N,)` -- class index for each sample
- Metadata: `cache_version`, `sample_format`, dimensions, train/test flag

**Cache versioning**: The `CACHE_VERSION = "v3_fixed_swipe_reverse"` string ensures that when you change the preprocessing logic, old caches are automatically invalidated and rebuilt.

#### `sanitize_dataset()`

A safety function that validates every sample after loading:
- Ensures correct shape `(30, 63)`
- Removes any invalid samples
- Reports how many samples were skipped

---

## 9. Normalization Statistics

### Section 11: `compute_norm_stats()`

Neural networks train better when input features are **centered around zero** with **unit variance**. This section computes the mean and standard deviation of each feature across the entire training set.

```python
def compute_norm_stats(samples):
    # For each sample:
    #   1. Convert raw landmarks to 144-dim features
    #   2. Accumulate sum and sum-of-squares
    
    # After processing all samples:
    mean = total_sum / total_frames          # Per-feature mean
    std = sqrt(sum_squares / N - mean^2)     # Per-feature standard deviation
    
    return {"mean": mean, "std": std}  # Both shape (144,)
```

**How normalization is later applied** (in Stage 2):
```python
normalized_feature = (raw_feature - mean) / (std + 1e-8)
```

The `1e-8` prevents division by zero for features with zero variance.

**Why this matters**: Without normalization, features with large values (e.g., distances ~1.0) would dominate features with small values (e.g., angles ~0.01), making training unstable.

### Feature Computation in Detail

The `compute_features()` function in `common/utils.py` transforms 63 raw landmark values into 144 meaningful features:

#### Feature 1: Normalized Landmarks (63 dims)

```python
relative = landmarks - wrist       # Make all positions relative to wrist
palm_size = ||mid_finger - wrist|| # Measure palm size
normalized = relative / palm_size  # Divide by palm size
```

**Why**: This makes the features **translation-invariant** (hand position in frame doesn't matter) and **scale-invariant** (hand distance from camera doesn't matter).

#### Feature 2: Velocity (63 dims)

```python
velocity[t] = normalized[t] - normalized[t-1]  # Frame-to-frame difference
```

**Why**: Velocity captures **motion patterns**. A grab gesture has landmarks moving inward; a swipe has them moving laterally.

#### Feature 3: Wrist Velocity (3 dims)

```python
wrist_velocity[t] = wrist[t] - wrist[t-1]  # Wrist movement in absolute coords
```

**Why**: Captures overall hand movement direction, crucial for distinguishing swipe_up vs swipe_down.

#### Feature 4: Finger Pair Distances (10 dims)

```python
# 10 distances between pairs of fingertips:
# (thumb-index), (index-middle), (middle-ring), (ring-pinky),
# (thumb-middle), (thumb-ring), (thumb-pinky),
# (index-ring), (index-pinky), (middle-pinky)
```

**Why**: These distances directly measure how open or closed the hand is -- essential for grab/release detection.

#### Feature 5: Finger Curl Angles (5 dims)

```python
# For each of the 5 fingers, compute the angle between:
# - Vector from wrist to base of finger
# - Vector from base of finger to fingertip
angle = arccos(cos_angle)
```

**Why**: Curl angles capture whether each finger is extended or bent, providing complementary information to distances.

---

## 10. Save Preprocessed Data

### Section 12: Output

The notebook saves three types of files:

```python
# 1. Normalization statistics (PyTorch format)
torch.save(norm_stats, "checkpoints/norm_stats.pt")

# 2. Dataset metadata (JSON)
json.dump(dataset_info, "checkpoints/dataset_info.json")

# 3. Cached datasets (already saved during loading)
# checkpoints/cache/train_{CACHE_VERSION}.npz
# checkpoints/cache/test_{CACHE_VERSION}.npz
```

The notebook also prints the class distribution to verify the dataset is reasonably balanced:

```
Class distribution:
          grab: train= 612, test=  6
       release: train= 684, test=  6
      swipe_up: train= 471, test=  5
    swipe_down: train= 584, test=  5
         noise: train= 667, test=  7
```

---

## 11. Output Files Summary

| File | Format | Contents |
|------|--------|----------|
| `checkpoints/cache/train_*.npz` | NumPy | Training raw landmark sequences + labels |
| `checkpoints/cache/test_*.npz` | NumPy | Test raw landmark sequences + labels |
| `checkpoints/norm_stats.pt` | PyTorch | Per-feature mean and std arrays (shape 144 each) |
| `checkpoints/dataset_info.json` | JSON | Class names, dimensions, feature parameters |

These files are consumed by **Stage 2 (Training)** and **Stage 3 (Deployment)**.

---

**Next**: [Stage 2: Model Training](./2_train.md) -- How the TCNN model learns from this data.
