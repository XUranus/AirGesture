# Gesture Recognition Module

A real-time hand gesture recognition system built with MediaPipe and PyTorch. It extracts 21 hand landmarks (63 values) per frame using MediaPipe's pre-trained Hand Landmarker, buffers 30 consecutive frames into a sliding window, and classifies the gesture with a custom-trained 1D-CNN temporal classifier. The fully-connected classification head is pruned via L1 unstructured pruning to reduce inference cost while keeping convolutional feature extraction intact.

## Project Structure

```

project/
├── train.py                 # Training + L1 pruning pipeline
├── inference.py             # Real-time / batch inference
├── hand_landmarker.task     # MediaPipe model (auto-downloaded)
├── data/
│   ├── Train/
│   │   ├── finger_heart/
│   │   ├── grab/
│   │   ├── noise/
│   │   ├── swipe_down/
│   │   ├── swipe_left/
│   │   ├── swipe_right/
│   │   ├── swipe_up/
│   │   └── wave/
│   └── Test/
│       └── (same subfolders)
└── checkpoints/             # Generated after training
├── gesture_cnn1d_best.pth
├── gesture_cnn1d_pruned.pth
├── config.json
└── cache/

````

## Supported Gestures

| Class         | Description        |
|---------------|--------------------|
| `grab`        | Grab / clench      |
| `swipe_up`    | Swipe up           |
| `swipe_down`  | Swipe / swipe down |
| `swipe_left`  | Swipe left         |
| `swipe_right` | Swipe right        |
| `finger_heart` | Finger heart       |
| `wave`        | Wave               |
| `noise`       | Idle / noise       |

## Requirements

- Python 3.8+
- PyTorch
- MediaPipe
- OpenCV
- scikit-learn
- scipy
- tqdm
- requests

```bash
pip install torch torchvision mediapipe opencv-python scikit-learn scipy tqdm requests
````

## Usage

### Train

```bash
python train.py --data_dir data --epochs 120 --batch_size 32
```

Key options:

| Flag             | Default       | Description                                         |
| ---------------- | ------------- | --------------------------------------------------- |
| `--data_dir`     | `data`        | Root data directory containing `Train/` and `Test/` |
| `--epochs`       | `120`         | Maximum training epochs                             |
| `--batch_size`   | `32`          | Batch size                                          |
| `--lr`           | `0.001`       | Learning rate                                       |
| `--save_dir`     | `checkpoints` | Output directory for model and config               |
| `--no_cache`     | off           | Force re-extraction of landmarks (skip cache)       |
| `--prune_amount` | `0.35`        | L1 pruning ratio for FC layers                      |

### Inference (webcam)

```bash
python inference.py
```

### Inference (video file)

```bash
python inference.py --video path/to/video.mp4
```

### Batch evaluation on test set

```bash
python inference.py --eval --data_dir data
```

### Force CPU

```bash
python inference.py --cpu
```

## Pipeline Overview

```
Video frames
    │
    ▼
MediaPipe Hand Landmarker (pre-trained, not fine-tuned)
    │
    ▼
21 landmarks × 3 coords = 63-dim vector per frame
    │
    ▼
Sliding window buffer (30 frames)
    │
    ▼
Input tensor (30, 63) → transpose → (63, 30)
    │
    ▼
1D-CNN feature extractor (Conv1d: 64→128→256→512)
    │
    ▼
FC classification head (512→256→128→8) ← L1 pruned
    │
    ▼
Softmax → class prediction + confidence
    │
    ▼
Temporal smoothing (majority vote over 5 predictions)
```

## Data Augmentation

Eight augmentation strategies are applied online during training:

| Method          | Description                            |
| --------------- | -------------------------------------- |
| Gaussian noise  | Add random noise (σ=0.005)             |
| Random scale    | Uniform scale in [0.85, 1.15]          |
| Random rotation | Z-axis rotation up to ±15°             |
| Random shift    | Translate x/y/z by up to ±0.05         |
| Time warp       | Non-linear temporal resampling         |
| Spatial flip    | Horizontal mirror (x → 1−x)            |
| Random mask     | Zero out up to 5 random landmarks      |
| Frame dropout   | Drop frames and interpolate neighbours |

## L1 Pruning

Only the fully-connected layers inside `model.classifier` are pruned. Convolutional layers in `model.features` remain intact to preserve feature extraction quality. Default pruning ratio is 35%. After pruning, masks are made permanent and the sparse model is saved.

## Extending to New Gestures

1. Create a new subfolder under `data/Train/` and `data/Test/` with the gesture name.
2. Add video clips (1–5 seconds, `.mp4` or `.avi`).
3. Update `CLASS_NAMES` in `train.py` (and `NUM_CLASSES` if the count changes).
4. Re-run `python train.py --no_cache`.

No changes are needed in the MediaPipe extraction layer or the inference communication layer.
