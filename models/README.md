# Gesture Recognition Model Training

This directory contains the model training pipeline for the GrabDrop hand gesture recognition system.

## Pipeline Overview

The training pipeline consists of three stages:

```
Raw Gesture Video Clips
        |
        v
+----------------------------+
| Stage 1: Preprocessing     |  (preprocess.ipynb)
| - Extract hand landmarks   |
|   from video via MediaPipe |
| - Data augmentation        |
| - Feature engineering      |
+----------------------------+
        |
        v
  Processed Dataset (.npz)
  + Normalization Stats (.pt)
        |
        v
+----------------------------+
| Stage 2: Training          |  (train.ipynb)
| - Build GestureTCN model   |
| - Train with early stopping|
| - Evaluate performance     |
+----------------------------+
        |
        v
  Trained Model Weights (.pth)
        |
        v
+----------------------------+
| Stage 3: Deployment        |  (deploy.ipynb)
| - Structured pruning       |
| - ONNX export              |
| - INT8 quantization        |
+----------------------------+
        |
        v
  Lightweight ONNX Model
  (ready for edge devices)
```

## Directory Structure

```
models/
  |-- preprocess.ipynb              # Stage 1: Data preprocessing
  |-- train.ipynb                   # Stage 2: Model training
  |-- deploy.ipynb                  # Stage 3: Model optimization & export
  |-- train_gesture_recognition.ipynb  # PPT demo notebook (not for training)
  |-- hand_landmarker.task          # MediaPipe hand landmark model
  |-- common/                       # Shared modules
  |   |-- __init__.py              # Package exports
  |   |-- constants.py             # Global constants (SEQ_LEN, classes, etc.)
  |   |-- log.py                   # Logging utilities
  |   |-- env.py                   # Environment detection (local/HPC/Colab)
  |   |-- utils.py                 # Helper functions
  |   |-- features.py              # Feature engineering
  |   |-- augmentation.py          # Data augmentation transforms
  |   |-- detection.py             # Hand landmark detection (MediaPipe)
  |   |-- model.py                 # GestureTCN model definition
  |   |-- dataset.py               # PyTorch Dataset and DataLoader
  |   |-- evaluation.py            # Model evaluation utilities
  |-- data/                         # Dataset (gitignored)
      |-- Train/
      |   |-- grab/
      |   |-- release/
      |   |-- swipe_up/
      |   |-- swipe_down/
      |   |-- noise/
      |-- Test/
          |-- grab/
          |-- release/
          |-- swipe_up/
          |-- swipe_down/
          |-- noise/
```

## Gesture Classes

| Class | Gesture | Description |
|-------|---------|-------------|
| `grab` | Grab | Hand closing from open palm to fist |
| `release` | Release | Hand opening from fist to open palm |
| `swipe_up` | Swipe Up | Hand swiping upward |
| `swipe_down` | Swipe Down | Hand swiping downward |
| `noise` | Noise | Invalid/unrecognized action |

## Model Architecture

The model uses a **Temporal Convolutional Network (TCN)** designed for sequential gesture classification:

- **Input**: 144-dimensional feature vector per frame, 30 frames per sequence
- **Architecture**: Causal 1D convolutions with residual connections
- **Output**: 5-class classification (grab, release, swipe_up, swipe_down, noise)

### Feature Engineering (144 dimensions)

| Feature Group | Dims | Description |
|---------------|------|-------------|
| Normalized landmarks | 63 | Landmarks relative to wrist, scale-invariant |
| Velocity | 63 | Frame-to-frame motion of normalized landmarks |
| Wrist velocity | 3 | Absolute wrist movement (critical for swipe detection) |
| Finger pair distances | 10 | Distances between fingertip pairs (hand openness) |
| Finger curl angles | 5 | Bending angle for each finger |

## Training Results

| Metric | Value |
|--------|-------|
| Training samples | 3,018 |
| Test samples | 29 |
| Best test accuracy | 89.66% |
| Best epoch | 9 |
| Early stopping epoch | 49 |
| F1 Score (Macro) | 0.8981 |
| F1 Score (Weighted) | 0.8954 |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| grab | 1.000 | 0.667 | 0.800 | 6 |
| release | 1.000 | 1.000 | 1.000 | 6 |
| swipe_up | 0.714 | 1.000 | 0.833 | 5 |
| swipe_down | 1.000 | 1.000 | 1.000 | 5 |
| noise | 0.857 | 0.857 | 0.857 | 7 |

## Model Optimization

| Model | Parameters | Size | Accuracy |
|-------|------------|------|----------|
| Original (FP32) | 87,077 | 0.34 MB | 89.66% |
| Pruned (FP32) | 45,877 | 0.18 MB | 89.66% |
| Original ONNX (FP32) | 87,077 | 0.09 MB | 89.66% |
| Pruned ONNX (FP32) | 45,877 | 0.09 MB | 89.66% |
| Pruned + Quantized (INT8 ONNX) | 45,877 | 0.14 MB | 89.66% |

**Compression ratio**: 1.90x parameter reduction (via structured pruning)

**Quantization Note**: The INT8 quantized model is slightly larger than the FP32 ONNX models due to QDQ node overhead. For small models, this metadata overhead can exceed the weight compression benefit. However, quantization is still valuable for **inference acceleration** on mobile hardware, where INT8 operations are typically 2-4× faster than FP32.

## Output Files

After running the pipeline, the following files are generated in `checkpoints/`:

```
checkpoints/
  |-- cache/                             # Cached preprocessed datasets
  |   |-- train_v3_fixed_swipe_reverse.npz  # Training data
  |   |-- test_v3_fixed_swipe_reverse.npz   # Test data
  |-- norm_stats.pt                      # Normalization statistics
  |-- dataset_info.json                  # Dataset metadata
  |-- gesture_tcn_best.pth               # Best model weights
  |-- gesture_tcn_structured_pruned.pth  # Pruned model weights
  |-- gesture_tcn_original.onnx          # Original ONNX model
  |-- gesture_tcn_pruned.onnx            # Pruned ONNX model
  |-- gesture_tcn_pruned_quantized.onnx  # INT8 quantized ONNX model
  |-- config.json                        # Deployment configuration
  |-- training_curves.png                # Training visualization
  |-- confusion_matrix.png               # Confusion matrix
```

## Common Module Reference

The `common/` package provides shared utilities:

| Module | Description |
|--------|-------------|
| `constants.py` | Global constants (sequence length, dimensions, class names) |
| `log.py` | Formatted logging (`log_info`, `log_warn`, `log_err`) |
| `env.py` | Environment detection (local/HPC/Colab) and path management |
| `utils.py` | Helper functions (resampling, interpolation, format conversion) |
| `features.py` | Feature engineering (144-dim feature computation) |
| `augmentation.py` | Data augmentation (jitter, rotation, scale, time warp, speed change) |
| `detection.py` | Hand landmark detection using MediaPipe |
| `model.py` | GestureTCN model definition and parameter counting |
| `dataset.py` | PyTorch Dataset, DataLoader, and class balancing utilities |
| `evaluation.py` | Model evaluation with metrics computation |

## Requirements

- Python 3.8+
- PyTorch
- MediaPipe
- ONNX Runtime
- scikit-learn
- OpenCV
- NumPy
