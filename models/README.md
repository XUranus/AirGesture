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

## Files

| File | Description |
|------|-------------|
| `preprocess.ipynb` | Data preprocessing: landmark extraction, augmentation, feature engineering |
| `train.ipynb` | Model training: GestureTCN definition, training loop, evaluation |
| `deploy.ipynb` | Model optimization: pruning, ONNX export, INT8 quantization |
| `train_gesture_recognition.ipynb` | **Note**: This notebook contains data for PPT demonstration purposes only |

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
| Best test accuracy | 96.55% |
| F1 Score (Macro) | 0.9636 |
| F1 Score (Weighted) | 0.9655 |

## Model Optimization

| Model | Parameters | Size | Accuracy |
|-------|------------|------|----------|
| Original (FP32) | 87,077 | 0.34 MB | 96.55% |
| Pruned (FP32) | 45,877 | 0.18 MB | 89.66% |
| Pruned + Quantized (INT8 ONNX) | 45,877 | ~0.14 MB | 89.66% |

**Compression ratio**: 1.90x parameter reduction

## Output Files

After running the pipeline, the following files are generated in `checkpoints/`:

```
checkpoints/
  |-- cache/                        # Cached preprocessed datasets
  |   |-- train_v3_*.npz           # Training data
  |   |-- test_v3_*.npz            # Test data
  |-- norm_stats.pt                 # Normalization statistics
  |-- dataset_info.json             # Dataset metadata
  |-- gesture_tcn_best.pth          # Best model weights
  |-- gesture_tcn_structured_pruned.pth  # Pruned model weights
  |-- gesture_tcn_pruned_quantized.onnx  # INT8 quantized ONNX model
  |-- config.json                   # Deployment configuration
  |-- training_curves.png           # Training visualization
  |-- confusion_matrix.png          # Confusion matrix
```

## Requirements

- Python 3.8+
- PyTorch
- MediaPipe
- ONNX Runtime
- scikit-learn
- OpenCV
- NumPy
