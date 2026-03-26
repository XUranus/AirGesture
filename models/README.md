# Project Overview: Gesture Motion Classification with TCNN

## What This Project Does

This project builds a **hand gesture recognition system** that can classify five types of gestures from video clips:

| Class | Gesture | Description |
|-------|---------|-------------|
| `grab` | Grab | Hand closing from open palm to fist |
| `release` | Release | Hand opening from fist to open palm |
| `swipe_up` | Swipe Up | Hand swiping upward |
| `swipe_down` | Swipe Down | Hand swiping downward |
| `noise` | Noise | Invalid/unrecognized action |

The system uses a **Temporal Convolutional Neural Network (TCNN/TCN)** -- a type of neural network specifically designed for processing sequential (time-series) data.

---

## The Three-Stage Pipeline

The project is organized into three stages, each implemented as a Jupyter notebook:

```
Raw Gesture Video Clips
        |
        v
+----------------------------+
| Stage 1: Preprocessing     |  (1.preprocess.ipynb)
| - Extract hand landmarks   |
|   from video via MediaPipe  |
| - Data augmentation         |
| - Feature engineering       |
+----------------------------+
        |
        v
  Processed Dataset (.npz)
  + Normalization Stats (.pt)
        |
        v
+----------------------------+
| Stage 2: Training           |  (2.train.ipynb)
| - Build GestureTCN model    |
| - Train with early stopping |
| - Evaluate performance      |
+----------------------------+
        |
        v
  Trained Model Weights (.pth)
        |
        v
+----------------------------+
| Stage 3: Deployment         |  (3.deploy.ipynb)
| - Structured pruning        |
| - ONNX export               |
| - INT8 quantization         |
+----------------------------+
        |
        v
  Lightweight ONNX Model
  (ready for edge devices)
```

---

## Project File Structure

```
models/
  |-- 1.preprocess.ipynb         # Stage 1: Data preprocessing
  |-- 2.train.ipynb              # Stage 2: Model training
  |-- 3.deploy.ipynb             # Stage 3: Pruning, quantization, ONNX export
  |-- hand_landmarker.task       # MediaPipe hand landmark model file
  |
  |-- common/                    # Shared Python modules
  |   |-- __init__.py            # (empty, makes it a Python package)
  |   |-- constants.py           # All shared constants and hyperparameters
  |   |-- gesture_tcn_model.py   # The TCNN model architecture definition
  |   |-- utils.py               # Utility functions (resampling, feature computation)
  |   |-- log.py                 # Simple timestamped logging helpers
  |
  |-- checkpoints/               # Output directory for all artifacts
  |   |-- cache/                 # Cached preprocessed datasets (.npz)
  |   |   |-- train_v3_*.npz     # Training data cache
  |   |   |-- test_v3_*.npz      # Test data cache
  |   |-- norm_stats.pt          # Normalization mean/std statistics
  |   |-- dataset_info.json      # Dataset metadata
  |   |-- gesture_tcn_best.pth   # Best model weights from training
  |   |-- gesture_tcn_structured_pruned.pth  # Pruned model weights
  |   |-- config.json            # Deployment configuration
  |   |-- training_curves.png    # Loss/accuracy plots
  |   |-- confusion_matrix.png   # Confusion matrix visualization
  |   |-- training_history.json  # Full training metrics history
  |
  |-- datasets/                  # (External) Raw video dataset
      |-- organized/
          |-- Train/
          |   |-- grab/          # Training videos for each class
          |   |-- release/
          |   |-- swipe_up/
          |   |-- swipe_down/
          |   |-- noise/
          |-- Test/
              |-- grab/          # Test videos for each class
              |-- ...
```

---

## Key Concepts for Beginners

### What is a TCN (Temporal Convolutional Network)?

A **TCN** is a type of neural network designed to process **sequential data** (data that changes over time). Unlike traditional RNNs/LSTMs that process sequences step-by-step, a TCN uses **1D convolutions** to process the entire sequence at once, which makes it:

- **Faster** to train (can be parallelized)
- **Better at capturing long-range dependencies** (using dilated convolutions)
- **More stable** during training (avoids vanishing gradient problems)

Think of it like sliding a filter across a timeline of hand positions to detect patterns.

### What are Hand Landmarks?

MediaPipe's hand detection identifies **21 key points** (landmarks) on a hand, each with **x, y, z** coordinates. These landmarks include the wrist, each finger joint, and each fingertip:

```
        8   12  16  20       <- Fingertips
        |   |   |   |
    7   11  15  19
    |   |   |   |
    6   10  14  18
    |   |   |   |
    5   9   13  17
     \  |   |  /
  4   \ |   | /
  |    \|   |/
  3     \   /
  |      \ /
  2       0                  <- Wrist (landmark 0)
  |
  1
```

This gives us `21 landmarks x 3 coordinates = 63` raw values per frame.

### What is Feature Engineering?

Raw landmark coordinates are not ideal for classification directly. The project computes **144 features** from the 63 raw values:

| Feature | Dimensions | Purpose |
|---------|-----------|---------|
| Normalized landmarks | 63 | Hand-size invariant positions |
| Velocity | 63 | How fast each point moves between frames |
| Wrist velocity | 3 | Overall hand movement direction |
| Finger pair distances | 10 | How open/closed the hand is |
| Finger curl angles | 5 | Bending angle of each finger |
| **Total** | **144** | |

### What are Pruning and Quantization?

- **Pruning**: Removing unnecessary parts of a neural network to make it smaller and faster, similar to pruning branches off a tree. This project uses **structured pruning**, which removes entire channels/filters.
- **Quantization**: Converting model weights from 32-bit floating point (FP32) to 8-bit integers (INT8). This makes the model ~4x smaller with minimal accuracy loss.

---

## Key Numbers at a Glance

| Metric | Value |
|--------|-------|
| Input sequence length | 30 frames |
| Feature dimensionality | 144 |
| Number of classes | 5 |
| Original model parameters | 87,077 |
| Pruned model parameters | 45,877 (~1.9x compression) |
| Original model size | 0.34 MB |
| Pruned model size | 0.18 MB |
| Best test accuracy | ~89.7% |

---

## How to Read This Guide

1. **[Stage 1: Data Preprocessing](./1_preprocess.md)** -- How raw videos become training data
2. **[Stage 2: Model Training](./2_train.md)** -- How the TCNN model learns to classify gestures
3. **[Stage 3: Deployment](./3_deploy.md)** -- How the trained model is optimized for edge devices
