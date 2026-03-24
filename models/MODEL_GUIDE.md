# Model Training and Deployment Guide

## Overview

This guide covers the complete pipeline for training, optimizing, and deploying the GestureTCN model for real-time hand gesture recognition.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Project Structure](#project-structure)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Model Optimization](#model-optimization)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended for training)
- **CPU**: Intel/AMD x86_64 (for inference)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for dataset and checkpoints

### Software

```bash
# Core dependencies
Python 3.8+
PyTorch 2.0+
NumPy 1.20+
OpenCV 4.5+
MediaPipe 0.10+

# Model optimization
ONNX 1.14+
ONNX Runtime 1.17+
ONNX Script (optional)

# Training utilities
scikit-learn 1.0+
tqdm 4.60+
matplotlib 3.5+
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch numpy opencv-python mediapipe
pip install onnx onnxruntime onnxruntime-tools
pip install scikit-learn tqdm matplotlib
```

---

## Project Structure

```
models/
├── 1.preprocess.ipynb          # Data preprocessing & augmentation
├── 2.train.ipynb               # Model training & evaluation
├── 3.deploy.ipynb              # Model optimization & export
├── train_gesture_recognition.ipynb  # Alternative training notebook
└── common/
    ├── gesture_tcn_model.py    # TCN model definition
    ├── utils.py                # Feature extraction utilities
    ├── constants.py            # Configuration constants
    └── log.py                  # Logging utilities

checkpoints/                    # Output directory
├── cache/
│   ├── train_v3_fixed_swipe_reverse.npz
│   └── test_v3_fixed_swipe_reverse.npz
├── gesture_tcn_best.pth        # Trained model weights
├── gesture_tcn_structured_pruned.pth
├── gesture_tcn_original.onnx
├── gesture_tcn_pruned.onnx
├── gesture_tcn_pruned_quantized.onnx
├── norm_stats.pt              # Normalization statistics
├── dataset_info.json          # Dataset metadata
└── config.json                # Deployment configuration
```

---

## Data Pipeline

### Step 1: Raw Data Collection

Videos are collected using the `utils/recorder/` application:
- 5 gesture classes: grab, release, swipe_up, swipe_down, noise
- Organized in Train/Test directories
- Metadata saved to labels.csv

### Step 2: Landmark Extraction (1.preprocess.ipynb)

#### MediaPipe Hand Tracking

```python
from mediapipe.tasks.python import vision as mpv

# Initialize hand landmarker
detector = mpv.HandLandmarker.create_from_options(options)

# Extract 21 3D landmarks per frame
result = detector.detect(image)
landmarks = result.hand_landmarks[0]  # 21 points × (x, y, z)
```

#### Feature Engineering (144 dimensions)

```python
def compute_features(raw_seq):
    """
    Input:  raw_seq shape (T, 63) - T frames, 21 landmarks × 3 coords
    Output: features shape (T, 144)
    
    Features:
    - Normalized landmarks: 63 dims (wrist-relative, palm-size normalized)
    - Velocity: 63 dims (frame-to-frame landmark movement)
    - Wrist velocity: 3 dims (global hand movement)
    - Finger distances: 10 dims (inter-fingertip distances)
    - Finger angles: 5 dims (finger joint angles)
    """
```

**Normalization Strategy:**
```python
# 1. Wrist-relative coordinates
relative = landmarks - wrist_position

# 2. Palm-size normalization
palm_size = norm(middle_finger - wrist)
normalized = relative / palm_size

# 3. Dataset-wide z-score normalization
mean, std = compute_dataset_statistics()
features = (features - mean) / (std + 1e-8)
```

### Step 3: Data Augmentation

| Augmentation | Parameters | Effect |
|-------------|------------|--------|
| Jitter | σ=0.003 | Robustness to landmark noise |
| Rotation | ±5°, ±10°, ±15° | Viewpoint invariance |
| Scaling | 0.85×, 0.9×, 1.1×, 1.15× | Hand size invariance |
| Time Warping | anchor=0.3-0.7, warp=0.8-1.2 | Speed variation |
| Speed Change | factor=0.8-1.2 | Temporal distortion |
| Mirroring | X-axis flip | Left/right hand invariance |
| Reversal | Time reverse | grab↔release, swipe_up↔swipe_down |

**Augmentation Code:**
```python
# Rotation around wrist
def rotate_2d(raw_seq, angle_deg):
    s = raw_seq.reshape(-1, 21, 3)
    wrist = s[:, 0:1, :2].copy()
    s[:, :, :2] -= wrist  # Center at wrist
    
    # Apply 2D rotation matrix
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = [[c, -s], [s, c]]
    
    s[:, :, :2] = np.dot(s[:, :, :2], rotation_matrix.T)
    s[:, :, :2] += wrist
    return s.reshape(-1, 63)

# Time reversal with label mapping
def reverse_sequence(raw_seq, label):
    reversed_seq = raw_seq[::-1].copy()
    
    # Map labels for reversed motion
    label_map = {
        'grab': 'release',
        'release': 'grab',
        'swipe_up': 'swipe_down',
        'swipe_down': 'swipe_up',
        'noise': 'noise'
    }
    return reversed_seq, label_map[label]
```

### Step 4: Dataset Statistics

After preprocessing:
- **Training samples**: ~3000
- **Test samples**: ~30
- **Sequence length**: 30 frames
- **Feature dimension**: 144

**Output Files:**
- `checkpoints/cache/train_{version}.npz` - Training data
- `checkpoints/cache/test_{version}.npz` - Test data
- `checkpoints/norm_stats.pt` - mean (144,), std (144,)
- `checkpoints/dataset_info.json` - Dataset metadata

---

## Model Architecture

### GestureTCN Structure

```python
class GestureTCN(nn.Module):
    def __init__(self, num_classes=5, feat_dim=144, dropout=0.15, channels=None):
        super().__init__()
        
        # Channel configuration (supports pruning)
        c = channels or {"stem": 48, "mid": 48, "out": 64, "head": 32}
        
        # Stem: Project input to working dimension
        self.stem = nn.Sequential(
            nn.Conv1d(feat_dim, c["stem"], 1, bias=False),
            nn.BatchNorm1d(c["stem"]),
            nn.ReLU(inplace=True),
        )
        
        # TCN Blocks with increasing receptive field
        self.blocks = nn.Sequential(
            ResBlock(c["stem"], ks=3, dilation=1),   # RF: 3
            ResBlock(c["stem"], ks=3, dilation=2),   # RF: 7
            ChannelBlock(c["stem"], c["out"], ks=3, dilation=4),  # RF: 15
            ResBlock(c["out"], ks=3, dilation=1),    # RF: 19
        )
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(c["out"], c["head"]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c["head"], num_classes),
        )
```

### Building Blocks

#### Causal Convolution

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation=1):
        super().__init__()
        self.pad = (ks - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, ks, 
                              padding=self.pad, 
                              dilation=dilation, 
                              bias=False)
    
    def forward(self, x):
        o = self.conv(x)
        # Remove future information
        if self.pad > 0:
            o = o[:, :, : -self.pad]
        return o
```

**Why Causal?**
- Ensures predictions only use past and present information
- Critical for real-time streaming inference
- No lookahead bias

#### Residual Block

```python
class ResBlock(nn.Module):
    def __init__(self, ch, ks=3, dilation=1, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, ks, dilation),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(ch, ch, ks, dilation),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.net(x) + x)  # Residual connection
```

#### Channel Block

```python
class ChannelBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, dilation=1, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, ks, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, ks, dilation),
            nn.BatchNorm1d(out_ch),
        )
        # 1x1 convolution for channel matching
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))
```

### Receptive Field Analysis

```
Block 1 (d=1):  ███                    → 3 frames
Block 2 (d=2):  █ █ █                  → 7 frames
Block 3 (d=4):  █   █   █              → 15 frames
Block 4 (d=1):  ███                    → 19 frames (total)

Effective receptive field: ~19 frames (~0.6 seconds at 30 FPS)
```

### Model Variants

| Configuration | stem | mid | out | head | Params | Size |
|--------------|------|-----|-----|------|--------|------|
| Original | 48 | 48 | 64 | 32 | 87,077 | 0.34 MB |
| Pruned (30%) | 32 | 32 | 48 | 24 | 45,877 | 0.18 MB |

---

## Training Process

### Training Configuration (2.train.ipynb)

```python
# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 300
LR = 2e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 40  # Early stopping

# Class-balanced weights
class_weights = compute_class_weights(train_labels)
# Example: [0.969, 0.867, 1.259, 1.016, 0.889]

# Loss with label smoothing
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE), 
    label_smoothing=0.1
)

# Optimizer
optimizer = optim.AdamW(
    model.parameters(), 
    lr=LR, 
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS, 
    eta_min=1e-5
)
```

### Training Loop

```python
for epoch in range(1, EPOCHS + 1):
    model.train()
    
    # Training pass
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        
        optimizer.step()
    
    # Learning rate update
    scheduler.step()
    
    # Evaluation
    te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    
    # Save best model
    if te_acc > best_acc:
        best_acc = te_acc
        torch.save(model.state_dict(), "gesture_tcn_best.pth")
    
    # Early stopping
    if patience_ctr >= PATIENCE:
        break
```

### Training Best Practices

1. **Class Balancing**: Weighted loss handles imbalanced datasets
2. **Label Smoothing**: Prevents overconfidence, improves generalization
3. **Gradient Clipping**: Stabilizes training, prevents exploding gradients
4. **Cosine Annealing**: Smooth convergence, better final accuracy
5. **Early Stopping**: Prevents overfitting, saves training time

### Expected Training Curve

```
Epoch   1 | TrL:0.72 TrA:0.86 | TeL:0.99 TeA:0.72
Epoch  10 | TrL:0.42 TrA:0.99 | TeL:0.80 TeA:0.86  <- best so far
Epoch  28 | TrL:0.41 TrA:1.00 | TeL:0.81 TeA:0.90  <- best
...
Epoch  68 | Early stop (no improvement for 40 epochs)

Best test accuracy: 0.8966 @ epoch 28
```

### Output Files

- `checkpoints/gesture_tcn_best.pth` - Best model weights
- `checkpoints/training_curves.png` - Loss/accuracy curves
- `checkpoints/confusion_matrix.png` - Confusion matrix visualization
- `checkpoints/training_history.json` - Full training metrics

---

## Model Optimization

### Structured Pruning (3.deploy.ipynb)

**Goal**: Reduce model size by removing entire channels

```python
def create_pruned_model(prune_ratio=0.3):
    def round_to_8(x):
        # Round to multiple of 8 for SIMD efficiency
        return max(8, round(x / 8) * 8)
    
    channels = {
        "stem": round_to_8(int(48 * (1 - prune_ratio))),  # 48 → 32
        "mid": round_to_8(int(48 * (1 - prune_ratio))),   # 48 → 32
        "out": round_to_8(int(64 * (1 - prune_ratio))),   # 64 → 48
        "head": round_to_8(int(32 * (1 - prune_ratio))),  # 32 → 24
    }
    
    return GestureTCN(channels=channels), channels
```

**Fine-tuning Process:**
```python
# Create pruned model
pruned_model, _ = create_pruned_model(prune_ratio=0.3)

# Fine-tune for 100 epochs
pruned_model, best_acc = fine_tune_model(
    pruned_model, 
    train_loader, 
    test_loader,
    epochs=100,
    lr=1e-3
)

# Result: 45,877 params (1.90x reduction), 92.59% accuracy
```

### INT8 Quantization

**Method**: ONNX Runtime static quantization with calibration

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

# 1. Export to ONNX
torch.onnx.export(
    pruned_model.cpu(),
    dummy_input,
    "gesture_tcn_pruned.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}},
    opset_version=18,
)

# 2. Create calibration data reader
class GestureCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(data_loader)
    
    def get_next(self):
        try:
            x, _ = next(self.iter)
            return {"input": x.numpy()}
        except StopIteration:
            return None
    
    def rewind(self):
        self.iter = iter(self.data_loader)

# 3. Apply static quantization
calib_reader = GestureCalibrationDataReader(test_loader)

quantize_static(
    model_input="gesture_tcn_pruned.onnx",
    model_output="gesture_tcn_pruned_quantized.onnx",
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format
    per_channel=False,
    weight_type=QuantType.QInt8,
)
```

### Optimization Results

| Metric | Original | Pruned | Pruned+INT8 |
|--------|----------|--------|-------------|
| Parameters | 87,077 | 45,877 | 45,877 |
| Size (MB) | 0.34 | 0.18 | 0.17 |
| Accuracy | 88.89% | 92.59% | 92.59% |
| F1-Score | 0.8877 | 0.9290 | 0.9290 |
| Latency (ms) | 0.92 | 0.79 | 1.23 |
| Throughput | 1087/s | 1271/s | 816/s |

**Key Insights:**
- Pruning improved accuracy by +3.7% (regularization effect)
- INT8 quantization maintains accuracy with minimal overhead
- Pruned model is 1.17× faster than original

---

## Deployment

### Desktop Application

**Setup:**
```bash
# Copy model to desktop assets
cp checkpoints/gesture_tcn_pruned_quantized.onnx desktop/assets/
cp checkpoints/config.json desktop/assets/

# Run desktop application
cd desktop
python main.py
```

**Configuration (desktop/config.py):**
```python
DETECTION_METHOD = "neural_network"  # or "legacy"
TCN_MODEL_PATH = "assets/gesture_tcn_pruned_quantized.onnx"
TCN_CONFIG_PATH = "assets/config.json"
TCN_CONFIDENCE_THRESHOLD = 0.5
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession(
    "gesture_tcn_pruned_quantized.onnx",
    providers=["CPUExecutionProvider"]
)

# Prepare input (preprocessed features)
# features shape: (30, 144) -> transpose to (144, 30) -> add batch dim
input_tensor = features.T[np.newaxis, ...].astype(np.float32)

# Run inference
outputs = session.run(None, {"input": input_tensor})
logits = outputs[0]

# Get prediction
gesture_id = np.argmax(logits, axis=1)[0]
confidence = np.max(np.softmax(logits, axis=1))
```

### Mobile Deployment (Android)

```kotlin
// Load ONNX model
val session = Session(
    context, 
    "gesture_tcn_pruned_quantized.onnx",
    SessionOptions()
)

// Prepare input
val inputTensor = Tensor.create(
    floatBuffer, 
    longArrayOf(1, 144, 30)
)

// Run inference
val outputs = session.run(listOf(inputTensor))
val logits = outputs[0].floatBuffer
```

### Deployment Checklist

- [ ] Model exported to ONNX format
- [ ] ONNX model verified with test data
- [ ] INT8 quantization applied (optional)
- [ ] Config file includes normalization stats
- [ ] Config file includes pruned channel info
- [ ] Inference code matches training preprocessing
- [ ] Latency meets requirements (<10ms for real-time)

---

## Troubleshooting

### Common Issues

#### 1. Low Training Accuracy

**Symptoms**: Training accuracy <80% after 50 epochs

**Solutions**:
- Check data quality (landmark extraction errors)
- Verify normalization statistics
- Increase model capacity (more channels)
- Reduce dropout rate
- Check class balance

#### 2. Overfitting

**Symptoms**: Train acc >> Test acc

**Solutions**:
- Increase dropout (0.15 → 0.25)
- Add weight decay (1e-3 → 5e-3)
- More aggressive data augmentation
- Early stopping with smaller patience

#### 3. ONNX Export Fails

**Error**: `No module named 'onnxscript'`

**Solution**:
```bash
pip install onnxscript
# Or use opset_version=14 instead of 18
```

**Error**: Shape inference failed

**Solution**:
```python
import onnx
model = onnx.load("model.onnx")
del model.graph.value_info[:]  # Clear cached shapes
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, "model_fixed.onnx")
```

#### 4. Quantization Accuracy Drop

**Symptoms**: >2% accuracy drop after INT8 quantization

**Solutions**:
- Use more calibration data (500+ samples)
- Try per-channel quantization
- Exclude sensitive layers from quantization
- Use QDQ format instead of QOperator

#### 5. Slow Inference

**Symptoms**: Latency >10ms

**Solutions**:
- Use pruned model
- Enable ONNX Runtime optimizations
- Use OpenVINO execution provider (Intel CPU)
- Reduce sequence length (30 → 20 frames)

```python
# ONNX Runtime optimization
session_options = ort.SessionOptions()
session_options.graph_optimization_level = \
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)
```

---

## Performance Benchmarks

### Inference Latency by Platform

| Platform | Hardware | Latency | Throughput |
|----------|----------|---------|------------|
| Desktop (CPU) | Intel i7-8550U | 0.79 ms | 1271/s |
| Desktop (INT8) | Intel i7-8550U | 1.23 ms | 816/s |
| Android | Snapdragon 865 | 2.1 ms | 476/s |
| Raspberry Pi 4 | Cortex-A72 | 15.3 ms | 65/s |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Model weights | 0.18 MB |
| Activation buffers | 0.5 MB |
| Input/output tensors | 0.1 MB |
| **Total** | **~0.8 MB** |

---

## References

- [Temporal Convolutional Networks](https://arxiv.org/abs/1803.01271)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

---

## License

Copyright (C) 2026 XUranus. All rights reserved.
