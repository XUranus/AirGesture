# Quick Start Guide - Gesture Recognition System

## Learning Path

This guide helps you understand the complete gesture recognition pipeline from data collection to deployment.

---

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Complete Gesture Recognition Pipeline             │
└─────────────────────────────────────────────────────────────────────┘

Phase 1: DATA COLLECTION          Phase 2: MODEL TRAINING
┌──────────────────────┐         ┌──────────────────────┐
│ utils/recorder/      │         │ 1.preprocess.ipynb   │
│ - Record videos      │         │ - Extract landmarks  │
│ - Organize by class  │────────▶│ - Compute features   │
│ - Save metadata      │         │ - Augment data       │
└──────────────────────┘         └──────────────────────┘
                                          │
                                          ▼
┌──────────────────────┐         ┌──────────────────────┐
│ desktop/             │◀────────│ 2.train.ipynb        │
│ - Real-time detect   │         │ - Define TCN model   │
│ - Classify gestures  │         │ - Train & evaluate   │
│ - Trigger actions    │         │ - Save best weights  │
└──────────────────────┘         └──────────────────────┘
                                          │
                                          ▼
                                 ┌──────────────────────┐
                                 │ 3.deploy.ipynb       │
                                 │ - Prune model        │
                                 │ - Quantize to INT8   │
                                 │ - Export to ONNX     │
                                 └──────────────────────┘
```

---

## Step-by-Step Guide

### Step 1: Understand the Problem (15 min)

**What are we building?**
- A system that recognizes 5 hand gestures from webcam video
- Gestures: grab (fist close), release (fist open), swipe_up, swipe_down, noise
- Real-time processing: ~30 FPS with <10ms latency

**Read these files first:**
1. `README.md` - Project overview
2. `desktop/config.py` - Configuration parameters
3. `models/common/constants.py` - Model constants

---

### Step 2: Explore the Data Pipeline (30 min)

**Open: `models/1.preprocess.ipynb`**

**Key concepts:**
- **Raw landmarks**: 21 hand keypoints × 3 coordinates = 63 values per frame
- **Sequence**: 30 consecutive frames = one sample
- **Features**: 144-dimensional vector per frame (normalized landmarks + velocity + angles)

**Code to try:**
```python
# Load a sample video
video_path = "dataset/Train/grab/grab_001.avi"

# Extract landmarks
detector = HandDetector()
landmarks = detector.detect(frame)  # Shape: (63,)

# Compute features
features = compute_features(landmarks)  # Shape: (30, 144)
```

**Exercise:**
1. Run the preview cells to see video frames
2. Visualize hand landmarks on a frame
3. Plot feature values over time

---

### Step 3: Understand the Model (45 min)

**Open: `models/common/gesture_tcn_model.py`**

**Architecture summary:**
```
Input (144, 30)
   │
   ▼
Conv1d(144→48) + BN + ReLU
   │
   ▼
ResBlock(d=1) → ResBlock(d=2) → ChannelBlock(d=4) → ResBlock(d=1)
   │
   ▼
Global Average Pool → (64,)
   │
   ▼
Linear(64→32) → ReLU → Dropout → Linear(32→5)
   │
   ▼
Output: 5 class probabilities
```

**Key concepts:**
- **Causal convolution**: Only uses past information (critical for real-time)
- **Dilation**: Expands receptive field without extra parameters
- **Residual connection**: Helps gradient flow, enables deeper networks

**Code to try:**
```python
from common.gesture_tcn_model import GestureTCN
import torch

# Create model
model = GestureTCN()

# Check parameters
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass
x = torch.randn(1, 144, 30)  # (batch, features, time)
output = model(x)
print(f"Output shape: {output.shape}")  # (1, 5)
```

---

### Step 4: Train the Model (1 hour)

**Open: `models/2.train.ipynb`**

**Training loop breakdown:**
```python
for epoch in range(EPOCHS):
    # 1. Forward pass
    logits = model(batch_x)
    
    # 2. Compute loss
    loss = criterion(logits, batch_y)
    
    # 3. Backward pass
    loss.backward()
    optimizer.step()
    
    # 4. Evaluate on test set
    test_acc = evaluate(model, test_loader)
    
    # 5. Save best model
    if test_acc > best_acc:
        torch.save(model.state_dict(), "best.pth")
```

**Key concepts:**
- **CrossEntropyLoss with class weights**: Handles imbalanced data
- **AdamW optimizer**: Adam with decoupled weight decay
- **Cosine annealing**: Smooth learning rate decay
- **Early stopping**: Prevents overfitting

**Exercise:**
1. Run the full training notebook
2. Plot training curves
3. Analyze confusion matrix

---

### Step 5: Optimize the Model (45 min)

**Open: `models/3.deploy.ipynb`**

**Optimization techniques:**

1. **Structured Pruning** (remove 30% channels):
```python
# Before: 87K parameters
channels = {"stem": 48, "mid": 48, "out": 64, "head": 32}

# After: 46K parameters
channels = {"stem": 32, "mid": 32, "out": 48, "head": 24}
```

2. **INT8 Quantization** (reduce precision):
```python
# FP32 (32-bit floating point) → INT8 (8-bit integer)
# 4x reduction in memory bandwidth
```

**Exercise:**
1. Run pruning and fine-tuning
2. Compare pruned vs original accuracy
3. Export to ONNX format

---

### Step 6: Deploy to Desktop (30 min)

**Open: `desktop/gesture_detector.py`**

**Detection pipeline:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │ →  │  MediaPipe  │ →  │    TCN      │
│   Capture   │    │  Landmarks  │    │  Classifier │
└─────────────┘    └─────────────┘    └─────────────┘
                          │                    │
                          ▼                    ▼
                    ┌─────────────┐    ┌─────────────┐
                    │   Legacy    │    │   Gesture   │
                    │  (fallback) │    │   Event     │
                    └─────────────┘    └─────────────┘
```

**Two-stage detection:**
1. **IDLE**: Scan at 10 FPS for hand presence
2. **WAKEUP**: Process at 30 FPS, classify gesture

**Code to try:**
```python
from desktop.gesture_detector import GestureDetector

detector = GestureDetector()
detector.on_gesture = lambda e: print(f"Gesture: {e}")
detector.start()
# ... running ...
detector.stop()
```

**Exercise:**
1. Copy model to `desktop/assets/`
2. Run the desktop application
3. Test each gesture

---

## Code Flow Examples

### Example 1: From Video to Prediction

```python
# 1. Capture video frame
frame = camera.read()  # Shape: (480, 640, 3)

# 2. Extract hand landmarks
detector = HandLandmarkDetector()
detail = detector.detect(frame)
raw_landmarks = detail.raw_landmarks  # Shape: (63,)

# 3. Build sequence (store 30 frames)
sequence.append(raw_landmarks)
if len(sequence) == 30:
    # 4. Compute features
    features = compute_features(sequence)  # Shape: (30, 144)
    
    # 5. Normalize
    features = (features - mean) / (std + 1e-8)
    
    # 6. Prepare input
    x = torch.FloatTensor(features.T).unsqueeze(0)  # Shape: (1, 144, 30)
    
    # 7. Run inference
    model = GestureTCN()
    model.load_state_dict(torch.load("best.pth"))
    logits = model(x)  # Shape: (1, 5)
    
    # 8. Get prediction
    gesture_id = logits.argmax().item()
    confidence = logits.softmax(dim=1).max().item()
    gesture_name = CLASS_NAMES[gesture_id]
    
    print(f"Detected: {gesture_name} ({confidence:.2%})")
```

### Example 2: Training Iteration

```python
# 1. Load batch
batch_x, batch_y = next(iter(train_loader))
# batch_x shape: (32, 144, 30)
# batch_y shape: (32,)

# 2. Forward pass
model.train()
logits = model(batch_x)  # Shape: (32, 5)

# 3. Compute loss
loss = criterion(logits, batch_y)

# 4. Backward pass
optimizer.zero_grad()
loss.backward()

# 5. Clip gradients (prevent exploding)
torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

# 6. Update weights
optimizer.step()
```

---

## Common Questions

### Q: Why 30 frames?
**A**: At 30 FPS, 30 frames = 1 second of motion. This captures the complete gesture duration while keeping latency low.

### Q: Why 144 features?
**A**: Feature breakdown:
- 63: Normalized landmarks (position)
- 63: Velocity (motion)
- 3: Wrist velocity (global motion)
- 10: Finger distances (hand shape)
- 5: Finger angles (hand pose)
Total: 144 dimensions

### Q: How does pruning improve accuracy?
**A**: Pruning acts as regularization, similar to dropout. By removing redundant channels, the model generalizes better.

### Q: Why INT8 quantization?
**A**: INT8 reduces:
- Memory usage: 4x less
- Memory bandwidth: 4x less
- Power consumption: 2-3x less
Trade-off: ~5-10% latency increase on CPU (but faster on NPUs)

### Q: Can I use a different backbone?
**A**: Yes! Popular alternatives:
- LSTM/GRU: Good for sequences, but slower
- Transformer: Better accuracy, but needs more data
- MobileNet: Optimized for mobile, but 2D (needs adaptation)

---

## Debugging Tips

### Check Data Quality
```python
# Verify landmark extraction
for video in dataset:
    landmarks = extract_landmarks(video)
    assert landmarks.shape == (num_frames, 63)
    assert not np.isnan(landmarks).any()
```

### Check Model Input
```python
# Verify preprocessing
features = compute_features(raw_landmarks)
assert features.shape == (30, 144)
assert not np.isnan(features).any()

# After normalization
features_norm = (features - mean) / std
assert features_norm.mean() ≈ 0
assert features_norm.std() ≈ 1
```

### Check Inference
```python
# Verify output
model.eval()
with torch.no_grad():
    logits = model(x)
    probs = logits.softmax(dim=1)
    
assert logits.shape == (batch_size, 5)
assert probs.sum(dim=1) ≈ 1.0
assert not torch.isnan(logits).any()
```

---

## Next Steps

1. **Experiment**: Try different hyperparameters
2. **Visualize**: Plot attention maps, feature distributions
3. **Optimize**: Try knowledge distillation, neural architecture search
4. **Deploy**: Build mobile app, web demo
5. **Extend**: Add more gestures, multi-hand tracking

---

## Resources

- **PyTorch Docs**: https://pytorch.org/docs/
- **MediaPipe**: https://google.github.io/mediapipe/
- **ONNX Runtime**: https://onnxruntime.ai/
- **Papers**:
  - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (TCN)
  - "MediaPipe Hands: On-device Real-time Hand Tracking"

---

## Getting Help

1. Check existing issues on GitHub
2. Review training logs for errors
3. Verify environment (Python version, dependencies)
4. Test with smaller dataset first
5. Use debug mode (print intermediate shapes/values)

Good luck! 🚀
