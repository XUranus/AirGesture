# Stage 2: Model Training (2.train.ipynb)

This document walks you through the model training notebook, explaining the TCNN architecture, the training loop, and the evaluation process.

---

## Table of Contents

1. [Big Picture: What This Stage Does](#1-big-picture)
2. [Sections 1-3: Loading Preprocessed Data](#2-loading-preprocessed-data)
3. [Section 4: PyTorch Dataset and DataLoader](#3-pytorch-dataset-and-dataloader)
4. [Section 5: The GestureTCN Model Architecture](#4-the-gesturetcn-model-architecture)
5. [Section 6: Training Loop](#5-training-loop)
6. [Section 7: Training Curves](#6-training-curves)
7. [Section 8: Evaluation](#7-evaluation)
8. [Section 9: Saving Results](#8-saving-results)

---

## 1. Big Picture

This stage takes the preprocessed data from Stage 1 and trains a Temporal Convolutional Network to classify gestures:

```
Preprocessed Data                         Trained Model
+-------------------+                    +-------------------+
| train_*.npz       |                    | gesture_tcn_      |
| (3018 samples)    |  --> GestureTCN    |   best.pth        |
| test_*.npz        |     Training       | training_curves   |
| (29 samples)      |     Loop           | confusion_matrix  |
| norm_stats.pt     |                    | training_history  |
+-------------------+                    +-------------------+
```

---

## 2. Loading Preprocessed Data

### Sections 1-3: Environment Setup and Data Loading

```python
# Load dataset metadata from Stage 1
with open("checkpoints/dataset_info.json") as f:
    dataset_info = json.load(f)

# Training hyperparameters
BATCH_SIZE = 32    # Number of samples processed together
EPOCHS = 300       # Maximum training iterations over full dataset
LR = 2e-3          # Learning rate (0.002)
WEIGHT_DECAY = 1e-3  # L2 regularization strength
PATIENCE = 40      # Early stopping patience
```

Let's understand each hyperparameter:

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `BATCH_SIZE` | 32 | How many samples the model sees before updating its weights. Larger = more stable but slower per update |
| `EPOCHS` | 300 | Maximum number of times to iterate through the entire dataset |
| `LR` (Learning Rate) | 0.002 | How big each weight update step is. Too high = unstable; too low = slow learning |
| `WEIGHT_DECAY` | 0.001 | Penalizes large weights to prevent overfitting (L2 regularization) |
| `PATIENCE` | 40 | Stop training if test accuracy hasn't improved for 40 consecutive epochs |

The data is loaded from the cached `.npz` files produced by Stage 1, along with the normalization statistics (`norm_stats.pt`).

---

## 3. PyTorch Dataset and DataLoader

### Section 4: `GestureDataset` class

PyTorch requires data to be wrapped in a `Dataset` object:

```python
class GestureDataset(Dataset):
    def __init__(self, raw_samples, labels, norm_stats=None, augment=False):
        self.raw_samples = raw_samples   # List of (30, 63) arrays
        self.labels = labels             # List of class indices (0-4)
        self.norm_stats = norm_stats     # Mean and std for normalization
        self.augment = augment           # Apply online augmentation?

    def __getitem__(self, idx):
        raw = self.raw_samples[idx].copy()   # Get one sample
        
        # Online augmentation (training only)
        if self.augment and random.random() < 0.5:
            raw = add_jitter(raw, sigma=0.002)   # 50% chance of jitter
        if self.augment and random.random() < 0.3:
            raw = time_warp(raw)                  # 30% chance of time warp
            raw = resample(raw, SEQ_LEN)
        
        # Convert raw landmarks (63 dims) -> features (144 dims)
        feat = compute_features(raw)
        
        # Normalize: (x - mean) / std
        if self.norm_stats is not None:
            feat = (feat - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-8)
        
        # Return as PyTorch tensors
        x = torch.FloatTensor(feat.T)  # Shape: (144, 30) -- channels first!
        y = torch.tensor(label, dtype=torch.long)
        return x, y
```

**Important detail**: Note `feat.T` -- the feature matrix is **transposed** from `(30, 144)` to `(144, 30)`. This is because PyTorch's `Conv1d` expects the input shape to be `(batch, channels, length)`, where:
- `channels = 144` (feature dimensions)
- `length = 30` (time steps)

### Online vs. Offline Augmentation

Stage 1 already applied offline augmentation (rotation, scaling, etc.) to create multiple copies per video. Here, the `GestureDataset` applies **additional online augmentation** during training:
- **Jitter** (50% chance): Adds small noise each time a sample is accessed
- **Time warp** (30% chance): Randomly distorts timing

Why both? Offline augmentation creates fixed variants. Online augmentation means the model sees slightly different data **every epoch**, further improving generalization.

### Class Balancing

The dataset has unequal class sizes (e.g., 612 grab vs 471 swipe_up). Two mechanisms address this:

#### 1. Class Weights for Loss Function

```python
def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    w = total / (NUM_CLASSES * counts)  # Inverse frequency weighting
    w = w / w.sum() * NUM_CLASSES       # Normalize
    return torch.FloatTensor(w)
```

Classes with fewer samples get **higher weights** in the loss function, so the model pays more attention to underrepresented classes.

Result: `[0.969, 0.867, 1.259, 1.016, 0.889]` -- swipe_up (index 2) has highest weight because it has fewest samples.

#### 2. Weighted Random Sampler

```python
def make_sampler(labels):
    counts = np.bincount(labels)
    sw = [1.0 / counts[l] for l in labels]  # Sample weight = 1/class_count
    return WeightedRandomSampler(sw, len(sw), replacement=True)
```

Instead of iterating through samples in order, the `WeightedRandomSampler` ensures each class is sampled approximately equally in each epoch.

---

## 4. The GestureTCN Model Architecture

### Section 5 and `common/gesture_tcn_model.py`

This is the heart of the project. Let's build up understanding layer by layer.

### 4.1 What is a 1D Convolution?

A regular 2D convolution (used in image processing) slides a filter across height and width. A **1D convolution** slides a filter across just one dimension -- in our case, **time**.

```
Input (144 channels, 30 time steps):

Channel 0:  [v0, v1, v2, v3, v4, ..., v29]
Channel 1:  [v0, v1, v2, v3, v4, ..., v29]
...
Channel 143: [v0, v1, v2, v3, v4, ..., v29]

Conv1d with kernel_size=3 looks at 3 consecutive time steps:
  [v0, v1, v2] -> output[0]
  [v1, v2, v3] -> output[1]
  [v2, v3, v4] -> output[2]
  ...
```

Each filter learns to detect a specific temporal pattern (e.g., "landmarks moving inward rapidly").

### 4.2 Causal Convolution

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation=1):
        self.pad = (ks - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, ks, 
                              padding=self.pad, dilation=dilation, bias=False)
    
    def forward(self, x):
        o = self.conv(x)
        if self.pad > 0:
            o = o[:, :, :-self.pad]  # Remove right padding
        return o
```

A **causal convolution** ensures that the output at time `t` only depends on inputs at times `<= t` (never future frames). This is achieved by:
1. Adding padding only on the **left** side
2. After the convolution, **trimming** the extra right-side output

**Why causal?** In deployment, gestures are recognized in real-time. The model should make predictions based only on frames it has seen so far, not future frames.

### 4.3 Dilated Convolution

Normal convolution with kernel_size=3 looks at 3 **adjacent** time steps. **Dilated convolution** skips steps:

```
Dilation = 1 (normal):  looks at t, t+1, t+2     (3 adjacent steps)
Dilation = 2:           looks at t, t+2, t+4     (skips every other step)
Dilation = 4:           looks at t, t+4, t+8     (skips 3 steps)
```

This allows the network to capture patterns at different time scales without increasing the number of parameters:
- Dilation 1: Detects short-range patterns (quick finger movements)
- Dilation 2: Detects medium-range patterns (finger closing sequence)
- Dilation 4: Detects long-range patterns (entire gesture trajectory)

### 4.4 Residual Block (ResBlock)

```python
class ResBlock(nn.Module):
    def __init__(self, ch, ks=3, dilation=1, dropout=0.15):
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, ks, dilation),  # Causal conv
            nn.BatchNorm1d(ch),                   # Normalize activations
            nn.ReLU(inplace=True),                # Non-linearity
            nn.Dropout(dropout),                   # Regularization
            CausalConv1d(ch, ch, ks, dilation),  # Second causal conv
            nn.BatchNorm1d(ch),                   # Normalize
        )
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.net(x) + x)  # <-- THE RESIDUAL CONNECTION
```

The key innovation is `self.net(x) + x` -- the **residual connection** (or skip connection). Instead of learning the full transformation, the block only needs to learn the **difference** (residual) from the input. This makes deep networks much easier to train because:

1. Gradients flow directly through the `+ x` path during backpropagation
2. If the block can't learn anything useful, it can default to identity (`net(x) = 0`)
3. Each block only needs to learn a small refinement

The components inside each ResBlock:
- **BatchNorm1d**: Normalizes activations to have zero mean and unit variance, stabilizing training
- **ReLU**: `max(0, x)` -- introduces non-linearity so the network can learn complex patterns
- **Dropout (15%)**: Randomly zeroes 15% of values during training to prevent overfitting

### 4.5 Channel Block (ChannelBlock)

```python
class ChannelBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ...):
        self.net = nn.Sequential(...)       # Same as ResBlock but changes channels
        self.skip = nn.Sequential(          # 1x1 conv to match dimensions
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
    
    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))  # Skip has 1x1 conv
```

Like a ResBlock, but the input and output have **different channel counts**. The `skip` connection uses a `1x1 convolution` to project the input to match the new channel dimension.

### 4.6 Full Model Architecture

```python
class GestureTCN(nn.Module):
    # Default channels: stem=48, out=64, head=32
    
    def __init__(self, num_classes=5, feat_dim=144, dropout=0.15, channels=None):
        # STEM: Reduce 144 features to 48 channels
        self.stem = nn.Sequential(
            nn.Conv1d(144, 48, kernel_size=1),   # 1x1 conv (pointwise)
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
        )
        
        # BLOCKS: 4 temporal processing blocks
        self.blocks = nn.Sequential(
            ResBlock(48, ks=3, dilation=1),      # Local patterns
            ResBlock(48, ks=3, dilation=2),      # Medium-range patterns
            ChannelBlock(48, 64, ks=3, dilation=4),  # Long-range + expand channels
            ResBlock(64, ks=3, dilation=1),      # Refine with more channels
        )
        
        # POOL: Average across all time steps
        self.pool = nn.AdaptiveAvgPool1d(1)      # (batch, 64, 30) -> (batch, 64, 1)
        
        # HEAD: Classify
        self.head = nn.Sequential(
            nn.Linear(64, 32),                   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(32, 5),                    # 32 -> 5 classes
        )
```

**Data flow through the model:**

```
Input: (batch, 144, 30)        # 144 features, 30 time steps
         |
   [Stem: 1x1 Conv]
         |
       (batch, 48, 30)         # Compressed to 48 channels
         |
   [ResBlock dilation=1]       # Detect local patterns
         |
       (batch, 48, 30)
         |
   [ResBlock dilation=2]       # Detect medium patterns
         |
       (batch, 48, 30)
         |
   [ChannelBlock dilation=4]   # Detect long patterns, expand to 64 ch
         |
       (batch, 64, 30)
         |
   [ResBlock dilation=1]       # Final refinement
         |
       (batch, 64, 30)
         |
   [AdaptiveAvgPool1d]         # Average over time
         |
       (batch, 64)             # Single vector per sample
         |
   [Linear 64->32, ReLU]
   [Linear 32->5]
         |
       (batch, 5)              # Logits for 5 classes
```

**Total parameters: 87,077** -- this is a very lightweight model, suitable for edge deployment.

### 4.7 Weight Initialization

```python
def _init(self):
    for m in self.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)    # gamma = 1
            nn.init.zeros_(m.bias)     # beta = 0
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
```

**Kaiming initialization** sets initial weights to random values with a variance that accounts for the ReLU activation function. This prevents the common problem of signals "exploding" or "vanishing" as they pass through many layers at the start of training.

---

## 5. Training Loop

### Section 6: The Core Training Logic

```python
for epoch in range(1, EPOCHS + 1):
    # === TRAINING PHASE ===
    model.train()                          # Enable dropout and batch norm training mode
    for bx, by in train_loader:            # Iterate over mini-batches
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        
        optimizer.zero_grad()              # Clear previous gradients
        logits = model(bx)                 # Forward pass
        loss = criterion(logits, by)       # Compute loss
        loss.backward()                    # Backpropagation
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # Gradient clipping
        optimizer.step()                   # Update weights
    
    scheduler.step()                       # Update learning rate
    
    # === EVALUATION PHASE ===
    te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    
    # === EARLY STOPPING ===
    if te_acc > best_acc:
        best_acc = te_acc
        patience_ctr = 0
        torch.save(model.state_dict(), ckpt_path)  # Save best model
    else:
        patience_ctr += 1
    
    if patience_ctr >= PATIENCE:   # No improvement for 40 epochs
        break                       # Stop training
```

Let's understand each component:

### 5.1 Forward Pass

```python
logits = model(bx)  # Shape: (32, 5) -- 32 samples, 5 class scores
```

The model processes a batch of 32 samples and outputs **logits** (raw scores) for each class. Higher logits = higher confidence for that class.

### 5.2 Loss Function: Cross-Entropy with Label Smoothing

```python
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

**Cross-entropy loss** measures how different the model's predictions are from the true labels:
- If the model predicts the correct class with high confidence -> low loss
- If the model predicts the wrong class -> high loss

**Label smoothing (0.1)**: Instead of training with "hard" labels like `[0, 0, 1, 0, 0]` (100% confidence for one class), label smoothing uses "soft" labels like `[0.02, 0.02, 0.92, 0.02, 0.02]`. This prevents the model from becoming overconfident and improves generalization.

**Class weights**: The loss for underrepresented classes is amplified (e.g., swipe_up gets weight 1.259 vs release's 0.867).

### 5.3 Backpropagation and Optimization

```python
loss.backward()                    # Compute gradients for all parameters
torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # Clip large gradients
optimizer.step()                   # Update parameters
```

**Backpropagation**: PyTorch automatically computes how each weight should change to reduce the loss, using the chain rule of calculus.

**Gradient clipping**: Limits the magnitude of gradients to 2.0. This prevents "gradient explosion" -- a situation where gradients become extremely large and cause the model weights to change too drastically, destabilizing training.

**AdamW optimizer**: An improved version of the Adam optimizer that handles weight decay correctly:
- **Adam** maintains per-parameter learning rates that adapt based on gradient history
- **W (weight decay)** adds a penalty proportional to weight magnitude, encouraging smaller weights

### 5.4 Learning Rate Schedule

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
```

**Cosine annealing** gradually reduces the learning rate following a cosine curve:

```
LR
0.002 |*
      |  *
      |    *
      |      **
      |         ***
      |            ****
0.00001|                ****...
      +----+----+----+-----> Epoch
      0   75  150  225  300
```

**Why decrease the learning rate?** Early in training, large learning rates help the model explore and learn quickly. Later, smaller learning rates help the model fine-tune its predictions without overshooting optimal values.

### 5.5 Early Stopping

```python
if patience_ctr >= PATIENCE:  # PATIENCE = 40
    break
```

Training stops if test accuracy hasn't improved for 40 consecutive epochs. This prevents:
- **Overfitting**: The model memorizes training data instead of learning generalizable patterns
- **Wasted computation**: No point continuing if the model isn't improving

In practice, the model trained for 49 epochs before early stopping triggered, with the best accuracy achieved at epoch 9.

### 5.6 Evaluation Function

```python
@torch.no_grad()  # Disable gradient computation (saves memory and time)
def evaluate(model, loader, criterion, device):
    model.eval()   # Disable dropout and use running batch norm statistics
    for bx, by in loader:
        logits = model(bx)
        p = logits.argmax(1)           # Predicted class = highest logit
        correct += (p == by).sum()     # Count correct predictions
    return total_loss / total, correct / total, preds_all, labels_all
```

`model.eval()` is critical -- it changes the behavior of:
- **Dropout**: Disabled (use all neurons, not random 85%)
- **BatchNorm**: Uses running statistics accumulated during training, not batch statistics

---

## 6. Training Curves

### Section 7: Visualization

The notebook plots three graphs:

1. **Loss Curve**: Shows training loss and test loss over epochs
   - Both should decrease initially
   - If test loss increases while training loss continues to decrease, that indicates **overfitting**

2. **Accuracy Curve**: Shows training and test accuracy over epochs
   - Training accuracy reaches ~100% quickly (the model memorizes training data)
   - Test accuracy is the true measure of generalization

3. **Learning Rate Schedule**: Shows the cosine annealing curve

---

## 7. Evaluation

### Section 8: Detailed Performance Analysis

After training, the best model is loaded and evaluated:

#### Classification Report

```
              precision    recall  f1-score   support
        grab     1.0000    0.6667    0.8000         6
     release     1.0000    1.0000    1.0000         6
    swipe_up     0.7143    1.0000    0.8333         5
  swipe_down     1.0000    1.0000    1.0000         5
       noise     0.8571    0.8571    0.8571         7
```

**Understanding the metrics:**
- **Precision**: Of all samples the model predicted as class X, what fraction were actually class X? (e.g., "when the model says grab, is it right?")
- **Recall**: Of all actual class X samples, what fraction did the model correctly identify? (e.g., "does the model catch all grabs?")
- **F1-score**: Harmonic mean of precision and recall -- a balanced single metric
- **Support**: Number of test samples in each class

#### Confusion Matrix

Shows which classes get confused with each other:

```
            grab  release  swipe_up  swipe_down  noise
    grab       4        0         1           0      1
 release       0        6         0           0      0
swipe_up       0        0         5           0      0
swipe_down     0        0         0           5      0
   noise       0        0         1           0      6
```

Reading this: Row = true class, Column = predicted class. The diagonal shows correct predictions. Off-diagonal entries are errors. For example, 1 grab sample was predicted as swipe_up, and 1 grab was predicted as noise.

---

## 8. Saving Results

### Section 9: Output Files

```python
# Best model weights
torch.save(model.state_dict(), "checkpoints/gesture_tcn_best.pth")

# Training history (for reproducibility and analysis)
history = {
    "train_losses": [...],
    "test_losses": [...],
    "train_accs": [...],
    "test_accs": [...],
    "learning_rates": [...],
    "best_epoch": 9,
    "best_acc": 0.8966,
    "f1_macro": 0.8981,
    "f1_weighted": 0.8954,
}
json.dump(history, "checkpoints/training_history.json")
```

| File | Contents |
|------|----------|
| `gesture_tcn_best.pth` | Model weights at the epoch with highest test accuracy |
| `training_curves.png` | Loss, accuracy, and learning rate plots |
| `confusion_matrix.png` | Visual confusion matrix |
| `training_history.json` | All metrics from every epoch |

---

**Next**: [Stage 3: Deployment](./3_deploy.md) -- How the model is pruned, quantized, and exported for edge devices.
