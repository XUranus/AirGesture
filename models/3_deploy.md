# Stage 3: Deployment (3.deploy.ipynb)

This document walks you through the deployment notebook, explaining model pruning, ONNX export, and INT8 quantization -- the techniques used to make the trained model small and fast enough to run on edge devices (like smartphones).

---

## Table of Contents

1. [Big Picture: What This Stage Does](#1-big-picture)
2. [Sections 1-4: Setup and Data Loading](#2-setup-and-data-loading)
3. [Section 6: Structured Pruning](#3-structured-pruning)
4. [Section 7: Fine-tuning the Pruned Model](#4-fine-tuning-the-pruned-model)
5. [Section 8: ONNX Export](#5-onnx-export)
6. [Section 9: INT8 Quantization](#6-int8-quantization)
7. [Sections 10-11: Verification and Evaluation](#7-verification-and-evaluation)
8. [Section 12: Deployment Configuration](#8-deployment-configuration)
9. [Summary: The Full Optimization Pipeline](#9-summary)

---

## 1. Big Picture

The trained model from Stage 2 works well, but it's a **PyTorch model in FP32 (32-bit floating point)** format. For deployment on edge devices (Android phones, embedded systems), we need to:

1. **Make it smaller** -- less memory, smaller download
2. **Make it faster** -- fewer computations per inference
3. **Convert the format** -- from PyTorch to ONNX (which mobile runtimes can execute)

```
Original Model (FP32, 87K params, 0.34 MB)
    |
    | [Structured Pruning: remove 30% of channels]
    v
Pruned Model (FP32, 46K params, 0.18 MB)  -- 1.9x smaller
    |
    | [Fine-tuning: recover accuracy]
    v
Pruned Model (fine-tuned, accuracy recovered)
    |
    | [ONNX Export]
    v
ONNX Model (FP32, portable format)
    |
    | [INT8 Quantization: FP32 -> INT8 weights]
    v
Quantized ONNX Model (INT8, ~4x smaller)  -- Ready for deployment!
```

---

## 2. Setup and Data Loading

### Sections 1-4

The notebook loads:
- The **trained model weights** (`gesture_tcn_best.pth`) from Stage 2
- The **preprocessed dataset** (same caches from Stage 1) for fine-tuning and calibration
- The **normalization statistics** for consistent feature scaling

```python
# Load the trained model
original_model = GestureTCN().to(DEVICE)
original_model.load_state_dict(
    torch.load("checkpoints/gesture_tcn_best.pth", map_location=DEVICE)
)
original_model.eval()

# Result: 87,077 parameters, 0.3365 MB
```

Two utility functions are defined for measuring model size:

```python
def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Calculate model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)
```

---

## 3. Structured Pruning

### Section 6: `create_pruned_model()`

**Pruning** removes unnecessary parts of a neural network. There are two types:

| Type | What it removes | Actual speedup? |
|------|----------------|-----------------|
| **Unstructured** | Individual weights (set to zero) | No -- needs sparse hardware |
| **Structured** | Entire channels/filters | Yes -- physically smaller model |

This project uses **structured pruning**, which creates a genuinely smaller model.

### How Structured Pruning Works

Instead of removing individual weights, structured pruning reduces the **number of channels** in each layer:

```python
def create_pruned_model(prune_ratio=0.3):
    def round_to_8(x):
        return max(8, round(x / 8) * 8)  # Round to multiple of 8

    channels = {
        "stem": round_to_8(int(48 * (1 - 0.3))),  # 48 -> 32
        "mid":  round_to_8(int(48 * (1 - 0.3))),   # 48 -> 32
        "out":  round_to_8(int(64 * (1 - 0.3))),   # 64 -> 48
        "head": round_to_8(int(32 * (1 - 0.3))),   # 32 -> 24
    }
    return GestureTCN(channels=channels)  # Create smaller model
```

**What `round_to_8` does**: Modern CPUs and GPUs process data in chunks (SIMD instructions work on 8/16/32 values at once). Rounding channel counts to multiples of 8 ensures optimal hardware utilization.

**Before and after:**

```
Original Model:                    Pruned Model:
  Stem: 144 -> 48 channels          Stem: 144 -> 32 channels
  Block1: 48 -> 48                   Block1: 32 -> 32
  Block2: 48 -> 48                   Block2: 32 -> 32
  Block3: 48 -> 64                   Block3: 32 -> 48
  Block4: 64 -> 64                   Block4: 48 -> 48
  Head: 64 -> 32 -> 5               Head: 48 -> 24 -> 5

  Parameters: 87,077                 Parameters: 45,877
  Size: 0.3365 MB                    Size: 0.1781 MB
  Compression: 1.00x                 Compression: 1.90x
```

**Key insight**: The pruned model is a **brand new model** with fewer channels -- it doesn't inherit any weights from the original. This is why fine-tuning is necessary.

### Why 30% Pruning?

The `PRUNE_RATIO = 0.3` was chosen as a balance between:
- **Too little pruning** (e.g., 10%): Minimal size reduction
- **Too much pruning** (e.g., 60%): Significant accuracy loss that's hard to recover

30% typically achieves ~1.9x compression with recoverable accuracy loss.

---

## 4. Fine-tuning the Pruned Model

### Section 7: `fine_tune_model()`

The pruned model has randomly initialized weights, so it needs to be **trained from scratch** (or fine-tuned). The code trains it for 100 epochs:

```python
def fine_tune_model(model, train_loader, test_loader, epochs=100, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(1, epochs + 1):
        # Standard training loop (same as Stage 2)
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        
        scheduler.step()
        
        # Track best model
        metrics = evaluate_model(model, test_loader, DEVICE)
        if metrics["accuracy"] > best_acc:
            best_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_state)  # Restore best weights
    return model, best_acc
```

**Fine-tuning results:**
- The pruned model reached **89.66% test accuracy** after fine-tuning
- The accuracy is identical to the original model, demonstrating that pruning did not hurt performance
- The learning rate is lower (1e-3 vs 2e-3) since the model is smaller and needs gentler updates

> **Note**: All three models (original, pruned, quantized) achieve identical accuracy (89.66%) on the test set. This is attributed to the limited test set size (29 samples), where accuracy granularity is approximately 3.4% (1/29). The models correctly classify the same 26/29 samples.

After fine-tuning, the pruned model weights are saved:

```python
torch.save(pruned_model.state_dict(), "checkpoints/gesture_tcn_structured_pruned.pth")
```

---

## 5. ONNX Export

### Section 8: `export_to_onnx()`

**ONNX (Open Neural Network Exchange)** is a standardized format for neural network models. Unlike PyTorch's `.pth` files (which require Python + PyTorch), ONNX models can run on:
- Android/iOS (via ONNX Runtime Mobile)
- Web browsers (via ONNX.js)
- Edge devices (via TensorRT, OpenVINO, etc.)

```python
def export_to_onnx(model, save_path, name):
    model_cpu = model.cpu().eval()
    dummy_input = torch.randn(1, FEATURE_DIM, SEQ_LEN)  # (1, 144, 30)

    torch.onnx.export(
        model_cpu,
        dummy_input,
        save_path,
        input_names=["input"],           # Name for the input tensor
        output_names=["logits"],         # Name for the output tensor
        dynamic_axes={                    # Allow variable batch size
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,               # ONNX operator set version
    )

    # Verify the exported model is valid
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
```

**How ONNX export works:**

1. **Tracing**: PyTorch runs the model with a dummy input and records all operations
2. **Graph construction**: The recorded operations are converted to an ONNX computation graph
3. **Serialization**: The graph + weights are saved to a `.onnx` file
4. **Validation**: `onnx.checker.check_model()` verifies the file is well-formed

**Key parameter -- `dynamic_axes`**: By specifying `{0: "batch"}`, we tell ONNX that the first dimension (batch size) can vary. This allows the model to process 1 sample at a time (real-time inference) or multiple samples in a batch.

**Opset version 18**: ONNX defines versioned sets of operators. Higher versions support more operations. Version 18 is modern enough to support all operations used in our model.

---

## 6. INT8 Quantization

### Section 9: Static Quantization with Calibration

**Quantization** converts model weights and activations from **32-bit floating point (FP32)** to **8-bit integers (INT8)**:

```
FP32: 3.14159265...  (32 bits = 4 bytes per number)
INT8: 3              (8 bits = 1 byte per number) + scale/zero-point
```

This gives approximately **4x memory reduction** with typically <1% accuracy loss.

### Types of Quantization

| Type | When quantization happens | Quality | Speed |
|------|--------------------------|---------|-------|
| **Dynamic** | At inference time | Good | Moderate |
| **Static** | Ahead of time, using calibration data | Better | Fast |

This project uses **static quantization** with a **calibration dataset**:

```python
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader

class GestureCalibrationDataReader(CalibrationDataReader):
    """Feeds real data to determine optimal quantization parameters."""
    
    def __init__(self, data_loader):
        self.iter = iter(data_loader)
    
    def get_next(self):
        try:
            x, _ = next(self.iter)
            return {"input": x.numpy()}  # Feed one batch
        except StopIteration:
            return None
```

### How Static Quantization Works

1. **Calibration**: Run real data through the FP32 model and observe the range of values at each layer
2. **Determine scales**: For each layer, compute a `scale` and `zero_point` that maps FP32 values to INT8:
   ```
   INT8_value = round(FP32_value / scale) + zero_point
   ```
3. **Convert weights**: Apply the mapping to all model weights
4. **Insert QDQ nodes**: Add Quantize/Dequantize operations in the computation graph

```python
quantize_static(
    model_input=pruned_onnx_path,        # Input: FP32 ONNX model
    model_output=quantized_onnx_path,     # Output: INT8 ONNX model
    calibration_data_reader=calib_reader,  # Real data for calibration
    quant_format=QuantFormat.QDQ,          # Quantize-Dequantize format
    per_channel=False,                     # Same scale for all channels in a layer
    weight_type=QuantType.QInt8,           # Use signed 8-bit integers
)
```

**`QuantFormat.QDQ`**: This format inserts explicit Quantize and Dequantize nodes around each operation. The runtime can then fuse these with the actual operations for optimal performance.

**`per_channel=False`**: Uses one scale factor per entire tensor (simpler, slightly less accurate than per-channel which uses one scale per output channel).

---

## 7. Verification and Evaluation

### Section 10: ONNX Inference Verification

After export, the code verifies that the ONNX model produces valid outputs:

```python
def verify_onnx_inference(onnx_path, model_name):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, FEATURE_DIM, SEQ_LEN).astype(np.float32)
    
    inputs = {session.get_inputs()[0].name: dummy_input}
    outputs = session.run(None, inputs)
    
    print(f"Output shape: {outputs[0].shape}")  # Should be (1, 5)
```

**ONNX Runtime** is a high-performance inference engine that can run ONNX models. The `CPUExecutionProvider` runs on CPU; other providers are available for GPU (CUDA), NPU, etc.

### Section 11: Quantized Model Evaluation

The quantized ONNX model is evaluated on the test set to measure accuracy degradation:

```python
ort_session = ort.InferenceSession(quantized_onnx_path, providers=["CPUExecutionProvider"])

for x, y in test_loader:
    for i in range(x.shape[0]):
        sample = x[i:i+1].numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: sample}
        ort_outputs = ort_session.run(None, ort_inputs)
        pred = np.argmax(ort_outputs[0], axis=1)[0]
```

This compares the quantized model's predictions against the PyTorch pruned model to ensure quantization didn't significantly hurt accuracy.

---

## 8. Deployment Configuration

### Section 12: `config.json`

The deployment config bundles everything needed to use the model on a device:

```python
config = {
    "model_name": "gesture_tcn",
    "class_names": ["grab", "release", "swipe_up", "swipe_down", "noise"],
    "seq_len": 30,
    "feature_dim": 144,
    "raw_dim": 63,
    "num_classes": 5,
    "num_landmarks": 21,
    "normalize_mean": [...],     # 144 values
    "normalize_std": [...],      # 144 values
    "pairs": [[4,8], [8,12], ...],
    "fingertip_ids": [4, 8, 12, 16, 20],
    "base_ids": [2, 5, 9, 13, 17],
    "finger_chains": [[0,1,2,3,4], ...],
    "prune_ratio": 0.3,
    "pruned_channels": {"stem": 32, "mid": 32, "out": 48, "head": 24},
    "original_params": 87077,
    "pruned_params": 45877,
}
```

**Why this config is needed**: The edge device (e.g., Android app) needs to know:
1. How to compute features from raw landmarks (pairs, chains, etc.)
2. How to normalize features (mean and std)
3. How to interpret the model output (class names)
4. The model architecture parameters (for creating the correct model structure)

---

## 9. Summary: The Full Optimization Pipeline

### Model Comparison

| Model | Parameters | Size | Accuracy | Format |
|-------|-----------|------|----------|--------|
| Original (FP32) | 87,077 | 0.34 MB | 89.66% | PyTorch |
| Pruned (FP32) | 45,877 | 0.18 MB | 89.66% | PyTorch |
| Original ONNX (FP32) | 87,077 | 0.09 MB | 89.66% | ONNX |
| Pruned ONNX (FP32) | 45,877 | 0.09 MB | 89.66% | ONNX |
| Pruned + Quantized (INT8) | 45,877 | 0.14 MB | 89.66% | ONNX |

### Key Observations

**1. ONNX Format Advantage**: The ONNX format produces more compact files than PyTorch's serialization format (0.09 MB vs 0.34 MB for the original model).

**2. Quantization Size Paradox**: The INT8 quantized model (0.14 MB) is larger than the FP32 ONNX models (0.09 MB). This occurs because:
- QDQ quantization inserts QuantizeLinear and DequantizeLinear nodes around each operation
- Each node adds metadata (scale, zero_point) to the model file
- For small models (<100 KB), this fixed overhead exceeds the weight compression benefit (4× reduction from FP32→INT8)
- This is a well-documented phenomenon for compact neural networks

**3. Why Use Quantization?**: Despite the larger file size, INT8 quantization is beneficial because:
- **Inference speed**: INT8 operations are 2-4× faster on mobile CPUs/NPUs
- **Memory bandwidth**: Transferring INT8 data is faster
- **Power efficiency**: Lower power consumption on mobile devices

**4. Accuracy Preservation**: All models achieve identical accuracy (89.66%) on the test set. This is due to the small test set (29 samples), where accuracy granularity is ~3.4% (1/29).

### Output Files

| File | Purpose |
|------|---------|
| `gesture_tcn_best.pth` | Original trained model (PyTorch) |
| `gesture_tcn_structured_pruned.pth` | Pruned model (PyTorch) |
| `gesture_tcn_original.onnx` | Original model in ONNX format |
| `gesture_tcn_pruned.onnx` | Pruned model in ONNX format |
| `gesture_tcn_pruned_quantized.onnx` | Pruned + quantized model for deployment |
| `config.json` | Complete deployment configuration |

### Deployment Flow on Edge Device

```
Camera Frame
     |
     | [MediaPipe hand detection]
     v
21 Landmarks (x, y, z) per frame
     |
     | [Buffer 30 frames]
     v
Raw Sequence: (30, 63)
     |
     | [Feature computation using config.json]
     v
Feature Matrix: (30, 144)
     |
     | [Normalize using mean/std from config.json]
     v
Normalized Features: (144, 30)  -- transposed for Conv1d
     |
     | [ONNX Runtime inference]
     v
Logits: (5,)
     |
     | [argmax]
     v
Predicted Class: "grab" / "release" / "swipe_up" / "swipe_down" / "noise"
```

---

**Previous**: [Stage 2: Model Training](./2_train.md)
**Overview**: [Project Overview](./0_overview.md)
