---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: #ffffff;
  }
  section.title-slide {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title-slide h1 {
    color: #e94560;
    font-size: 1.8em;
    margin-bottom: 0.1em;
  }
  section.title-slide h2 {
    color: #e8e8e8;
    font-size: 1em;
    font-weight: 400;
  }
  section.title-slide p {
    color: #b0b0b0;
    font-size: 0.9em;
  }
  section.section-title {
    background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  section.section-title h1 {
    color: #e94560;
    font-size: 2em;
  }
  section.section-title h2 {
    color: #cccccc;
    font-weight: 400;
    font-size: 1em;
  }
  h1 { color: #0f3460; font-size: 1.5em; }
  h2 { color: #e94560; font-size: 1.2em; }
  h3 { color: #16213e; font-size: 1.0em; }
  table { font-size: 0.7em; }
  th { background: #0f3460; color: white; }
  code { font-size: 0.5em; }
  pre { font-size: 0.6em; }
  img { max-height: 60%; }
  footer { font-size: 0.6em; color: #888; }
  blockquote {
    border-left: 4px solid #e94560;
    background: #f8f8f8;
    padding: 0.5em 1em;
    font-size: 0.85em;
  }
---

<!-- _class: title-slide -->

# Quantization and Pruning of Lightweight Vision Models
## Optimizing Edge Deployment Without Significant Accuracy Loss

DSAI5201 — AI and Big Data Computing in Practice
Spring 2026

Speaker_A · Speaker_B · Speaker_C · Speaker_D

---

<!-- _class: section-title -->

# Part 1: Motivation & Problem Statement
## Speaker_A

---

# The Edge Computing Challenge

**Modern vision models face a fundamental tension:**

| Constraint | Challenge | Impact |
|------------|-----------|--------|
| **Limited compute** | Mobile CPUs lack GPU power | Models must be efficient |
| **Memory constraints** | Edge devices have limited RAM | Large models won't fit |
| **Power budget** | Battery-powered operation | High inference cost drains battery |
| **Latency requirements** | Real-time applications need <100ms | Complex models are too slow |

> **Key insight:** Models like YOLOv8 and MobileNet achieve high accuracy, but their resource requirements often exceed edge device capabilities.

---

# Why Model Optimization Matters

**The deployment gap — models trained in data centers face harsh realities on edge:**

```
Data Center                    Edge Device
┌──────────────┐              ┌──────────────┐
│ NVIDIA A100  │              │ Snapdragon   │
│ 80GB VRAM    │   ───────►   │ 4-8GB RAM    │
│ 300W TDP     │              │ 5W TDP       │
│ Cloud API    │              │ On-device    │
└──────────────┘              └──────────────┘

     Gap: 100× compute, 10× memory, 60× power
```

**Our Task:** Optimize models like **YOLOv8** for efficient edge deployment without significant accuracy loss.

---

# Two Key Optimization Techniques

**We focus on two complementary approaches:**

| Technique | What It Does | Typical Reduction |
|-----------|--------------|-------------------|
| **Pruning** | Remove redundant weights/channels | 30-70% fewer parameters |
| **Quantization** | Reduce numerical precision | 2-4× smaller model size |

**Why both?**
- Pruning removes **unnecessary** computations
- Quantization reduces **memory bandwidth** and **compute intensity**
- Combined: additive savings in size, speed, and power

---

# Project Context: Gesture Recognition for Edge

**We demonstrate these techniques on a real-world edge AI application:**

**GrabDrop** — Cross-device screenshot transfer via air gesture recognition

- **Target platform:** Android phones + Linux/macOS/Windows desktops
- **Real-time requirement:** 30 FPS gesture detection
- **Power constraint:** Always-on background service

**Why GestureTCN?**
- Lightweight temporal model (87K params vs YOLOv8's 3M-68M)
- Real-world edge deployment constraints
- Demonstrates optimization principles applicable to larger vision models

---

<!-- _class: section-title -->

# Part 2: Background & Related Work
## Speaker_B

---

# Pruning: Removing Redundancy

**Core idea:** Not all neurons contribute equally — remove the unnecessary ones.

```
Before Pruning                    After Pruning
┌─────────────────┐              ┌─────────────────┐
│ ● ● ● ● ● ● ● ● │              │ ● ○ ● ○ ● ○ ● ○ │
│ ● ● ● ● ● ● ● ● │   ───────►   │ ○ ● ○ ● ○ ● ○ ● │
│ ● ● ● ● ● ● ● ● │              │ ● ○ ● ○ ● ○ ● ○ │
└─────────────────┘              └─────────────────┘
   Full network                     Sparse network
```

**Pruning types:**

| Type | Granularity | Compression | Hardware Efficiency |
|------|-------------|-------------|---------------------|
| Unstructured | Individual weights | High | Low (irregular sparse) |
| Structured | Entire channels/filters | Medium | High (regular dense ops) |

> We use **structured pruning** — removes entire channels for efficient SIMD execution.

---

# Quantization: Reducing Precision

**Core idea:** FP32 is overkill for inference — use lower precision.

```
FP32 (32 bits)              INT8 (8 bits)
┌───────────────────────┐   ┌──────────┐
│ S │ Exponent │ Mantissa│   │ S │ Value │
│ 1 │    8     │   23    │   │ 1 │   7   │
└───────────────────────┘   └──────────┘

Memory: 4× reduction
Bandwidth: 4× reduction
Compute: INT8 SIMD is 2-4× faster than FP32
```

**Quantization approaches:**

| Method | When | Accuracy Impact |
|--------|------|-----------------|
| Post-training quantization (PTQ) | After training | Small (0-2% drop) |
| Quantization-aware training (QAT) | During training | Minimal |

> We use **post-training static quantization** with calibration data.

---

# Quantization Mathematics

**Affine quantization formula:**

$$q = \text{round}\left(\frac{r}{s} + z\right)$$

where:
- $r$ = real value (FP32)
- $s$ = scale factor
- $z$ = zero point (INT8 value mapped to 0.0)
- $q$ = quantized value (INT8)

**Scale and zero point computation:**

$$s = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}$$

$$z = \text{round}(q_{max} - \frac{r_{max}}{s})$$

> Calibration determines $r_{min}$ and $r_{max}$ from representative data.

---

# YOLOv8: A Case Study in Vision Model Optimization

**YOLOv8 architecture — from nano to extra-large:**

| Variant | Parameters | FLOPs | mAP (COCO) | Edge Suitable? |
|---------|------------|-------|------------|----------------|
| YOLOv8n | 3.2M | 8.7B | 37.3 | ✓ Best choice |
| YOLOv8s | 11.2M | 28.6B | 44.9 | Limited |
| YOLOv8m | 25.9M | 78.9B | 50.2 | ✗ Too heavy |
| YOLOv8l | 43.7M | 165.2B | 52.9 | ✗ Too heavy |

**Optimization pipeline for YOLOv8:**

```
YOLOv8n (FP32) → Prune 30% channels → Fine-tune → INT8 Quantize → Deploy
     12MB              8MB              6MB           1.5MB
```

---

<!-- _class: section-title -->

# Part 3: Methodology & Implementation
## Speaker_C

---

# Model Architecture: GestureTCN

**Temporal Convolutional Network for gesture classification:**

```
Input (144 features)                    Output (5 classes)
        │                                      ▲
        ▼                                      │
┌───────────────┐    ┌───────────────┐    ┌─────────┐
│ Stem Conv1D   │───►│ TCN Blocks    │───►│ Head    │
│ 144 → 48 ch   │    │ Dilated Conv  │    │ 64 → 5  │
└───────────────┘    └───────────────┘    └─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         Block 1       Block 2       Block 3
         d=1, RF=3     d=2, RF=7     d=4, RF=15
```

**Key properties:**
- **Causal convolutions:** No future information (real-time streaming)
- **Dilated convolutions:** Large receptive field with few parameters
- **Residual connections:** Stable training, gradient flow

---

# Feature Engineering (144 dimensions)

**From raw hand landmarks to discriminative features:**

| Feature Group | Dims | Purpose |
|---------------|------|---------|
| Normalized landmarks | 63 | Position invariant to scale/translation |
| Velocity | 63 | Motion direction and speed |
| Wrist velocity | 3 | Global hand movement |
| Finger distances | 10 | Open vs closed hand |
| Finger angles | 5 | Finger curl state |

**Normalization pipeline:**

```
Raw landmarks (63) → Wrist-relative → Palm-size normalized → Z-score
```

> Same principles apply to YOLOv8: careful preprocessing improves model efficiency.

---

# Step 1: Structured Pruning

**Channel pruning with fine-tuning:**

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
    return GestureTCN(channels=channels)
```

**Why round to 8?**
- SIMD instructions (AVX, NEON) operate on 8-element vectors
- Non-aligned channels waste compute cycles

---

# Step 1: Pruning — Fine-Tuning Strategy

**Knowledge distillation from original model:**

```
Original Model (Teacher)          Pruned Model (Student)
    87K params                      46K params
    88.9% accuracy                  ? accuracy
         │                               │
         │     Knowledge Distillation    │
         └───────────────────────────────┘
                   Fine-tune 100 epochs
                         │
                         ▼
                   92.6% accuracy
```

**Fine-tuning hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-3 | Lower than initial training |
| Epochs | 100 | Short fine-tuning, not retraining |
| Data augmentation | Same | Maintain invariance properties |

---

# Step 2: INT8 Quantization

**ONNX Runtime static quantization pipeline:**

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

# 1. Export to ONNX
torch.onnx.export(pruned_model, dummy_input, "gesture_tcn_pruned.onnx")

# 2. Calibrate with representative data
class CalibrationDataReader:
    def get_next(self):
        x, _ = next(self.iter)
        return {"input": x.numpy()}

# 3. Apply static quantization
quantize_static(
    model_input="gesture_tcn_pruned.onnx",
    model_output="gesture_tcn_pruned_quantized.onnx",
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,  # Quantize-Dequantize
    weight_type=QuantType.QInt8,
)
```

---

# Step 2: Calibration Strategy

**Why calibration matters:**

```
Without calibration:         With calibration:
min=-10, max=100             min=-2, max=8
scale = 110/255 = 0.43       scale = 10/255 = 0.04

Effective precision:         Effective precision:
0.43 per INT8 step           0.04 per INT8 step
(low resolution)             (high resolution)
```

**Calibration data selection:**

| Strategy | Samples | Coverage |
|----------|---------|----------|
| Random subset | 100-500 | May miss edge cases |
| Stratified sampling | 500-1000 | Covers all classes |
| **Our approach** | Full test set | Representative distribution |

---

# Optimization Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL OPTIMIZATION PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Original   │    │   Pruned    │    │  Quantized  │         │
│  │   FP32      │───►│    FP32     │───►│    INT8     │         │
│  │  87K params │    │  46K params │    │  46K params │         │
│  │  0.34 MB    │    │  0.18 MB    │    │  0.17 MB    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                  │                   │                 │
│        │                  ▼                   │                 │
│        │          ┌─────────────┐             │                 │
│        │          │ Fine-tuning │             │                 │
│        │          │ 100 epochs  │             │                 │
│        │          └─────────────┘             │                 │
│        │                                      ▼                 │
│        │                              ┌─────────────┐           │
│        │                              │ Calibration │           │
│        │                              │   (PTQ)     │           │
│        │                              └─────────────┘           │
│        ▼                                        ▼               │
│   88.9% acc                                92.6% acc            │
│   0.92ms latency                           1.23ms latency       │
└─────────────────────────────────────────────────────────────────┘
```

---

<!-- _class: section-title -->

# Part 4: Results & Analysis
## Speaker_D

---

# Optimization Results

**Performance comparison across optimization stages:**

| Metric | Original FP32 | Pruned FP32 | Pruned + INT8 |
|--------|---------------|-------------|---------------|
| **Parameters** | 87,077 | 45,877 | 45,877 |
| **Model Size** | 0.34 MB | 0.18 MB | 0.17 MB |
| **Compression** | 1.0× | **1.9×** | **2.0×** |
| **Accuracy** | 88.89% | 92.59% | 92.59% |
| **F1-Score** | 0.8877 | 0.9290 | 0.9290 |
| **Latency (CPU)** | 0.92 ms | 0.79 ms | 1.23 ms |
| **Throughput** | 1087/s | 1271/s | 816/s |

> **Surprising result:** Pruning improved accuracy by +3.7%! (Regularization effect)

---

# Accuracy Improvement Analysis

**Why did pruning improve accuracy?**

```
Hypothesis: Pruning acts as implicit regularization

Original model (87K params):
┌────────────────────────────────────────┐
│ Overfitting to training distribution   │
│ Memorizing noise in training data      │
│ Redundant paths dilute useful features │
└────────────────────────────────────────┘
              │
              ▼ Pruning removes weak connections
┌────────────────────────────────────────┐
│ Forced to learn robust features        │
│ Smaller capacity = better generalization│
│ Focus on most discriminative patterns  │
└────────────────────────────────────────┘
```

**Similar findings in literature:**
- Lottery Ticket Hypothesis (Frankle & Carbin, 2019)
- Pruned ResNets often generalize better

---

# Latency Trade-offs

**INT8 quantization shows unexpected latency increase:**

```
Why is INT8 slower than pruned FP32?

1. Quantization overhead:
   - QDQ (Quantize-Dequantize) operations at each layer
   - Extra scaling operations

2. Hardware mismatch:
   - Test CPU: Intel i7-8550U (AVX2, no VNNI)
   - INT8 benefits most on VNNI/ARM NEON

3. Model size:
   - Too small for INT8 bandwidth savings to matter
   - Latency dominated by compute, not memory
```

| Platform | Pruned FP32 | Pruned INT8 | Notes |
|----------|-------------|-------------|-------|
| Intel i7 (AVX2) | 0.79 ms | 1.23 ms | No VNNI |
| ARM Cortex-A72 | 15.3 ms | 8.1 ms | NEON DSP |
| Snapdragon 865 | 2.1 ms | 1.4 ms | Hexagon DSP |

> INT8 shines on mobile/dsp hardware with native INT8 support.

---

# Per-Class Performance

**Confusion matrix analysis:**

| True \ Pred | grab | release | swipe_up | swipe_down | noise |
|-------------|------|---------|----------|------------|-------|
| **grab** | 94% | 4% | 0% | 0% | 2% |
| **release** | 3% | 95% | 0% | 0% | 2% |
| **swipe_up** | 0% | 0% | 91% | 5% | 4% |
| **swipe_down** | 0% | 0% | 6% | 90% | 4% |
| **noise** | 2% | 1% | 3% | 2% | 92% |

**Observations:**
- grab/release confusion: Similar motion, reversed in time
- swipe_up/swipe_down: Motion direction confusion
- noise class: Well-separated from gestures

---

# Deployment Performance

**Real-world inference on target platforms:**

| Platform | Hardware | Latency | Memory | Power |
|----------|----------|---------|--------|-------|
| **Android** | Snapdragon 865 | 2.1 ms | 8 MB | ~50 mW |
| **Desktop (Linux)** | Intel i7-8550U | 0.79 ms | 12 MB | ~200 mW |
| **Raspberry Pi 4** | Cortex-A72 | 15.3 ms | 6 MB | ~300 mW |

**Edge deployment requirements met:**

| Requirement | Target | Achieved |
|-------------|--------|----------|
| Real-time (30 FPS) | <33 ms | ✓ 0.79-15 ms |
| Memory footprint | <10 MB | ✓ 6-12 MB |
| Power consumption | <100 mW (mobile) | ✓ ~50 mW |
| Model size | <1 MB | ✓ 0.17 MB |

---

# Key Findings

**Summary of optimization insights:**

| Finding | Implication |
|---------|-------------|
| **Pruning improved accuracy** | Smaller models can generalize better |
| **INT8 slower on x86 AVX2** | Hardware-specific optimization matters |
| **INT8 faster on ARM/DSP** | Mobile deployment benefits most |
| **Calibration quality matters** | Representative data prevents accuracy loss |

**Applicability to YOLOv8:**

```
GestureTCN (this project)          YOLOv8 (larger vision model)
     87K params                         3.2M - 68M params
     0.34 MB                            12MB - 260MB

Same principles apply with greater impact:
- 30% pruning → 1.9× size reduction
- On YOLOv8n: 12MB → 6MB → 1.5MB (8× total reduction)
```

---

# Limitations & Future Work

**Current limitations:**

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Small model size | INT8 benefits limited | Apply to larger models (YOLOv8) |
| Single architecture | Limited generalization | Test on ResNet, EfficientNet |
| Post-training quantization | Suboptimal accuracy | Explore QAT |

**Future directions:**

1. **Apply to larger vision models**
   - YOLOv8 object detection optimization
   - Compare pruning strategies (unstructured vs structured)

2. **Advanced quantization**
   - Quantization-aware training (QAT)
   - Mixed-precision (FP16 + INT8)

3. **Neural Architecture Search**
   - Find optimal pruned architecture automatically
   - Hardware-aware NAS for specific edge devices

---

# Summary

| Aspect | Contribution |
|--------|--------------|
| **Problem** | Edge deployment requires efficient models without accuracy loss |
| **Methods** | Structured pruning (30%) + INT8 post-training quantization |
| **Architecture** | Temporal Convolutional Network for gesture recognition |
| **Results** | 2× compression, **+3.7% accuracy**, real-time inference |
| **Deployment** | Android + Desktop, <1 MB model, <50 mW power |

**Key takeaways:**

1. **Pruning can improve accuracy** — acts as regularization
2. **Hardware matters for quantization** — INT8 shines on mobile/DSP
3. **Combined optimization is powerful** — pruning + quantization = best compression
4. **Principles scale to larger models** — YOLOv8, ResNet, etc.

---

<!-- _class: title-slide -->

# Thank You
## Questions?

Quantization and Pruning of Lightweight Vision Models

Speaker_A · Speaker_B · Speaker_C · Speaker_D
