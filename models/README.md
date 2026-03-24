## Model Training & Development

For detailed information about the TCN model architecture, training process, and optimization techniques, see [MODEL_GUIDE.md](MODEL_GUIDE.md).

### Quick Start - Model Development

```bash
# 1. Preprocess dataset
cd models
jupyter notebook 1.preprocess.ipynb

# 2. Train model
jupyter notebook 2.train.ipynb

# 3. Optimize and export
jupyter notebook 3.deploy.ipynb
```

### Model Performance

| Model | Params | Size | Accuracy | Latency |
|-------|--------|------|----------|---------|
| Original (FP32) | 87K | 0.34 MB | 88.89% | 0.92 ms |
| Pruned (FP32) | 46K | 0.18 MB | 92.59% | 0.79 ms |
| Pruned + INT8 | 46K | 0.17 MB | 92.59% | 1.23 ms |
