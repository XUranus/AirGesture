# Gesture Motion Capture

A cross-platform desktop tool (Linux / Windows) for efficiently recording and
labelling short hand-gesture video clips to build ML training datasets.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Features

| Feature | Details |
|---|---|
| **Live preview** | Mirrored front-camera feed with FPS counter |
| **One-click recording** | Start / stop with a button or `Ctrl+R` |
| **Auto-naming** | Clips saved as `<Label>_<timestamp>.avi` |
| **CSV manifest** | Every clip appended to `labels.csv` with duration, frame count, resolution |
| **Per-label counter** | Shows how many clips already exist for the current label |
| **Dark UI** | Comfortable for long capture sessions |
| **Recording overlay** | Red border + blinking REC indicator + elapsed timer |
| **Configurable output** | Change dataset directory via UI or `--output` flag |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run (uses default camera 0, output to ./dataset/)
python main.py

# Or specify camera and output directory
python main.py --camera 1 --output /path/to/my_dataset

