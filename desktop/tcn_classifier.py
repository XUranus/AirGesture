#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*
#*   File:         tcn_classifier.py
#*   Author:       XUranus
#*   Date:         2026-03-21
#*   Description:  TCN-based gesture classifier with fallback support
#*
#*================================================================*/

"""
TCN Gesture Classifier Module

Provides gesture classification using trained TCN model.
Falls back to None if model is not available.
"""

import os
import json
import time
import logging
from collections import deque, Counter
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger("TCNClassifier")

# Constants
NUM_LANDMARKS = 21
NUM_COORDS = 3
RAW_DIM = NUM_LANDMARKS * NUM_COORDS
WRIST_IDX = 0
MID_FINGER_IDX = 9

PAIRS = [
    (4, 8), (8, 12), (12, 16), (16, 20),
    (4, 12), (4, 16), (4, 20),
    (8, 16), (8, 20), (12, 20)
]
N_PAIRS = len(PAIRS)

FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]
N_FINGERS = 5
FEATURE_DIM = RAW_DIM + RAW_DIM + 3 + N_PAIRS + N_FINGERS


def compute_features(raw_seq: np.ndarray) -> np.ndarray:
    """Compute hand-crafted features from raw landmark sequence."""
    raw_seq = np.asarray(raw_seq, dtype=np.float32)
    T = raw_seq.shape[0]
    lms = raw_seq.reshape(T, NUM_LANDMARKS, NUM_COORDS)

    wrist = lms[:, WRIST_IDX, :]
    relative = lms - wrist[:, np.newaxis, :]
    mid = lms[:, MID_FINGER_IDX, :]
    palm_size = np.maximum(np.linalg.norm(mid - wrist, axis=-1, keepdims=True), 1e-6)
    norm_lms = relative / palm_size[:, np.newaxis, :]
    norm_flat = norm_lms.reshape(T, -1).astype(np.float32)

    vel = np.zeros_like(norm_flat)
    if T > 1:
        vel[1:] = norm_flat[1:] - norm_flat[:-1]
        vel[0] = vel[1]

    wrist_vel = np.zeros((T, 3), dtype=np.float32)
    if T > 1:
        wrist_vel[1:] = wrist[1:] - wrist[:-1]
        wrist_vel[0] = wrist_vel[1]

    dists = np.zeros((T, N_PAIRS), dtype=np.float32)
    for k, (i, j) in enumerate(PAIRS):
        dists[:, k] = np.linalg.norm(norm_lms[:, i] - norm_lms[:, j], axis=-1)

    angles = np.zeros((T, N_FINGERS), dtype=np.float32)
    for fi, chain in enumerate(FINGER_CHAINS):
        v1 = lms[:, chain[1]] - lms[:, chain[0]]
        v2 = lms[:, chain[-1]] - lms[:, chain[1]]
        n1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8
        n2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8
        cos_a = np.clip((v1 / n1 * v2 / n2).sum(-1), -1.0, 1.0)
        angles[:, fi] = np.arccos(cos_a)

    feat = np.concatenate([norm_flat, vel, wrist_vel, dists, angles], axis=1)
    return feat.astype(np.float32)


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Resample sequence to fixed length via linear interpolation."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        return np.zeros((target_len, RAW_DIM), dtype=np.float32)
    n, d = seq.shape
    if n == 0:
        return np.zeros((target_len, d), dtype=np.float32)
    if n == target_len:
        return seq.copy()
    x_old = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.zeros((target_len, d), dtype=np.float32)
    for i in range(d):
        out[:, i] = np.interp(x_new, x_old, seq[:, i]).astype(np.float32)
    return out


class GestureBuffer:
    """Time-based sliding window buffer for landmarks."""

    def __init__(self, seq_len: int, window_seconds: float = 1.0,
                 source_fps: float = 30.0, fill_frames: int = 3):
        self.seq_len = int(seq_len)
        self.window_seconds = float(window_seconds)
        self.source_fps = float(source_fps) if source_fps else 30.0
        if not np.isfinite(self.source_fps) or self.source_fps < 1.0 or self.source_fps > 240.0:
            self.source_fps = 30.0
        self.frame_dt = 1.0 / self.source_fps
        self.fill_frames = int(fill_frames)
        self.buf: deque = deque()
        self.miss = 0
        self.last: Optional[np.ndarray] = None

    def _sanitize(self, lm) -> np.ndarray:
        arr = np.asarray(lm, dtype=np.float32).reshape(-1)
        if arr.size < RAW_DIM:
            out = np.zeros((RAW_DIM,), dtype=np.float32)
            out[:arr.size] = arr
            arr = out
        elif arr.size > RAW_DIM:
            arr = arr[:RAW_DIM]
        return arr.astype(np.float32)

    def _purge(self, current_t: float):
        cutoff = float(current_t) - self.window_seconds
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def push(self, lm: Optional[np.ndarray], current_t: float):
        current_t = float(current_t)
        if lm is not None:
            arr = self._sanitize(lm)
            self.miss = 0
            self.last = arr.copy()
            self.buf.append((current_t, arr.copy()))
        else:
            self.miss += 1
            if self.miss <= self.fill_frames and self.last is not None:
                self.buf.append((current_t, self.last.copy()))
        self._purge(current_t)

    def coverage_seconds(self, current_t: Optional[float] = None) -> float:
        if current_t is not None:
            self._purge(current_t)
        if not self.buf:
            return 0.0
        return max(0.0, float(self.buf[-1][0] - self.buf[0][0]) + self.frame_dt)

    def ready(self, current_t: Optional[float] = None) -> bool:
        if current_t is not None:
            self._purge(current_t)
        return len(self.buf) >= 2 and self.coverage_seconds() >= self.window_seconds

    def get(self, current_t: float) -> Optional[np.ndarray]:
        self._purge(current_t)
        if not self.ready():
            return None
        arr = np.stack([x[1] for x in self.buf], axis=0).astype(np.float32)
        return resample_sequence(arr, self.seq_len)

    def reset(self):
        self.buf.clear()
        self.miss = 0
        self.last = None

    def __len__(self):
        return len(self.buf)


class TCNClassifier:
    """
    TCN-based gesture classifier.

    Uses trained model if available, returns None otherwise.
    """

    def __init__(self, config_path: str, model_path: str,
                 threshold: float = 0.5, smooth_window: int = 5,
                 window_seconds: float = 1.0, source_fps: float = 30.0,
                 fill_frames: int = 3):
        self.is_available = False
        self.class_names: List[str] = []
        self.threshold = threshold

        # Check if model files exist
        if not os.path.exists(config_path):
            logger.warning(f"TCN config not found: {config_path}")
            return
        if not os.path.exists(model_path):
            logger.warning(f"TCN model not found: {model_path}")
            return

        try:
            # Load config
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            self.class_names = cfg["class_names"]
            self.seq_len = int(cfg["seq_len"])
            self.feat_dim = int(cfg["feature_dim"])
            self.norm_mean = np.array(cfg["normalize_mean"], dtype=np.float32)
            self.norm_std = np.array(cfg["normalize_std"], dtype=np.float32)

            # Load PyTorch model
            import torch
            import torch.nn as nn

            # Define model architecture
            class CausalConv1d(nn.Module):
                def __init__(self, in_ch, out_ch, ks, dilation=1):
                    super().__init__()
                    self.pad = (ks - 1) * dilation
                    self.conv = nn.Conv1d(in_ch, out_ch, ks, padding=self.pad, dilation=dilation, bias=False)

                def forward(self, x):
                    o = self.conv(x)
                    if self.pad > 0:
                        o = o[:, :, :-self.pad]
                    return o

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
                    return self.act(self.net(x) + x)

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
                    self.skip = nn.Sequential(
                        nn.Conv1d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm1d(out_ch),
                    )
                    self.act = nn.ReLU(inplace=True)

                def forward(self, x):
                    return self.act(self.net(x) + self.skip(x))

            class GestureTCN(nn.Module):
                def __init__(self, num_classes, feat_dim, dropout=0.15):
                    super().__init__()
                    self.stem = nn.Sequential(
                        nn.Conv1d(feat_dim, 48, 1, bias=False),
                        nn.BatchNorm1d(48),
                        nn.ReLU(inplace=True),
                    )
                    self.blocks = nn.Sequential(
                        ResBlock(48, 3, 1, dropout),
                        ResBlock(48, 3, 2, dropout),
                        ChannelBlock(48, 64, 3, 4, dropout),
                        ResBlock(64, 3, 1, dropout),
                    )
                    self.pool = nn.AdaptiveAvgPool1d(1)
                    self.head = nn.Sequential(
                        nn.Linear(64, 32),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(32, num_classes),
                    )

                def forward(self, x):
                    x = self.stem(x)
                    x = self.blocks(x)
                    x = self.pool(x).squeeze(-1)
                    return self.head(x)

            # Create and load model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = GestureTCN(len(self.class_names), self.feat_dim)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.to(self.device)
            self.model.eval()

            # Buffer for sliding window
            self.buffer = GestureBuffer(
                self.seq_len,
                window_seconds=window_seconds,
                source_fps=source_fps,
                fill_frames=fill_frames
            )

            # Smoothing
            self.smooth_win: deque = deque(maxlen=max(int(smooth_window), 1))
            self.prob_win: deque = deque(maxlen=max(int(smooth_window), 1))

            self.is_available = True
            logger.info(f"TCN classifier loaded: {len(self.class_names)} classes on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load TCN model: {e}")
            self.is_available = False

    def push(self, lm: Optional[np.ndarray], current_t: float):
        """Push landmark data to buffer."""
        if self.is_available:
            self.buffer.push(lm, current_t)

    def push_missing(self, current_t: float):
        """Mark a frame as missing."""
        if self.is_available:
            self.buffer.push(None, current_t)

    def predict(self, current_t: float = 0.0) -> Tuple[Optional[str], float]:
        """
        Predict gesture.

        Returns:
            (gesture_name, confidence) or (None, 0.0) if not ready
        """
        if not self.is_available:
            return None, 0.0

        raw = self.buffer.get(current_t)
        if raw is None:
            self.smooth_win.clear()
            self.prob_win.clear()
            return None, 0.0

        # Compute features
        feat = compute_features(raw)
        feat = (feat - self.norm_mean) / (self.norm_std + 1e-8)
        x = feat.T[np.newaxis].astype(np.float32)

        # Predict
        import torch
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).to(self.device)
            logits = self.model(x_tensor)
            logits = logits.detach().cpu().numpy().reshape(-1)

        logits = logits.astype(np.float32)

        # Softmax
        probs = np.exp(logits - logits.max())
        probs_sum = float(probs.sum())
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            return None, 0.0
        probs /= probs_sum

        # Smooth probabilities
        self.prob_win.append(probs.copy())
        avg = np.stack(list(self.prob_win), axis=0).mean(0)

        # Get best class
        idx = int(avg.argmax())
        self.smooth_win.append(idx)

        # Majority voting
        si = Counter(self.smooth_win).most_common(1)[0][0]
        sc = float(avg[si])

        if sc < self.threshold:
            return "unknown", sc

        return self.class_names[si], sc

    def reset(self):
        """Reset buffer and smoothing state."""
        if self.is_available:
            self.buffer.reset()
            self.smooth_win.clear()
            self.prob_win.clear()

    def get_gesture_event(self, gesture_name: str) -> Optional[str]:
        """Convert TCN gesture name to GestureEvent name."""
        mapping = {
            "swipe_up": "SWIPE_UP",
            "swipe_down": "SWIPE_DOWN",
            "grab": "GRAB",
            "release": "RELEASE",
        }
        return mapping.get(gesture_name.lower())
