#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*
#*   File:         gesture_classifier.py
#*   Author:       XUranus
#*   Date:         2026-03-24
#*   Description:  TCN-based gesture classifier using ONNX Runtime.
#*                 Python port of the Android GestureClassifier.kt.
#*
#================================================================*/

"""
TCN-based gesture classifier using ONNX Runtime.

Features (144 dimensions):
- Normalized landmarks (63): relative to wrist, divided by palm size
- Velocity (63): frame-to-frame difference
- Wrist velocity (3): wrist position change
- Finger distances (10): distances between fingertip pairs
- Finger angles (5): bending angle for each finger
"""

import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger("GestureClassifier")


@dataclass
class ClassificationResult:
    gesture: str
    confidence: float
    class_index: int
    valid_frames: int  # Number of real (non-zero) frames in window


class GestureClassifier:
    WRIST_IDX = 0
    MID_FINGER_IDX = 9  # middle finger MCP

    def __init__(self):
        self.is_initialized = False

        # Model configuration (loaded from config.json)
        self.seq_len: int = 30
        self.feature_dim: int = 144
        self.raw_dim: int = 63
        self.num_landmarks: int = 21
        self.class_names: List[str] = ["grab", "release", "swipe_up", "swipe_down", "noise"]

        # Normalization parameters
        self.normalize_mean: np.ndarray = np.zeros(0)
        self.normalize_std: np.ndarray = np.zeros(0)

        # Feature computation
        self.pairs: List[Tuple[int, int]] = []
        self.finger_chains: List[List[int]] = []

        # ONNX Runtime session
        self.session = None

        # Sliding window
        self.window: deque = deque()
        self.is_real_frame: deque = deque()

        # Previous frame state for velocity
        self._prev_norm_landmarks: Optional[np.ndarray] = None
        self._prev_wrist: Optional[np.ndarray] = None

        # Minimum real frames before classification
        self.min_valid_frames = 15

        try:
            logger.info("Initializing GestureClassifier...")
            self._load_config()
            self._load_model()
            self._init_window()
            self.is_initialized = True
            logger.info(
                f"GestureClassifier initialized: "
                f"seq_len={self.seq_len}, feature_dim={self.feature_dim}, "
                f"classes={self.class_names}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize GestureClassifier: {e}")
            self.is_initialized = False

    def _load_config(self):
        config_path = config.TCN_CONFIG_PATH
        logger.info(f"Loading TCN config from {config_path}")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.seq_len = cfg["seq_len"]
        self.feature_dim = cfg["feature_dim"]
        self.raw_dim = cfg["raw_dim"]
        self.num_landmarks = cfg["num_landmarks"]
        self.class_names = cfg["class_names"]

        self.normalize_mean = np.array(cfg["normalize_mean"], dtype=np.float32)
        self.normalize_std = np.array(cfg["normalize_std"], dtype=np.float32)

        self.pairs = [tuple(p) for p in cfg["pairs"]]
        self.finger_chains = cfg["finger_chains"]

        logger.info(
            f"Config: {len(self.class_names)} classes, "
            f"{len(self.pairs)} pairs, {len(self.finger_chains)} chains"
        )

    def _load_model(self):
        import onnxruntime as ort

        model_path = config.TCN_MODEL_PATH
        logger.info(f"Loading ONNX model from {model_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, opts)

        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        logger.info(
            f"Model loaded: input={input_info.name} {input_info.shape}, "
            f"output={output_info.name} {output_info.shape}"
        )

    def _init_window(self):
        self.window.clear()
        self.is_real_frame.clear()

    def reset(self):
        """Reset sliding window and velocity state."""
        self._init_window()
        self._prev_norm_landmarks = None
        self._prev_wrist = None

    def add_frame_and_classify(
        self, landmarks: Optional[np.ndarray]
    ) -> Optional[ClassificationResult]:
        """
        Add a frame of landmarks and get classification result.

        Args:
            landmarks: Raw landmarks as ndarray of shape (63,) — 21 points x 3 coords.
                       Pass None if no hand detected.

        Returns:
            ClassificationResult or None if not enough frames yet.
        """
        if not self.is_initialized or self.session is None:
            return None

        features = self._compute_features(landmarks)

        # Sliding window
        if len(self.window) >= self.seq_len:
            self.window.popleft()
            self.is_real_frame.popleft()
        self.window.append(features)
        self.is_real_frame.append(
            landmarks is not None and len(landmarks) >= self.raw_dim
        )

        valid_frames = sum(self.is_real_frame)
        if valid_frames < self.min_valid_frames:
            return None

        return self._classify(valid_frames)

    def _compute_features(self, landmarks: Optional[np.ndarray]) -> np.ndarray:
        """Compute 144-dimensional feature vector from raw landmarks."""
        features = np.zeros(self.feature_dim, dtype=np.float32)

        if landmarks is None or len(landmarks) < self.raw_dim:
            return features

        # Reshape to (21, 3)
        lms = landmarks.reshape(self.num_landmarks, 3)

        wrist = lms[self.WRIST_IDX]
        mid_finger = lms[self.MID_FINGER_IDX]

        # Palm size
        palm_size = max(np.linalg.norm(mid_finger - wrist), 1e-6)

        # 1. Normalized landmarks (63 dims)
        norm_landmarks = ((lms - wrist) / palm_size).flatten()
        features[0:self.raw_dim] = norm_landmarks

        # 2. Velocity (63 dims)
        if self._prev_norm_landmarks is not None:
            velocity = norm_landmarks - self._prev_norm_landmarks
        else:
            velocity = np.zeros(self.raw_dim, dtype=np.float32)
        features[self.raw_dim : self.raw_dim * 2] = velocity

        # 3. Wrist velocity (3 dims)
        offset = self.raw_dim * 2
        if self._prev_wrist is not None:
            features[offset : offset + 3] = wrist - self._prev_wrist
        # else: zeros already

        # 4. Finger distances (10 dims)
        offset += 3
        for idx, (i, j) in enumerate(self.pairs):
            d = np.linalg.norm(
                norm_landmarks[i * 3 : i * 3 + 3]
                - norm_landmarks[j * 3 : j * 3 + 3]
            )
            features[offset + idx] = d

        # 5. Finger angles (5 dims)
        offset += len(self.pairs)
        for idx, chain in enumerate(self.finger_chains):
            if len(chain) < 3:
                continue
            v1 = lms[chain[1]] - lms[chain[0]]
            v2 = lms[chain[-1]] - lms[chain[1]]

            n1 = np.linalg.norm(v1) + 1e-8
            n2 = np.linalg.norm(v2) + 1e-8

            cos_angle = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
            features[offset + idx] = math.acos(cos_angle)

        # Save state for next frame
        self._prev_norm_landmarks = norm_landmarks.copy()
        self._prev_wrist = wrist.copy()

        return features

    def _classify(self, valid_frames: int) -> Optional[ClassificationResult]:
        """Run inference on the current window."""
        if self.session is None:
            return None

        try:
            # Build input: shape (1, feature_dim, seq_len)
            input_array = np.zeros(
                (1, self.feature_dim, self.seq_len), dtype=np.float32
            )

            window_list = list(self.window)
            num_frames = min(len(window_list), self.seq_len)

            for t in range(num_frames):
                frame = window_list[t]
                for d in range(min(len(frame), self.feature_dim)):
                    normalized = (frame[d] - self.normalize_mean[d]) / (
                        self.normalize_std[d] + 1e-8
                    )
                    input_array[0, d, t] = normalized

            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_array})

            logits = outputs[0].flatten()

            # Softmax for confidence
            max_logit = logits.max()
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / exp_logits.sum()

            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])
            gesture = self.class_names[max_idx]

            return ClassificationResult(
                gesture=gesture,
                confidence=confidence,
                class_index=max_idx,
                valid_frames=valid_frames,
            )

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None

    def close(self):
        """Release ONNX session."""
        self.session = None
        logger.info("GestureClassifier closed")
