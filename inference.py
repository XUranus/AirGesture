#!/usr/bin/env python3
"""
inference.py - Gesture Recognition Real-time Inference

Loads a pruned 1D-CNN model, detects hand landmarks via MediaPipe in
real time, maintains a 30-frame sliding-window buffer and classifies
the current gesture.

Compatible with both the new MediaPipe Tasks API and the legacy Solutions API.

Usage:
  python inference.py                          # webcam
  python inference.py --video path/to/video    # video file
  python inference.py --eval --data_dir data   # batch evaluation on Test/
"""

import os
import sys
import json
import time
import argparse
import collections

import cv2
import numpy as np

import torch
import torch.nn as nn

# MediaPipe Hand Landmarker model file
HAND_LANDMARKER_MODEL = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Hand skeleton connectivity for manual drawing (Tasks API)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class GestureCNN1D(nn.Module):
    """1D-CNN temporal classifier (must match the architecture in train.py)."""

    def __init__(self, feature_dim=63, num_classes=8, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), nn.Dropout(dropout),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def ensure_model_file(model_path=HAND_LANDMARKER_MODEL):
    """Download the MediaPipe hand landmarker model if missing."""
    if os.path.exists(model_path):
        return model_path
    print(f"[INFO] Downloading {model_path}...")
    try:
        import requests
        resp = requests.get(MODEL_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[INFO] Download complete: {model_path}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print(f"[ERROR] Please download manually: {MODEL_URL}")
        sys.exit(1)
    return model_path


class HandDetector:
    """Hand landmark detector that auto-selects between the new MediaPipe
    Tasks API and the legacy Solutions API."""

    def __init__(self, static_mode=False, max_hands=1, min_detection_conf=0.6):
        import mediapipe as mp
        self.mp = mp
        self.api_version = None
        self.detector = None
        self.mp_drawing = None
        self.mp_hands_module = None

        # Try the new Tasks API
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = ensure_model_file(HAND_LANDMARKER_MODEL)
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=min_detection_conf,
                min_hand_presence_confidence=min_detection_conf,
                min_tracking_confidence=0.5,
            )
            self.detector = mp_vision.HandLandmarker.create_from_options(options)
            self.api_version = "tasks"
            print("[INFO] Using MediaPipe Tasks API (new)")
            return
        except (ImportError, AttributeError, Exception):
            pass

        # Fall back to legacy Solutions API
        try:
            self.detector = mp.solutions.hands.Hands(
                static_image_mode=static_mode,
                max_num_hands=max_hands,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=0.5,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands_module = mp.solutions.hands
            self.api_version = "solutions"
            print("[INFO] Using MediaPipe Solutions API (legacy)")
            return
        except (ImportError, AttributeError, Exception):
            pass

        print("[ERROR] Could not initialise MediaPipe. Try: pip install mediapipe --upgrade")
        sys.exit(1)

    def detect(self, frame_bgr):
        """Detect hand landmarks in a BGR frame.
        Returns (landmarks_63d | None, raw_landmarks_for_drawing | None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.api_version == "tasks":
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand = result.hand_landmarks[0]
                coords = []
                for lm in hand:
                    coords.extend([lm.x, lm.y, lm.z])
                return np.array(coords, dtype=np.float32), hand
            return None, None

        results = self.detector.process(rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32), hand
        return None, None

    def draw_landmarks(self, frame, raw_landmarks):
        """Render hand skeleton on the frame."""
        if raw_landmarks is None:
            return frame
        h, w = frame.shape[:2]

        if self.api_version == "tasks":
            points = []
            for lm in raw_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
            for s, e in HAND_CONNECTIONS:
                if s < len(points) and e < len(points):
                    cv2.line(frame, points[s], points[e], (255, 255, 255), 2)
        else:
            if self.mp_drawing and self.mp_hands_module:
                self.mp_drawing.draw_landmarks(
                    frame, raw_landmarks,
                    self.mp_hands_module.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )
        return frame

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()


def load_config(save_dir):
    """Load the JSON config produced by train.py."""
    path = os.path.join(save_dir, "config.json")
    if not os.path.exists(path):
        print(f"[ERROR] Config not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["normalize_mean"] = np.array(cfg["normalize_mean"], dtype=np.float32)
    cfg["normalize_std"] = np.array(cfg["normalize_std"], dtype=np.float32)
    return cfg


def load_model(save_dir, config, device):
    """Load the trained (and optionally pruned) model weights."""
    model = GestureCNN1D(
        feature_dim=config["feature_dim"],
        num_classes=config["num_classes"],
        dropout=0.0,
    ).to(device)

    model_path = os.path.join(save_dir, "gesture_cnn1d_pruned.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, "gesture_cnn1d_best.pth")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Report classifier sparsity
    tp, zp = 0, 0
    for name, param in model.named_parameters():
        if "classifier" in name and "weight" in name:
            tp += param.nelement()
            zp += (param == 0).sum().item()
    if tp > 0:
        print(f"[INFO] Classifier sparsity: {zp}/{tp} ({zp / tp:.1%})")
    print(f"[INFO] Model loaded: {model_path}")
    return model


class FrameBuffer:
    """Fixed-length FIFO buffer for the sliding-window input."""

    def __init__(self, seq_len=30, feature_dim=63):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.buffer = collections.deque(maxlen=seq_len)
        self.last_valid = np.zeros(feature_dim, dtype=np.float32)

    def push(self, landmarks):
        if landmarks is not None:
            self.last_valid = landmarks.copy()
            self.buffer.append(landmarks.copy())
        else:
            # Repeat last valid frame when no hand is detected
            self.buffer.append(self.last_valid.copy())

    def is_ready(self):
        return len(self.buffer) >= self.seq_len

    def get_sequence(self):
        if len(self.buffer) < self.seq_len:
            pad = self.seq_len - len(self.buffer)
            frames = [np.zeros(self.feature_dim, dtype=np.float32)] * pad + list(self.buffer)
        else:
            frames = list(self.buffer)
        return np.array(frames, dtype=np.float32)

    def clear(self):
        self.buffer.clear()
        self.last_valid = np.zeros(self.feature_dim, dtype=np.float32)


class GestureRecognizer:
    """Wraps model inference with normalisation and temporal smoothing."""

    def __init__(self, model, config, device, confidence_threshold=0.55, smoothing_window=5):
        self.model = model
        self.device = device
        self.threshold = confidence_threshold
        self.class_names = config["class_names"]
        self.mean = config["normalize_mean"]
        self.std = config["normalize_std"]
        self.history = collections.deque(maxlen=smoothing_window)

    def predict(self, sequence):
        """Classify a (seq_len, feature_dim) array.
        Returns (class_name, confidence, probability_vector)."""
        normed = (sequence - self.mean) / (self.std + 1e-8)
        tensor = torch.FloatTensor(normed.T).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        self.history.append(pred_idx)

        # Majority vote over recent predictions
        if len(self.history) >= 3:
            votes = np.bincount(list(self.history), minlength=len(self.class_names))
            smoothed_idx = int(np.argmax(votes))
        else:
            smoothed_idx = pred_idx

        if confidence < self.threshold:
            return "uncertain", confidence, probs

        return self.class_names[smoothed_idx], confidence, probs


def draw_info_panel(frame, gesture, confidence, probs, class_names, fps, buf_ready):
    """Render a semi-transparent info overlay on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 270), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "Gesture Recognition", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    colour = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
    cv2.putText(frame, f"Result: {gesture}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(frame, f"Conf: {confidence:.2%}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    status_text = "Ready" if buf_ready else "Buffering..."
    status_col = (0, 255, 0) if buf_ready else (0, 165, 255)
    cv2.putText(frame, f"Buffer: {status_text}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 1)

    # Per-class probability bars
    if probs is not None:
        best_i = int(np.argmax(probs))
        y = 180
        for i, (cn, p) in enumerate(zip(class_names, probs)):
            bw = int(p * 150)
            bc = (0, 200, 0) if i == best_i else (100, 100, 100)
            cv2.rectangle(frame, (130, y - 8), (130 + bw, y + 2), bc, -1)
            cv2.putText(frame, cn[:12], (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)
            cv2.putText(frame, f"{p:.0%}", (285, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)
            y += 14

    return frame


def run_inference(source, save_dir, device):
    """Main real-time inference loop."""
    config = load_config(save_dir)
    model = load_model(save_dir, config, device)
    recognizer = GestureRecognizer(model, config, device)

    hand_detector = HandDetector(static_mode=False, max_hands=1, min_detection_conf=0.6)
    buffer = FrameBuffer(seq_len=config["seq_len"], feature_dim=config["feature_dim"])

    # Open video source
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        src = int(source) if isinstance(source, str) else source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        is_camera = True
    else:
        cap = cv2.VideoCapture(source)
        is_camera = False

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    predict_interval = 3
    frame_count = 0
    fps = 0.0
    prev_time = time.time()

    cur_gesture = "waiting..."
    cur_conf = 0.0
    cur_probs = None

    print("\n[INFO] Real-time gesture recognition started (press 'q' to quit, 'c' to clear buffer)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_camera:
                continue
            break

        frame_count += 1
        if is_camera:
            frame = cv2.flip(frame, 1)

        landmarks, raw_hand = hand_detector.detect(frame)
        hand_detected = landmarks is not None

        if raw_hand is not None:
            frame = hand_detector.draw_landmarks(frame, raw_hand)

        buffer.push(landmarks)

        if frame_count % predict_interval == 0 and buffer.is_ready():
            cur_gesture, cur_conf, cur_probs = recognizer.predict(buffer.get_sequence())

        # FPS calculation
        now = time.time()
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now

        frame = draw_info_panel(
            frame, cur_gesture, cur_conf, cur_probs,
            config["class_names"], fps, buffer.is_ready(),
        )

        # Hand detection indicator
        ic = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 12, ic, -1)
        cv2.putText(
            frame, "Hand" if hand_detected else "No Hand",
            (frame.shape[1] - 110, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ic, 1,
        )

        cv2.imshow("Gesture Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            buffer.clear()
            cur_gesture, cur_conf, cur_probs = "waiting...", 0.0, None
            print("[INFO] Buffer cleared")

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    print("[INFO] Inference finished")


def run_batch_evaluation(data_dir, save_dir, device):
    """Evaluate the model on every video in the Test/ directory."""
    from scipy.interpolate import interp1d

    config = load_config(save_dir)
    model = load_model(save_dir, config, device)
    recognizer = GestureRecognizer(model, config, device, confidence_threshold=0.0)

    seq_len = config["seq_len"]
    feature_dim = config["feature_dim"]
    class_names = config["class_names"]

    detector = HandDetector(static_mode=True, max_hands=1, min_detection_conf=0.5)
    test_dir = os.path.join(data_dir, "Test")
    correct, total = 0, 0

    print(f"\n[EVAL] Test directory: {test_dir}")

    for cls_name in class_names:
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        label_idx = class_names.index(cls_name)
        videos = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        )

        cls_correct = 0
        for vf in videos:
            cap = cv2.VideoCapture(os.path.join(cls_dir, vf))
            lm_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                lm, _ = detector.detect(frame)
                lm_list.append(lm)
            cap.release()

            if not lm_list:
                continue

            n = len(lm_list)
            valid_ids = [i for i, x in enumerate(lm_list) if x is not None]
            if not valid_ids:
                continue

            arr = np.zeros((n, feature_dim), dtype=np.float32)
            for i in valid_ids:
                arr[i] = lm_list[i]

            if len(valid_ids) == 1:
                arr[:] = arr[valid_ids[0]]
            elif len(valid_ids) > 1:
                va = np.array([lm_list[i] for i in valid_ids])
                for d in range(feature_dim):
                    fn = interp1d(valid_ids, va[:, d], kind="linear", fill_value="extrapolate")
                    arr[:, d] = fn(np.arange(n))

            # Resample to seq_len
            x_old = np.linspace(0, 1, n)
            x_new = np.linspace(0, 1, seq_len)
            seq = np.zeros((seq_len, feature_dim), dtype=np.float32)
            for d in range(feature_dim):
                seq[:, d] = np.interp(x_new, x_old, arr[:, d])

            pred_name, _, _ = recognizer.predict(seq)
            pred_idx = class_names.index(pred_name) if pred_name in class_names else -1

            total += 1
            if pred_idx == label_idx:
                correct += 1
                cls_correct += 1

        print(f"  {cls_name:>15s}: {cls_correct}/{len(videos)} correct")

    if total > 0:
        print(f"\nOverall accuracy: {correct}/{total} ({correct / total:.2%})")

    detector.close()


def main():
    parser = argparse.ArgumentParser(description="Gesture Recognition Inference")
    parser.add_argument("--video", type=str, default="0", help="0 for webcam or path to video file")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--eval", action="store_true", help="Batch evaluation mode on Test/")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if args.eval:
        run_batch_evaluation(args.data_dir, args.save_dir, device)
    else:
        run_inference(args.video, args.save_dir, device)


if __name__ == "__main__":
    main()
