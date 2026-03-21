import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from collections import deque, Counter
from pathlib import Path

import torch
import torch.nn as nn

HAND_LANDMARKER_MODEL = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

NUM_LANDMARKS = 21
NUM_COORDS = 3
RAW_DIM = NUM_LANDMARKS * NUM_COORDS
WRIST_IDX = 0
MID_FINGER_IDX = 9
FINGERTIP_IDS = [4, 8, 12, 16, 20]
BASE_IDS = [2, 5, 9, 13, 17]
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


def ensure_model_file(path=HAND_LANDMARKER_MODEL):
    if os.path.exists(path):
        return path
    import requests
    r = requests.get(MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return path


def safe_load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resample_sequence(seq, target_len):
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


class HandDetector:
    def __init__(self):
        self.api = None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision as mpv
            p = ensure_model_file()
            opts = mpv.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=p),
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.detector = mpv.HandLandmarker.create_from_options(opts)
            self.mp = mp
            self.api = "tasks"
            return
        except Exception:
            pass
        try:
            import mediapipe as mp
            self.detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp = mp
            self.api = "solutions"
        except Exception as e:
            raise RuntimeError(f"MediaPipe init failed: {e}")

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.api == "tasks":
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.detector.detect(img)
            if res.hand_landmarks:
                c = []
                for lm in res.hand_landmarks[0]:
                    c.extend([lm.x, lm.y, lm.z])
                return np.array(c, dtype=np.float32)
        else:
            res = self.detector.process(rgb)
            if res.multi_hand_landmarks:
                c = []
                for lm in res.multi_hand_landmarks[0].landmark:
                    c.extend([lm.x, lm.y, lm.z])
                return np.array(c, dtype=np.float32)
        return None

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()


def compute_features(raw_seq):
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
    def __init__(self, num_classes=8, feat_dim=FEATURE_DIM, dropout=0.15):
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


class GestureBuffer:
    def __init__(self, seq_len, window_seconds=1.0, source_fps=30.0, fill_frames=3):
        self.seq_len = int(seq_len)
        self.window_seconds = float(window_seconds)
        self.source_fps = float(source_fps) if source_fps is not None else 30.0
        if not np.isfinite(self.source_fps) or self.source_fps < 1.0 or self.source_fps > 240.0:
            self.source_fps = 30.0
        self.frame_dt = 1.0 / self.source_fps
        self.fill_frames = int(fill_frames)
        self.buf = deque()
        self.miss = 0
        self.last = None

    def _sanitize_lm(self, lm):
        arr = np.asarray(lm, dtype=np.float32).reshape(-1)
        if arr.size < RAW_DIM:
            out = np.zeros((RAW_DIM,), dtype=np.float32)
            out[:arr.size] = arr
            arr = out
        elif arr.size > RAW_DIM:
            arr = arr[:RAW_DIM]
        return arr.astype(np.float32)

    def _purge(self, current_t):
        cutoff = float(current_t) - self.window_seconds
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def push(self, lm, current_t):
        current_t = float(current_t)
        if lm is not None:
            arr = self._sanitize_lm(lm)
            self.miss = 0
            self.last = arr.copy()
            self.buf.append((current_t, arr.copy()))
        else:
            self.miss += 1
            if self.miss <= self.fill_frames and self.last is not None:
                self.buf.append((current_t, self.last.copy()))
        self._purge(current_t)

    def coverage_seconds(self, current_t=None):
        if current_t is not None:
            self._purge(current_t)
        if not self.buf:
            return 0.0
        return max(0.0, float(self.buf[-1][0] - self.buf[0][0]) + self.frame_dt)

    def ready(self, current_t=None):
        if current_t is not None:
            self._purge(current_t)
        return len(self.buf) >= 2 and self.coverage_seconds() >= self.window_seconds

    def get(self, current_t):
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


class GesturePredictor:
    def __init__(self, config_path, model_path, backend="torch", smooth_window=5,
                 window_seconds=1.0, source_fps=30.0, fill_frames=3, threshold=0.5):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.class_names = cfg["class_names"]
        self.seq_len = int(cfg["seq_len"])
        self.feat_dim = int(cfg["feature_dim"])
        self.mean = np.array(cfg["normalize_mean"], dtype=np.float32)
        self.std = np.array(cfg["normalize_std"], dtype=np.float32)
        self.backend = backend
        self.threshold = threshold
        self.model = self._load(model_path, len(self.class_names))
        self.buffer = GestureBuffer(
            self.seq_len,
            window_seconds=window_seconds,
            source_fps=source_fps,
            fill_frames=fill_frames
        )
        self.smooth_win = deque(maxlen=max(int(smooth_window), 1))
        self.prob_win = deque(maxlen=max(int(smooth_window), 1))

    def _load(self, path, nc):
        suffix = Path(path).suffix.lower()
        if self.backend == "onnx" or suffix == ".onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.backend = "onnx"
            return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        if self.backend == "torchscript" or suffix == ".pt":
            m = torch.jit.load(path, map_location="cpu")
            m.eval()
            self.backend = "torchscript"
            return m
        m = GestureTCN(nc, self.feat_dim)
        m.load_state_dict(safe_load_state_dict(path))
        m.eval()
        self.backend = "torch"
        return m

    def push(self, lm, current_t):
        self.buffer.push(lm, current_t)

    def push_missing(self, current_t):
        self.buffer.push(None, current_t)

    def predict(self, current_t=0.0):
        raw = self.buffer.get(current_t)
        if raw is None:
            self.smooth_win.clear()
            self.prob_win.clear()
            return None, 0.0
        feat = compute_features(raw)
        feat = (feat - self.mean) / (self.std + 1e-8)
        x = feat.T[np.newaxis].astype(np.float32)
        if self.backend == "onnx":
            logits = self.model.run(None, {"input": x})[0]
            logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        else:
            with torch.no_grad():
                logits = self.model(torch.from_numpy(x))
                logits = logits.detach().cpu().numpy().reshape(-1)
        logits = logits.astype(np.float32)
        probs = np.exp(logits - logits.max())
        probs_sum = float(probs.sum())
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            return None, 0.0
        probs /= probs_sum
        self.prob_win.append(probs.copy())
        avg = np.stack(list(self.prob_win), axis=0).mean(0)
        idx = int(avg.argmax())
        self.smooth_win.append(idx)
        si = Counter(self.smooth_win).most_common(1)[0][0]
        sc = float(avg[si])
        if sc < self.threshold:
            return "unknown", sc
        return self.class_names[si], sc

    def reset(self):
        self.buffer.reset()
        self.smooth_win.clear()
        self.prob_win.clear()


def draw_landmarks(frame, lm):
    if lm is None:
        return frame
    h, w = frame.shape[:2]
    pts = lm.reshape(21, 3)
    conns = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    sp = [(int(p[0] * w), int(p[1] * h)) for p in pts]
    for i, j in conns:
        cv2.line(frame, sp[i], sp[j], (0, 200, 255), 1, cv2.LINE_AA)
    for p in sp:
        cv2.circle(frame, p, 3, (255, 100, 0), -1, cv2.LINE_AA)
    return frame


def draw_overlay(frame, gesture, conf, fps, buf_len, covered_sec, window_seconds):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 104), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    ok = gesture and gesture not in ("unknown", None)
    color = (0, 230, 90) if ok else (120, 120, 120)
    if gesture is None:
        label = f"COLLECTING {covered_sec:.1f}/{window_seconds:.1f}s"
    else:
        label = gesture.upper()
    cv2.putText(frame, label, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2, cv2.LINE_AA)
    if gesture is not None:
        bm = w - 220
        bw = int(max(0.0, min(conf, 1.0)) * bm)
        cv2.rectangle(frame, (12, 58), (12 + bw, 76), color, -1)
        cv2.rectangle(frame, (12, 58), (12 + bm, 76), (160, 160, 160), 1)
        cv2.putText(frame, f"{conf:.2f}", (w - 195, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)
    fr = max(0.0, min(covered_sec / max(window_seconds, 1e-6), 1.0))
    fw = int(fr * 140)
    cv2.rectangle(frame, (12, 84), (152, 92), (70, 70, 70), -1)
    cv2.rectangle(frame, (12, 84), (12 + fw, 92), (0, 190, 255), -1)
    cv2.putText(
        frame,
        f"win:{covered_sec:.1f}/{window_seconds:.1f}s  n:{buf_len}",
        (160, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1
    )
    cv2.putText(frame, f"FPS {fps:.1f}", (w - 100, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    return frame


def resolve_source_fps(cap, src):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not np.isfinite(fps) or fps < 1.0 or fps > 240.0:
        fps = 30.0
    return float(fps)


def get_stream_time(cap, src, frame_idx, source_fps):
    if isinstance(src, str):
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts is not None and np.isfinite(ts) and ts > 0:
            return float(ts)
    return float(frame_idx) / max(float(source_fps), 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/config.json")
    parser.add_argument("--model", default="checkpoints/gesture_tcn_best.pth")
    parser.add_argument("--backend", default="torch", choices=["torch", "torchscript", "onnx"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--window_seconds", type=float, default=1.0)
    parser.add_argument("--fill_frames", type=int, default=3)
    parser.add_argument("--show_landmarks", action="store_true", default=True)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {src}")
        sys.exit(1)

    if not isinstance(src, str):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    source_fps = resolve_source_fps(cap, src)

    detector = HandDetector()
    predictor = GesturePredictor(
        args.config,
        args.model,
        args.backend,
        smooth_window=args.smooth_window,
        window_seconds=args.window_seconds,
        source_fps=source_fps,
        fill_frames=args.fill_frames,
        threshold=args.threshold
    )

    print(f"[INFO] Loaded ({predictor.backend}), classes={predictor.class_names}")
    print(f"[INFO] Using sliding window: last {args.window_seconds:.1f}s, source_fps={source_fps:.2f}")
    print("[INFO] Controls: q=quit, r=reset, l=landmarks toggle, p=pause")

    show_lm = args.show_landmarks
    paused = False
    prev = time.time()
    fps_ema = 30.0
    frame_idx = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("[INFO] Video ended")
                break

            # Flip for camera
            if not args.video:
                frame = cv2.flip(frame, 1)

            # Use real time for consistent behavior
            current_t = time.time()

            lm = detector.detect(frame)

            if lm is not None:
                predictor.push(lm, current_t)
                if show_lm:
                    frame = draw_landmarks(frame, lm)
            else:
                predictor.push_missing(current_t)

            gesture, conf = predictor.predict(current_t=current_t)

            now = time.time()
            fps_ema = 0.9 * fps_ema + 0.1 / max(now - prev, 1e-9)
            prev = now

            covered_sec = predictor.buffer.coverage_seconds(current_t)
            buf_len = len(predictor.buffer)

            frame = draw_overlay(
                frame,
                gesture,
                conf,
                fps_ema,
                buf_len,
                covered_sec,
                predictor.buffer.window_seconds
            )

        # Status bar at bottom
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 25), (w, h), (0, 0, 0), -1)
        status = "PAUSED" if paused else "RUNNING"
        cv2.putText(frame, f"[{status}] q=quit r=reset l=landmarks p=pause | Classes: {', '.join(predictor.class_names)}",
                   (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Gesture Recognition", frame)
        key = cv2.waitKey(1 if not paused else 100) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            predictor.reset()
            print("[INFO] Buffer reset")
        elif key == ord("l"):
            show_lm = not show_lm
            print(f"[INFO] Landmarks: {'ON' if show_lm else 'OFF'}")
        elif key == ord("p"):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
