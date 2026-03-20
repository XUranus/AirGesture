import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

SEQ_LEN = 30
NUM_LANDMARKS = 21
NUM_COORDS = 3
RAW_DIM = NUM_LANDMARKS * NUM_COORDS
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 300
LR = 2e-3
WEIGHT_DECAY = 1e-3
PRUNE_AMOUNT = 0.20
MIN_VALID_RATIO = 0.10
PATIENCE = 40
CACHE_VERSION = "v6_raw"

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "grab", "swipe_up", "swipe_down", "swipe_left",
    "swipe_right", "finger_heart", "wave", "noise",
]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
SWIPE_CLASSES = {"swipe_up", "swipe_down", "swipe_left", "swipe_right"}
LR_SWAP_MAP = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}

HAND_LANDMARKER_MODEL = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ensure_model_file(path=HAND_LANDMARKER_MODEL):
    if os.path.exists(path):
        return path
    logger.info(f"Downloading {path} ...")
    import requests
    r = requests.get(MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return path


class HandDetector:
    def __init__(self, static_mode=True, max_hands=1, min_conf=0.5):
        self.api = None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision as mpv
            p = ensure_model_file()
            opts = mpv.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=p),
                num_hands=max_hands,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
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
                static_image_mode=static_mode,
                max_num_hands=max_hands,
                min_detection_confidence=min_conf,
                min_tracking_confidence=0.5,
            )
            self.mp = mp
            self.api = "solutions"
            return
        except Exception as e:
            logger.error(f"MediaPipe init failed: {e}")
            sys.exit(1)

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.api == "tasks":
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.detector.detect(img)
            if res.hand_landmarks and len(res.hand_landmarks) > 0:
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


def to_scalar(v, default=None):
    if v is None:
        return default
    if isinstance(v, np.ndarray):
        if v.shape == ():
            v = v.item()
        elif v.size == 1:
            v = v.reshape(()).item()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def resample(seq, target=SEQ_LEN):
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        return np.zeros((target, RAW_DIM), dtype=np.float32)
    n, d = seq.shape
    if n == 0:
        return np.zeros((target, d), dtype=np.float32)
    if n == target:
        return seq.copy()
    x_old = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target, dtype=np.float32)
    out = np.zeros((target, d), dtype=np.float32)
    for i in range(d):
        out[:, i] = np.interp(x_new, x_old, seq[:, i]).astype(np.float32)
    return out


def to_raw_sequence(seq, target_len=None):
    try:
        arr = np.asarray(seq, dtype=np.float32)
    except Exception:
        return None
    if arr.size == 0:
        t = 0 if target_len is None else target_len
        return np.zeros((t, RAW_DIM), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == NUM_LANDMARKS and arr.shape[2] == NUM_COORDS:
        arr = arr.reshape(arr.shape[0], RAW_DIM)
    elif arr.ndim == 2 and arr.shape == (NUM_LANDMARKS, NUM_COORDS):
        arr = arr.reshape(1, RAW_DIM)
    elif arr.ndim == 2 and arr.shape[1] == RAW_DIM:
        pass
    elif arr.ndim == 2 and arr.shape[0] == RAW_DIM and arr.shape[1] != RAW_DIM:
        arr = arr.T
        if arr.shape[1] != RAW_DIM:
            return None
    elif arr.ndim == 1 and arr.size % RAW_DIM == 0:
        arr = arr.reshape(-1, RAW_DIM)
    else:
        return None
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if target_len is not None and arr.shape[0] != target_len:
        arr = resample(arr, target_len)
    return arr


def interp_extrap_1d(valid_idx, valid_vals, n):
    xi = np.arange(n, dtype=np.float32)
    xp = np.asarray(valid_idx, dtype=np.float32)
    fp = np.asarray(valid_vals, dtype=np.float32)
    yi = np.interp(xi, xp, fp).astype(np.float32)
    if len(valid_idx) >= 2:
        left = xi < valid_idx[0]
        if left.any():
            dx = float(valid_idx[1] - valid_idx[0])
            slope = 0.0 if dx == 0 else float((fp[1] - fp[0]) / dx)
            yi[left] = fp[0] + (xi[left] - valid_idx[0]) * slope
        right = xi > valid_idx[-1]
        if right.any():
            dx = float(valid_idx[-1] - valid_idx[-2])
            slope = 0.0 if dx == 0 else float((fp[-1] - fp[-2]) / dx)
            yi[right] = fp[-1] + (xi[right] - valid_idx[-1]) * slope
    return yi.astype(np.float32)


def compute_features(raw_seq):
    raw_seq = to_raw_sequence(raw_seq)
    if raw_seq is None:
        shape = np.asarray(raw_seq).shape if raw_seq is not None else None
        raise ValueError(f"Invalid raw sequence shape: {shape}")
    T = raw_seq.shape[0]
    if T == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
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


def check_feature_dim():
    dummy = np.random.randn(SEQ_LEN, RAW_DIM).astype(np.float32)
    f = compute_features(dummy)
    actual = f.shape[1]
    if actual != FEATURE_DIM:
        logger.error(f"FEATURE_DIM mismatch: code says {FEATURE_DIM}, actual {actual}")
        sys.exit(1)
    logger.info(f"Feature dim verified: {actual}")


def interpolate_missing(lm_list):
    n = len(lm_list)
    if n == 0:
        return np.zeros((0, RAW_DIM), dtype=np.float32)
    valid_idx = [i for i, lm in enumerate(lm_list) if lm is not None]
    if len(valid_idx) == 0:
        return np.zeros((n, RAW_DIM), dtype=np.float32)
    result = np.zeros((n, RAW_DIM), dtype=np.float32)
    valid_arr = []
    for i in valid_idx:
        lm = np.asarray(lm_list[i], dtype=np.float32).reshape(-1)
        if lm.size < RAW_DIM:
            pad = np.zeros((RAW_DIM,), dtype=np.float32)
            pad[:lm.size] = lm
            lm = pad
        elif lm.size > RAW_DIM:
            lm = lm[:RAW_DIM]
        valid_arr.append(lm.astype(np.float32))
        result[i] = lm.astype(np.float32)
    valid_arr = np.stack(valid_arr, axis=0)
    if len(valid_idx) == 1:
        result[:] = valid_arr[0]
        return result
    for d in range(RAW_DIM):
        result[:, d] = interp_extrap_1d(valid_idx, valid_arr[:, d], n)
    return result.astype(np.float32)


def mirror_x(raw_seq):
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for mirror_x")
    s = raw.copy().reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    s[:, :, 0] = 1.0 - s[:, :, 0]
    return s.reshape(-1, RAW_DIM).astype(np.float32)


def rotate_2d(raw_seq, angle_deg):
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for rotate_2d")
    s = raw.copy().reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    wrist = s[:, WRIST_IDX:WRIST_IDX + 1, :2].copy()
    s[:, :, :2] -= wrist
    a = np.radians(angle_deg)
    c, sn = np.cos(a), np.sin(a)
    x = s[:, :, 0].copy()
    y = s[:, :, 1].copy()
    s[:, :, 0] = c * x - sn * y
    s[:, :, 1] = sn * x + c * y
    s[:, :, :2] += wrist
    return s.reshape(-1, RAW_DIM).astype(np.float32)


def scale_landmarks(raw_seq, factor):
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for scale_landmarks")
    s = raw.copy().reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    wrist = s[:, WRIST_IDX:WRIST_IDX + 1, :].copy()
    s -= wrist
    s *= factor
    s += wrist
    return s.reshape(-1, RAW_DIM).astype(np.float32)


def add_jitter(raw_seq, sigma=0.003):
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for add_jitter")
    noise = np.random.randn(*raw.shape).astype(np.float32) * sigma
    return (raw + noise).astype(np.float32)


def time_warp(raw_seq):
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for time_warp")
    n = len(raw)
    if n < 4:
        return raw.copy()
    anchor = np.random.uniform(0.3, 0.7)
    warp = np.random.uniform(0.8, 1.2)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x_new = np.where(
        x < anchor,
        x * warp,
        anchor * warp + (x - anchor) * (1.0 - anchor * warp) / (1.0 - anchor + 1e-8)
    )
    x_new = np.clip(x_new, 0.0, 1.0)
    out = np.zeros_like(raw)
    x_target = np.linspace(0.0, 1.0, n, dtype=np.float32)
    for d in range(raw.shape[1]):
        out[:, d] = np.interp(x_target, x_new, raw[:, d]).astype(np.float32)
    return out.astype(np.float32)


def speed_change(raw_seq):
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for speed_change")
    n = len(raw)
    factor = np.random.uniform(0.8, 1.2)
    new_n = max(int(n * factor), SEQ_LEN // 2)
    x_old = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, new_n, dtype=np.float32)
    out = np.zeros((new_n, raw.shape[1]), dtype=np.float32)
    for d in range(raw.shape[1]):
        out[:, d] = np.interp(x_new, x_old, raw[:, d]).astype(np.float32)
    return out.astype(np.float32)


def extract_video(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0
    lms = []
    total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        lms.append(detector.detect(frame))
    cap.release()
    return lms, total


def make_samples(lm_list, total_frames, class_name, is_train):
    valid = sum(1 for lm in lm_list if lm is not None)
    if total_frames == 0 or valid / total_frames < MIN_VALID_RATIO:
        return [], []
    raw = interpolate_missing(lm_list)
    n = len(raw)
    label = CLASS_TO_IDX[class_name]
    samples = []
    labels = []

    base = resample(raw, SEQ_LEN)
    samples.append(base.astype(np.float32))
    labels.append(label)

    if not is_train:
        return samples, labels

    if n > SEQ_LEN:
        for start_r in [0.0, 0.15, 0.25]:
            for end_r in [0.75, 0.85, 1.0]:
                s = int(n * start_r)
                e = int(n * end_r)
                if e - s >= SEQ_LEN // 2:
                    sub = resample(raw[s:e], SEQ_LEN)
                    samples.append(sub.astype(np.float32))
                    labels.append(label)

    for _ in range(3):
        aug = add_jitter(raw.copy())
        aug = resample(aug, SEQ_LEN)
        samples.append(aug.astype(np.float32))
        labels.append(label)

    for angle in [-15, -10, -5, 5, 10, 15]:
        rot = rotate_2d(raw.copy(), angle)
        rot = resample(rot, SEQ_LEN)
        samples.append(rot.astype(np.float32))
        labels.append(label)

    for sc in [0.85, 0.9, 1.1, 1.15]:
        scaled = scale_landmarks(raw.copy(), sc)
        scaled = resample(scaled, SEQ_LEN)
        samples.append(scaled.astype(np.float32))
        labels.append(label)

    for _ in range(2):
        tw = time_warp(raw.copy())
        tw = resample(tw, SEQ_LEN)
        samples.append(tw.astype(np.float32))
        labels.append(label)

    for _ in range(2):
        sp = speed_change(raw.copy())
        sp = resample(sp, SEQ_LEN)
        samples.append(sp.astype(np.float32))
        labels.append(label)

    if class_name in LR_SWAP_MAP:
        mir = mirror_x(raw.copy())
        mir = resample(mir, SEQ_LEN)
        swap_label = CLASS_TO_IDX[LR_SWAP_MAP[class_name]]
        samples.append(mir.astype(np.float32))
        labels.append(swap_label)
        for _ in range(2):
            aug_mir = add_jitter(mir.copy())
            samples.append(aug_mir.astype(np.float32))
            labels.append(swap_label)
    elif class_name not in SWIPE_CLASSES:
        mir = mirror_x(raw.copy())
        mir = resample(mir, SEQ_LEN)
        samples.append(mir.astype(np.float32))
        labels.append(label)

    if class_name not in SWIPE_CLASSES:
        rev = raw[::-1].copy()
        rev = resample(rev, SEQ_LEN)
        samples.append(rev.astype(np.float32))
        labels.append(label)

    return samples, labels


def save_cache(cache_path, samples, labels, is_train):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        if len(samples) == 0:
            sample_arr = np.zeros((0, SEQ_LEN, RAW_DIM), dtype=np.float32)
        else:
            sample_arr = np.stack([to_raw_sequence(s, target_len=SEQ_LEN) for s in samples], axis=0).astype(np.float32)
        label_arr = np.asarray(labels, dtype=np.int64)
        np.savez_compressed(
            cache_path,
            cache_version=np.array(CACHE_VERSION),
            sample_format=np.array("raw_landmarks"),
            raw_dim=np.array(RAW_DIM),
            seq_len=np.array(SEQ_LEN),
            is_train=np.array(int(is_train)),
            samples=sample_arr,
            labels=label_arr,
        )
        logger.info(f"Cache saved: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")


def try_load_cache(cache_path, is_train):
    if not cache_path or not os.path.exists(cache_path):
        return None
    logger.info(f"Loading cache: {cache_path}")
    try:
        with np.load(cache_path, allow_pickle=True) as c:
            version = to_scalar(c["cache_version"]) if "cache_version" in c.files else None
            sample_format = to_scalar(c["sample_format"]) if "sample_format" in c.files else None
            raw_dim = to_scalar(c["raw_dim"]) if "raw_dim" in c.files else None
            seq_len = to_scalar(c["seq_len"]) if "seq_len" in c.files else None
            cache_is_train = to_scalar(c["is_train"]) if "is_train" in c.files else None
            if (
                version != CACHE_VERSION
                or sample_format != "raw_landmarks"
                or raw_dim != RAW_DIM
                or seq_len != SEQ_LEN
                or cache_is_train != int(is_train)
            ):
                raise ValueError("cache metadata mismatch")
            samples = np.asarray(c["samples"], dtype=np.float32)
            labels = np.asarray(c["labels"], dtype=np.int64)
        if samples.ndim != 3 or samples.shape[1] != SEQ_LEN or samples.shape[2] != RAW_DIM:
            raise ValueError(f"cache sample shape mismatch: {samples.shape}")
        if len(samples) != len(labels):
            raise ValueError("cache sample/label count mismatch")
        out_samples = [np.ascontiguousarray(s, dtype=np.float32) for s in samples]
        out_labels = labels.astype(np.int64).tolist()
        logger.info(f"  {len(out_samples)} samples loaded")
        return out_samples, out_labels
    except Exception as e:
        logger.warning(f"Cache incompatible, rebuilding from videos: {e}")
        return None


def sanitize_dataset(samples, labels, name):
    fixed_samples = []
    fixed_labels = []
    if len(samples) != len(labels):
        logger.warning(f"{name}: sample/label count mismatch, using min length")
    for s, l in zip(samples, labels):
        raw = to_raw_sequence(s, target_len=SEQ_LEN)
        if raw is None:
            continue
        fixed_samples.append(raw.astype(np.float32))
        fixed_labels.append(int(l))
    skipped = len(list(zip(samples, labels))) - len(fixed_samples)
    if skipped > 0:
        logger.warning(f"{name}: skipped {skipped} invalid samples")
    return fixed_samples, fixed_labels


class GestureDataset(Dataset):
    def __init__(self, raw_samples, labels, norm_stats=None, augment=False):
        self.raw_samples = raw_samples
        self.labels = labels
        self.norm_stats = norm_stats
        self.augment = augment

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        raw = self.raw_samples[idx].copy()
        label = self.labels[idx]
        if self.augment and random.random() < 0.5:
            raw = add_jitter(raw, sigma=0.002)
        if self.augment and random.random() < 0.3:
            raw = time_warp(raw)
            raw = resample(raw, SEQ_LEN)
        feat = compute_features(raw)
        if self.norm_stats is not None:
            feat = (feat - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-8)
        x = torch.FloatTensor(feat.T)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


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
    def __init__(self, num_classes=NUM_CLASSES, feat_dim=FEATURE_DIM, dropout=0.15):
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
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def load_dataset(data_dir, detector, cache_path=None, is_train=True):
    cached = try_load_cache(cache_path, is_train)
    if cached is not None:
        return cached

    all_samples, all_labels = [], []
    counts = defaultdict(int)
    dp = Path(data_dir)
    for cn in CLASS_NAMES:
        cd = dp / cn
        if not cd.exists():
            logger.warning(f"Not found: {cd}")
            continue
        vfs = sorted(
            f for f in cd.iterdir()
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm")
        )
        logger.info(f"  [{cn}]: {len(vfs)} videos")
        for vf in tqdm(vfs, desc=f"  {cn}", leave=False):
            lm_list, total = extract_video(vf, detector)
            if total == 0:
                continue
            samps, labs = make_samples(lm_list, total, cn, is_train)
            for s, l in zip(samps, labs):
                raw = to_raw_sequence(s, target_len=SEQ_LEN)
                if raw is None:
                    continue
                all_samples.append(raw.astype(np.float32))
                all_labels.append(int(l))
                counts[CLASS_NAMES[l]] += 1

    logger.info("Dataset statistics:")
    for cn in CLASS_NAMES:
        logger.info(f"  {cn:>15s}: {counts[cn]:>5d} samples")
    logger.info(f"  {'Total':>15s}: {len(all_samples):>5d} samples")

    if cache_path:
        save_cache(cache_path, all_samples, all_labels, is_train)

    return all_samples, all_labels


def compute_norm_stats(samples):
    sum_feat = None
    sum_sq_feat = None
    total_frames = 0
    bad = 0

    for s in tqdm(samples, desc="  norm", leave=False):
        raw = to_raw_sequence(s, target_len=SEQ_LEN)
        if raw is None:
            bad += 1
            continue
        feat = compute_features(raw).astype(np.float64)
        cur_sum = feat.sum(axis=0)
        cur_sq = np.square(feat).sum(axis=0)
        if sum_feat is None:
            sum_feat = cur_sum
            sum_sq_feat = cur_sq
        else:
            sum_feat += cur_sum
            sum_sq_feat += cur_sq
        total_frames += feat.shape[0]

    if total_frames == 0:
        raise RuntimeError("No valid samples for normalization")

    mean64 = sum_feat / total_frames
    var64 = np.maximum(sum_sq_feat / total_frames - np.square(mean64), 1e-12)
    mean = mean64.astype(np.float32)
    std = np.sqrt(var64).astype(np.float32)

    if bad > 0:
        logger.warning(f"Skipped {bad} invalid samples while computing normalization stats")

    return {"mean": mean, "std": std}


def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = w / w.sum() * NUM_CLASSES
    return torch.FloatTensor(w)


def make_sampler(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts = np.maximum(counts, 1.0)
    sw = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    if loader is None:
        return 0.0, 0.0, [], []
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        loss = criterion(logits, by)
        total_loss += loss.item() * bx.size(0)
        p = logits.argmax(1)
        correct += (p == by).sum().item()
        total += bx.size(0)
        preds_all.extend(p.cpu().numpy())
        labels_all.extend(by.cpu().numpy())
    if total == 0:
        return 0.0, 0.0, preds_all, labels_all
    return total_loss / total, correct / total, preds_all, labels_all


def load_state_dict_file(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def train_model(train_samples, train_labels, test_samples, test_labels, norm_stats, class_weights, args):
    sampler = make_sampler(train_labels)
    tr_ds = GestureDataset(train_samples, train_labels, norm_stats, augment=True)
    te_ds = GestureDataset(test_samples, test_labels, norm_stats, augment=False)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    te_loader = DataLoader(
        te_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = GestureTCN(NUM_CLASSES, FEATURE_DIM, dropout=args.dropout).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params:,}")

    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = -1.0
    best_epoch = 0
    patience_ctr = 0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "gesture_tcn_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            t_loss += loss.item() * bx.size(0)
            t_correct += (logits.argmax(1) == by).sum().item()
            t_total += bx.size(0)

        scheduler.step()

        tr_loss = t_loss / max(t_total, 1)
        tr_acc = t_correct / max(t_total, 1)
        te_loss, te_acc, _, _ = evaluate(model, te_loader, criterion, DEVICE)
        lr_now = optimizer.param_groups[0]["lr"]

        tag = ""
        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            tag = "  <- best"
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or tag:
            logger.info(
                f"Epoch {epoch:>3d}/{args.epochs} | "
                f"TrL:{tr_loss:.4f} TrA:{tr_acc:.4f} | "
                f"TeL:{te_loss:.4f} TeA:{te_acc:.4f} | "
                f"LR:{lr_now:.2e}{tag}"
            )

        if patience_ctr >= PATIENCE:
            logger.info(f"Early stop at epoch {epoch}")
            break

    logger.info(f"Best test acc: {max(best_acc, 0.0):.4f} @ epoch {best_epoch}")
    model.load_state_dict(load_state_dict_file(ckpt_path, DEVICE))
    return model, te_loader, max(best_acc, 0.0)


def apply_pruning(model, amount):
    logger.info(f"Pruning {amount:.0%} ...")
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            prune.l1_unstructured(m, "weight", amount=amount)
    return model


def remove_pruning(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            try:
                prune.remove(m, "weight")
            except ValueError:
                pass


def sparsity_report(model):
    tp, tz = 0, 0
    for name, p in model.named_parameters():
        if "weight" in name:
            n = p.numel()
            z = (p == 0).sum().item()
            tp += n
            tz += z
            logger.info(f"  {name:>45s} | {n:>7d} | {z / max(n, 1):>5.1%}")
    logger.info(f"  {'Total':>45s} | {tp:>7d} | {tz / max(tp, 1):>5.1%}")


def export_torchscript(model, save_dir):
    m = deepcopy(model).cpu().eval()
    d = torch.randn(1, FEATURE_DIM, SEQ_LEN)
    p = os.path.join(save_dir, "gesture_tcn.pt")
    try:
        t = torch.jit.trace(m, d)
        t.save(p)
        logger.info(f"TorchScript: {p}")
    except Exception as e:
        logger.warning(f"TorchScript failed: {e}")


def export_onnx(model, save_dir):
    m = deepcopy(model).cpu().eval()
    d = torch.randn(1, FEATURE_DIM, SEQ_LEN)
    p = os.path.join(save_dir, "gesture_tcn.onnx")
    try:
        torch.onnx.export(
            m,
            d,
            p,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        logger.info(f"ONNX: {p}")
    except Exception as e:
        logger.warning(f"ONNX failed: {e}")


def export_int8(model, save_dir):
    try:
        from torch.quantization import quantize_dynamic
        m = deepcopy(model).cpu().eval()
        q = quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        p = os.path.join(save_dir, "gesture_tcn_int8.pt")
        torch.save(q.state_dict(), p)
        logger.info(f"INT8: {p}")
    except Exception as e:
        logger.warning(f"INT8 failed: {e}")


def save_all(model, norm_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pp = os.path.join(save_dir, "gesture_tcn_pruned.pth")
    torch.save(model.state_dict(), pp)
    logger.info(f"Weights: {pp}")
    cfg = {
        "cache_version": CACHE_VERSION,
        "class_names": CLASS_NAMES,
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "raw_dim": RAW_DIM,
        "num_classes": NUM_CLASSES,
        "num_landmarks": NUM_LANDMARKS,
        "normalize_mean": norm_stats["mean"].tolist(),
        "normalize_std": norm_stats["std"].tolist(),
        "pairs": PAIRS,
        "fingertip_ids": FINGERTIP_IDS,
        "base_ids": BASE_IDS,
        "n_fingers": N_FINGERS,
        "finger_chains": FINGER_CHAINS,
    }
    cp = os.path.join(save_dir, "config.json")
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    logger.info(f"Config: {cp}")
    export_torchscript(model, save_dir)
    export_onnx(model, save_dir)
    export_int8(model, save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save_dir", type=str, default="checkpoints_v4")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--prune_amount", type=float, default=PRUNE_AMOUNT)
    parser.add_argument("--dropout", type=float, default=0.15)
    args = parser.parse_args()

    check_feature_dim()

    logger.info("=" * 60)
    logger.info("  Gesture Recognition TCN v6")
    logger.info("=" * 60)
    logger.info(f"  Device      : {DEVICE}")
    logger.info(f"  Feature dim : {FEATURE_DIM}")
    logger.info(f"  Seq len     : {SEQ_LEN}")
    logger.info(f"  Classes     : {NUM_CLASSES} -> {CLASS_NAMES}")
    logger.info("=" * 60)

    detector = HandDetector(static_mode=True, max_hands=1, min_conf=0.5)

    train_dir = os.path.join(args.data_dir, "Train")
    test_dir = os.path.join(args.data_dir, "Test")
    cache_dir = os.path.join(args.save_dir, "cache")
    tr_cache = None if args.no_cache else os.path.join(cache_dir, f"train_{CACHE_VERSION}.npz")
    te_cache = None if args.no_cache else os.path.join(cache_dir, f"test_{CACHE_VERSION}.npz")

    logger.info("[Phase 1] Extracting landmarks ...")
    train_samples, train_labels = load_dataset(train_dir, detector, tr_cache, is_train=True)
    test_samples, test_labels = load_dataset(test_dir, detector, te_cache, is_train=False)
    detector.close()

    train_samples, train_labels = sanitize_dataset(train_samples, train_labels, "train")
    test_samples, test_labels = sanitize_dataset(test_samples, test_labels, "test")

    if len(train_samples) == 0:
        logger.error("No training data!")
        sys.exit(1)

    logger.info(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

    logger.info("[Phase 2] Computing normalization stats ...")
    norm_stats = compute_norm_stats(train_samples)
    logger.info(f"  Mean range: [{norm_stats['mean'].min():.4f}, {norm_stats['mean'].max():.4f}]")
    logger.info(f"  Std range:  [{norm_stats['std'].min():.4f}, {norm_stats['std'].max():.4f}]")

    logger.info("[Phase 3] Class weights ...")
    cw = compute_class_weights(train_labels)
    logger.info(f"  Weights: {cw.numpy().round(3)}")

    logger.info("[Phase 4] Training ...")
    model, te_loader, best_acc = train_model(
        train_samples, train_labels, test_samples, test_labels,
        norm_stats, cw, args
    )

    logger.info("[Phase 5] Evaluation ...")
    crit = nn.CrossEntropyLoss()
    _, te_acc, preds, gts = evaluate(model, te_loader, crit, DEVICE)
    logger.info(f"Accuracy: {te_acc:.4f}")

    label_ids = list(range(NUM_CLASSES))
    if len(gts) > 0:
        logger.info(
            "Report:\n" + classification_report(
                gts,
                preds,
                labels=label_ids,
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )
        cm = confusion_matrix(gts, preds, labels=label_ids)
        logger.info("Confusion matrix:")
        hdr = "          " + "  ".join(f"{n[:6]:>6s}" for n in CLASS_NAMES)
        logger.info(hdr)
        for i, row in enumerate(cm):
            logger.info(f"  {CLASS_NAMES[i][:8]:>8s}  " + "  ".join(f"{v:>6d}" for v in row))
    else:
        logger.warning("No test samples available, skipping detailed evaluation.")

    logger.info("[Phase 6] Pruning ...")
    model = apply_pruning(model, args.prune_amount)
    _, pa, _, _ = evaluate(model, te_loader, crit, DEVICE)
    logger.info(f"Post-pruning acc: {pa:.4f}")
    remove_pruning(model)
    sparsity_report(model)

    logger.info("[Phase 7] Saving ...")
    save_all(model, norm_stats, args.save_dir)

    logger.info("=" * 60)
    logger.info(f"  Done. Best={best_acc:.4f}  Pruned={pa:.4f}")
    logger.info(f"  Output -> {args.save_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
