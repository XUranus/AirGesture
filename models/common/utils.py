import os
import sys

from .constants import *

import numpy as np

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
    """Resample sequence to target length using linear interpolation."""
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
    """Convert various formats to raw sequence (T, RAW_DIM)."""
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



def compute_features(raw_seq):
    """Compute feature vector from raw landmark sequence.

    Features include:
    - Normalized landmarks (63 dims)
    - Velocity (63 dims)
    - Wrist velocity (3 dims)
    - Finger distances (10 dims)
    - Finger angles (5 dims)

    Total: 144 dims
    """
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