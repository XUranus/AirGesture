"""
Feature computation for hand gesture recognition.
"""

import numpy as np

from .constants import (
    SEQ_LEN,
    NUM_LANDMARKS,
    NUM_COORDS,
    RAW_DIM,
    WRIST_IDX,
    MID_FINGER_IDX,
    PAIRS,
    N_PAIRS,
    FINGER_CHAINS,
    N_FINGERS,
    FEATURE_DIM,
)
from .utils import to_raw_sequence


def compute_features(raw_seq):
    """
    Compute feature vector from raw landmark sequence.

    Features include:
    - Normalized landmarks (63 dims): relative to wrist, scaled by palm size
    - Velocity (63 dims): frame-to-frame velocity of normalized landmarks
    - Wrist velocity (3 dims): frame-to-frame wrist movement
    - Finger distances (10 dims): distances between key landmark pairs
    - Finger angles (5 dims): bend angles for each finger

    Total: 144 dims

    Args:
        raw_seq: Raw landmark sequence, shape (T, RAW_DIM) or compatible format

    Returns:
        Feature array of shape (T, FEATURE_DIM)
    """
    raw_seq = to_raw_sequence(raw_seq)
    if raw_seq is None:
        shape = np.asarray(raw_seq).shape if raw_seq is not None else None
        raise ValueError(f"Invalid raw sequence shape: {shape}")

    T = raw_seq.shape[0]
    if T == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    # Reshape to (T, 21, 3)
    lms = raw_seq.reshape(T, NUM_LANDMARKS, NUM_COORDS)

    # Get wrist position for each frame
    wrist = lms[:, WRIST_IDX, :]

    # Relative position to wrist
    relative = lms - wrist[:, np.newaxis, :]

    # Get middle finger tip for palm size normalization
    mid = lms[:, MID_FINGER_IDX, :]

    # Palm size (distance from wrist to middle finger tip)
    palm_size = np.maximum(np.linalg.norm(mid - wrist, axis=-1, keepdims=True), 1e-6)

    # Normalized landmarks (relative to wrist, scaled by palm size)
    norm_lms = relative / palm_size[:, np.newaxis, :]
    norm_flat = norm_lms.reshape(T, -1).astype(np.float32)

    # Velocity of normalized landmarks
    vel = np.zeros_like(norm_flat)
    if T > 1:
        vel[1:] = norm_flat[1:] - norm_flat[:-1]
        vel[0] = vel[1]

    # Wrist velocity
    wrist_vel = np.zeros((T, 3), dtype=np.float32)
    if T > 1:
        wrist_vel[1:] = wrist[1:] - wrist[:-1]
        wrist_vel[0] = wrist_vel[1]

    # Distances between key landmark pairs
    dists = np.zeros((T, N_PAIRS), dtype=np.float32)
    for k, (i, j) in enumerate(PAIRS):
        dists[:, k] = np.linalg.norm(norm_lms[:, i] - norm_lms[:, j], axis=-1)

    # Finger bend angles
    angles = np.zeros((T, N_FINGERS), dtype=np.float32)
    for fi, chain in enumerate(FINGER_CHAINS):
        # Vector from base to middle joint
        v1 = lms[:, chain[1]] - lms[:, chain[0]]
        # Vector from middle joint to tip
        v2 = lms[:, chain[-1]] - lms[:, chain[1]]
        # Compute angle
        n1 = np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8
        n2 = np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8
        cos_a = np.clip((v1 / n1 * v2 / n2).sum(-1), -1.0, 1.0)
        angles[:, fi] = np.arccos(cos_a)

    # Concatenate all features
    feat = np.concatenate([norm_flat, vel, wrist_vel, dists, angles], axis=1)

    return feat.astype(np.float32)
