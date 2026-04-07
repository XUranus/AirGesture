"""
Data augmentation functions for hand gesture recognition.
"""

import numpy as np

from .constants import NUM_LANDMARKS, NUM_COORDS, RAW_DIM, WRIST_IDX, SEQ_LEN
from .utils import to_raw_sequence, resample


def mirror_x(raw_seq):
    """
    Mirror landmarks along X axis.

    Args:
        raw_seq: Input sequence

    Returns:
        Mirrored sequence
    """
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for mirror_x")

    s = raw.copy().reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    s[:, :, 0] = 1.0 - s[:, :, 0]  # Flip X coordinate

    return s.reshape(-1, RAW_DIM).astype(np.float32)


def rotate_2d(raw_seq, angle_deg: float):
    """
    Rotate landmarks around wrist in 2D plane.

    Args:
        raw_seq: Input sequence
        angle_deg: Rotation angle in degrees

    Returns:
        Rotated sequence
    """
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for rotate_2d")

    s = raw.copy().reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    wrist = s[:, WRIST_IDX : WRIST_IDX + 1, :2].copy()

    # Translate to origin (wrist)
    s[:, :, :2] -= wrist

    # Rotation matrix
    a = np.radians(angle_deg)
    c, sn = np.cos(a), np.sin(a)
    x = s[:, :, 0].copy()
    y = s[:, :, 1].copy()
    s[:, :, 0] = c * x - sn * y
    s[:, :, 1] = sn * x + c * y

    # Translate back
    s[:, :, :2] += wrist

    return s.reshape(-1, RAW_DIM).astype(np.float32)


def scale_landmarks(raw_seq, factor: float):
    """
    Scale landmarks around wrist.

    Args:
        raw_seq: Input sequence
        factor: Scale factor (e.g., 1.1 for 10% larger)

    Returns:
        Scaled sequence
    """
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for scale_landmarks")

    s = raw.copy().reshape(-1, NUM_LANDMARKS, NUM_COORDS)
    wrist = s[:, WRIST_IDX : WRIST_IDX + 1, :].copy()

    # Scale relative to wrist
    s -= wrist
    s *= factor
    s += wrist

    return s.reshape(-1, RAW_DIM).astype(np.float32)


def add_jitter(raw_seq, sigma: float = 0.003):
    """
    Add Gaussian noise to landmarks.

    Args:
        raw_seq: Input sequence
        sigma: Standard deviation of noise

    Returns:
        Noisy sequence
    """
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for add_jitter")

    noise = np.random.randn(*raw.shape).astype(np.float32) * sigma
    return (raw + noise).astype(np.float32)


def time_warp(raw_seq):
    """
    Apply time warping augmentation.

    Randomly warps the time axis to simulate speed variations.

    Args:
        raw_seq: Input sequence

    Returns:
        Time-warped sequence
    """
    raw = to_raw_sequence(raw_seq)
    if raw is None:
        raise ValueError("Invalid raw sequence for time_warp")

    n = len(raw)
    if n < 4:
        return raw.copy()

    # Random warp parameters
    anchor = np.random.uniform(0.3, 0.7)
    warp = np.random.uniform(0.8, 1.2)

    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x_new = np.where(
        x < anchor,
        x * warp,
        anchor * warp + (x - anchor) * (1.0 - anchor * warp) / (1.0 - anchor + 1e-8),
    )
    x_new = np.clip(x_new, 0.0, 1.0)

    # Interpolate
    out = np.zeros_like(raw)
    x_target = np.linspace(0.0, 1.0, n, dtype=np.float32)
    for d in range(raw.shape[1]):
        out[:, d] = np.interp(x_target, x_new, raw[:, d]).astype(np.float32)

    return out.astype(np.float32)


def speed_change(raw_seq):
    """
    Change speed of sequence by resampling.

    Args:
        raw_seq: Input sequence

    Returns:
        Speed-changed sequence (resampled back to SEQ_LEN)
    """
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
