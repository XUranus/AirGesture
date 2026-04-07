"""
Utility functions for sequence processing.
"""

import numpy as np

from .constants import SEQ_LEN, NUM_LANDMARKS, NUM_COORDS, RAW_DIM


def to_scalar(v, default=None):
    """
    Convert a value to scalar.

    Args:
        v: Input value (can be numpy array, bytes, or scalar)
        default: Default value if input is None

    Returns:
        Scalar value
    """
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


def resample(seq, target: int = SEQ_LEN):
    """
    Resample sequence to target length using linear interpolation.

    Args:
        seq: Input sequence of shape (n, d)
        target: Target sequence length

    Returns:
        Resampled sequence of shape (target, d)
    """
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


def to_raw_sequence(seq, target_len: int = None):
    """
    Convert various formats to raw sequence (T, RAW_DIM).

    Handles multiple input formats:
    - (T, NUM_LANDMARKS, NUM_COORDS) -> (T, RAW_DIM)
    - (NUM_LANDMARKS, NUM_COORDS) -> (1, RAW_DIM)
    - (T, RAW_DIM) -> (T, RAW_DIM)
    - Flattened formats

    Args:
        seq: Input sequence in various formats
        target_len: Optional target length for resampling

    Returns:
        Sequence of shape (T, RAW_DIM) or None if conversion fails
    """
    try:
        arr = np.asarray(seq, dtype=np.float32)
    except Exception:
        return None

    if arr.size == 0:
        t = 0 if target_len is None else target_len
        return np.zeros((t, RAW_DIM), dtype=np.float32)

    # Handle different input shapes
    if arr.ndim == 3 and arr.shape[1] == NUM_LANDMARKS and arr.shape[2] == NUM_COORDS:
        # (T, 21, 3) -> (T, 63)
        arr = arr.reshape(arr.shape[0], RAW_DIM)
    elif arr.ndim == 2 and arr.shape == (NUM_LANDMARKS, NUM_COORDS):
        # (21, 3) -> (1, 63)
        arr = arr.reshape(1, RAW_DIM)
    elif arr.ndim == 2 and arr.shape[1] == RAW_DIM:
        # Already in correct format (T, 63)
        pass
    elif arr.ndim == 2 and arr.shape[0] == RAW_DIM and arr.shape[1] != RAW_DIM:
        # Transpose (63, T) -> (T, 63)
        arr = arr.T
        if arr.shape[1] != RAW_DIM:
            return None
    elif arr.ndim == 1 and arr.size % RAW_DIM == 0:
        # Flatten to (T, 63)
        arr = arr.reshape(-1, RAW_DIM)
    else:
        return None

    arr = np.ascontiguousarray(arr, dtype=np.float32)

    if target_len is not None and arr.shape[0] != target_len:
        arr = resample(arr, target_len)

    return arr


def interp_extrap_1d(valid_idx, valid_vals, n: int):
    """
    Interpolate and extrapolate 1D values.

    Args:
        valid_idx: Indices of valid values
        valid_vals: Valid values at those indices
        n: Total number of points

    Returns:
        Interpolated and extrapolated values of length n
    """
    xi = np.arange(n, dtype=np.float32)
    xp = np.asarray(valid_idx, dtype=np.float32)
    fp = np.asarray(valid_vals, dtype=np.float32)
    yi = np.interp(xi, xp, fp).astype(np.float32)

    if len(valid_idx) >= 2:
        # Left extrapolation
        left = xi < valid_idx[0]
        if left.any():
            dx = float(valid_idx[1] - valid_idx[0])
            slope = 0.0 if dx == 0 else float((fp[1] - fp[0]) / dx)
            yi[left] = fp[0] + (xi[left] - valid_idx[0]) * slope

        # Right extrapolation
        right = xi > valid_idx[-1]
        if right.any():
            dx = float(valid_idx[-1] - valid_idx[-2])
            slope = 0.0 if dx == 0 else float((fp[-1] - fp[-2]) / dx)
            yi[right] = fp[-1] + (xi[right] - valid_idx[-1]) * slope

    return yi.astype(np.float32)


def interpolate_missing(lm_list):
    """
    Interpolate missing landmarks in a sequence.

    Args:
        lm_list: List of landmarks, with None for missing frames

    Returns:
        Interpolated sequence of shape (n, RAW_DIM)
    """
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
            pad[: lm.size] = lm
            lm = pad
        elif lm.size > RAW_DIM:
            lm = lm[:RAW_DIM]
        valid_arr.append(lm.astype(np.float32))
        result[i] = lm.astype(np.float32)

    valid_arr = np.stack(valid_arr, axis=0)

    if len(valid_idx) == 1:
        result[:] = valid_arr[0]
        return result

    # Interpolate each dimension
    for d in range(RAW_DIM):
        result[:, d] = interp_extrap_1d(valid_idx, valid_arr[:, d], n)

    return result.astype(np.float32)
