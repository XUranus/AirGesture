"""
Common modules for AirGesture project.

This package contains shared code used across preprocessing, training, and deployment notebooks.
"""

from .constants import (
    SEQ_LEN,
    NUM_LANDMARKS,
    NUM_COORDS,
    RAW_DIM,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_IDX,
    SWIPE_CLASSES,
    CACHE_VERSION,
    MIN_VALID_RATIO,
    WRIST_IDX,
    MID_FINGER_IDX,
    FINGERTIP_IDS,
    BASE_IDS,
    PAIRS,
    N_PAIRS,
    FINGER_CHAINS,
    N_FINGERS,
    FEATURE_DIM,
)

from .log import log_info, log_warn, log_err

from .env import (
    detect_environment,
    get_save_dir,
    get_dataset_path,
    setup_environment,
    get_device,
)

from .utils import to_scalar, resample, to_raw_sequence, interp_extrap_1d, interpolate_missing

from .features import compute_features

from .augmentation import (
    mirror_x,
    rotate_2d,
    scale_landmarks,
    add_jitter,
    time_warp,
    speed_change,
)

from .detection import HandDetector

from .model import GestureTCN, count_parameters, get_model_size_mb

from .dataset import GestureDataset, compute_class_weights, make_sampler

from .evaluation import evaluate, evaluate_model

__all__ = [
    # Constants
    "SEQ_LEN",
    "NUM_LANDMARKS",
    "NUM_COORDS",
    "RAW_DIM",
    "NUM_CLASSES",
    "CLASS_NAMES",
    "CLASS_TO_IDX",
    "SWIPE_CLASSES",
    "CACHE_VERSION",
    "MIN_VALID_RATIO",
    "WRIST_IDX",
    "MID_FINGER_IDX",
    "FINGERTIP_IDS",
    "BASE_IDS",
    "PAIRS",
    "N_PAIRS",
    "FINGER_CHAINS",
    "N_FINGERS",
    "FEATURE_DIM",
    # Log
    "log_info",
    "log_warn",
    "log_err",
    # Environment
    "detect_environment",
    "get_save_dir",
    "get_dataset_path",
    "setup_environment",
    "get_device",
    # Utils
    "to_scalar",
    "resample",
    "to_raw_sequence",
    "interp_extrap_1d",
    "interpolate_missing",
    # Features
    "compute_features",
    # Augmentation
    "mirror_x",
    "rotate_2d",
    "scale_landmarks",
    "add_jitter",
    "time_warp",
    "speed_change",
    # Detection
    "HandDetector",
    # Model
    "GestureTCN",
    "count_parameters",
    "get_model_size_mb",
    # Dataset
    "GestureDataset",
    "compute_class_weights",
    "make_sampler",
    # Evaluation
    "evaluate",
    "evaluate_model",
]
