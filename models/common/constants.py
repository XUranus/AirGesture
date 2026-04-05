"""
Constants for AirGesture project.
"""

# Sequence and data dimensions
SEQ_LEN = 30
NUM_LANDMARKS = 21
NUM_COORDS = 3
RAW_DIM = NUM_LANDMARKS * NUM_COORDS  # 63

# Class definitions
NUM_CLASSES = 5
CLASS_NAMES = [
    "grab",       # 握拳
    "release",    # 释放
    "swipe_up",   # 上滑
    "swipe_down", # 下滑
    "noise",      # 无效动作
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SWIPE_CLASSES = {"swipe_up", "swipe_down"}

# Cache version - update this when changing preprocessing logic
CACHE_VERSION = "v3_fixed_swipe_reverse"

# Minimum ratio of valid landmarks required
MIN_VALID_RATIO = 0.10

# Landmark indices
WRIST_IDX = 0
MID_FINGER_IDX = 9
FINGERTIP_IDS = [4, 8, 12, 16, 20]
BASE_IDS = [2, 5, 9, 13, 17]

# Distance pairs for feature computation
PAIRS = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (4, 12),
    (4, 16),
    (4, 20),
    (8, 16),
    (8, 20),
    (12, 20),
]
N_PAIRS = len(PAIRS)  # 10

# Finger chains for angle computation
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],   # thumb
    [0, 5, 6, 7, 8],   # index
    [0, 9, 10, 11, 12], # middle
    [0, 13, 14, 15, 16], # ring
    [0, 17, 18, 19, 20], # pinky
]
N_FINGERS = 5

# Feature dimension: normalized landmarks (63) + velocity (63) + wrist velocity (3) + distances (10) + angles (5)
FEATURE_DIM = RAW_DIM + RAW_DIM + 3 + N_PAIRS + N_FINGERS  # 63 + 63 + 3 + 10 + 5 = 144
