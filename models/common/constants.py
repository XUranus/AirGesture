SEQ_LEN = 30
NUM_LANDMARKS = 21
NUM_COORDS = 3
RAW_DIM = NUM_LANDMARKS * NUM_COORDS
NUM_CLASSES = 5
CACHE_VERSION = "v3_fixed_swipe_reverse"  # Fixed swipe reverse augmentation
MIN_VALID_RATIO = 0.10

# Feature computation constants
WRIST_IDX = 0
MID_FINGER_IDX = 9
FINGERTIP_IDS = [4, 8, 12, 16, 20]
BASE_IDS = [2, 5, 9, 13, 17]
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

# Class names
CLASS_NAMES = [
    "grab", 
    "release",
    "swipe_up",
    "swipe_down",
    "noise", 
]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
SWIPE_CLASSES = {"swipe_up", "swipe_down"}