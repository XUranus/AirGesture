#!/usr/bin/env python3
"""
train.py - Gesture Recognition 1D-CNN Training + L1 Pruning Pipeline

Extracts hand landmarks via MediaPipe, trains a 1D-CNN temporal classifier
on (30, 63) keypoint sequences, and applies L1 unstructured pruning to
the fully-connected classification head.

Compatible with both the new MediaPipe Tasks API and the legacy Solutions API.
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report

# Global constants
SEQ_LEN = 30
NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS  # 63
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-4
PRUNE_AMOUNT = 0.35
SLIDING_STRIDE = 10
MIN_VALID_RATIO = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "grab",
    "swipe_up",
    "swipe_down",
    "swipe_left",
    "swipe_right",
    "finger_heart",
    "wave",
    "noise",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# MediaPipe Hand Landmarker model file
HAND_LANDMARKER_MODEL = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model_file(model_path=HAND_LANDMARKER_MODEL):
    """Download hand_landmarker.task if it does not exist locally."""
    if os.path.exists(model_path):
        return model_path

    logger.info(f"Model file {model_path} not found. Downloading...")
    try:
        import requests

        resp = requests.get(MODEL_URL, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(model_path, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  Download progress: {pct:.1f}% ({downloaded}/{total})", end="")
        print()
        logger.info(f"Model file downloaded: {model_path}")
    except ImportError:
        logger.error("Please install requests: pip install requests")
        logger.error(f"Or manually download: {MODEL_URL}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error(f"Please manually download: {MODEL_URL}")
        sys.exit(1)

    return model_path


class HandDetector:
    """Hand landmark detector compatible with both MediaPipe Tasks API and
    the legacy Solutions API. Automatically selects the available backend."""

    def __init__(self, static_mode=True, max_hands=1, min_detection_conf=0.5):
        self.api_version = None
        self.detector = None

        # Try the new Tasks API first
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = ensure_model_file(HAND_LANDMARKER_MODEL)
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_hands,
                min_hand_detection_confidence=min_detection_conf,
                min_hand_presence_confidence=min_detection_conf,
                min_tracking_confidence=0.5,
            )
            self.detector = mp_vision.HandLandmarker.create_from_options(options)
            self.mp = mp
            self.api_version = "tasks"
            logger.info("Using MediaPipe Tasks API (new)")
            return
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Tasks API unavailable: {e}")

        # Fall back to the legacy Solutions API
        try:
            import mediapipe as mp

            hands = mp.solutions.hands
            self.detector = hands.Hands(
                static_image_mode=static_mode,
                max_num_hands=max_hands,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=0.5,
            )
            self.mp = mp
            self.api_version = "solutions"
            logger.info("Using MediaPipe Solutions API (legacy)")
            return
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Solutions API unavailable: {e}")

        logger.error("Failed to initialise MediaPipe. Try: pip install mediapipe --upgrade")
        sys.exit(1)

    def detect(self, frame_bgr):
        """Return a (63,) numpy array of hand landmarks or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.api_version == "tasks":
            return self._detect_tasks(rgb)
        return self._detect_solutions(rgb)

    def _detect_tasks(self, rgb):
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32)
        return None

    def _detect_solutions(self, rgb):
        results = self.detector.process(rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords, dtype=np.float32)
        return None

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()


def extract_landmarks_from_video(video_path, detector):
    """Extract per-frame hand landmarks from a video file.
    Returns (landmarks_list, total_frame_count)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return [], 0

    landmarks = []
    total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        landmarks.append(detector.detect(frame))
    cap.release()
    return landmarks, total_frames


def interpolate_missing(landmarks_list):
    """Fill missing frames with linear interpolation across the feature dims."""
    n = len(landmarks_list)
    if n == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    valid_indices = [i for i, lm in enumerate(landmarks_list) if lm is not None]
    if len(valid_indices) == 0:
        return np.zeros((n, FEATURE_DIM), dtype=np.float32)

    result = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for i in valid_indices:
        result[i] = landmarks_list[i]

    if len(valid_indices) == 1:
        result[:] = result[valid_indices[0]]
        return result

    valid_arr = np.array([landmarks_list[i] for i in valid_indices])
    for dim in range(FEATURE_DIM):
        fn = interp1d(valid_indices, valid_arr[:, dim], kind="linear", fill_value="extrapolate")
        result[:, dim] = fn(np.arange(n))
    return result


def resample_sequence(seq, target_len=SEQ_LEN):
    """Resample a variable-length sequence to exactly *target_len* frames."""
    n = len(seq)
    if n == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    if n == target_len:
        return seq.copy()

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    for dim in range(FEATURE_DIM):
        resampled[:, dim] = np.interp(x_new, x_old, seq[:, dim])
    return resampled


def generate_samples_from_video(landmarks_list, total_frames):
    """Generate fixed-length training samples from one video using
    global resampling, sliding window and random cropping."""
    valid_count = sum(1 for lm in landmarks_list if lm is not None)
    if total_frames == 0 or valid_count / total_frames < MIN_VALID_RATIO:
        return []

    seq = interpolate_missing(landmarks_list)
    samples = []
    n = len(seq)

    # Strategy 1: resample the whole video to SEQ_LEN frames
    samples.append(resample_sequence(seq, SEQ_LEN))

    # Strategy 2: sliding window
    if n >= SEQ_LEN:
        for start in range(0, n - SEQ_LEN + 1, SLIDING_STRIDE):
            samples.append(seq[start : start + SEQ_LEN].copy())

    # Strategy 3: random crops
    if n > SEQ_LEN:
        for _ in range(2):
            start = random.randint(0, n - SEQ_LEN)
            samples.append(seq[start : start + SEQ_LEN].copy())

    return samples


class GestureAugmentor:
    """Online data augmentation applied to (SEQ_LEN, 63) keypoint sequences."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, seq):
        seq = seq.copy()
        if random.random() < self.p:
            seq = self.add_noise(seq)
        if random.random() < self.p:
            seq = self.random_scale(seq)
        if random.random() < self.p:
            seq = self.random_rotation(seq)
        if random.random() < self.p:
            seq = self.random_shift(seq)
        if random.random() < self.p:
            seq = self.time_warp(seq)
        if random.random() < self.p * 0.6:
            seq = self.spatial_flip(seq)
        if random.random() < self.p * 0.4:
            seq = self.random_mask(seq)
        if random.random() < self.p * 0.5:
            seq = self.frame_dropout(seq)
        return seq

    @staticmethod
    def add_noise(seq, sigma=0.005):
        return seq + np.random.randn(*seq.shape).astype(np.float32) * sigma

    @staticmethod
    def random_scale(seq, low=0.85, high=1.15):
        return seq * np.random.uniform(low, high)

    @staticmethod
    def random_rotation(seq, max_angle=15):
        angle = np.radians(np.random.uniform(-max_angle, max_angle))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = seq.copy()
        for i in range(NUM_LANDMARKS):
            xi, yi = i * 3, i * 3 + 1
            x, y = seq[:, xi], seq[:, yi]
            rotated[:, xi] = x * cos_a - y * sin_a
            rotated[:, yi] = x * sin_a + y * cos_a
        return rotated

    @staticmethod
    def random_shift(seq, max_shift=0.05):
        sx = np.random.uniform(-max_shift, max_shift)
        sy = np.random.uniform(-max_shift, max_shift)
        sz = np.random.uniform(-max_shift * 0.5, max_shift * 0.5)
        shifted = seq.copy()
        for i in range(NUM_LANDMARKS):
            shifted[:, i * 3] += sx
            shifted[:, i * 3 + 1] += sy
            shifted[:, i * 3 + 2] += sz
        return shifted

    @staticmethod
    def time_warp(seq, sigma=0.2):
        n = len(seq)
        warp_pts = np.sort(np.random.uniform(0, 1, 4))
        src = np.array([0, *warp_pts, 1])
        jitter = np.clip(np.random.randn(4) * sigma, -0.4, 0.4)
        dst = np.sort(np.clip(np.array([0, *(warp_pts + jitter * 0.1), 1]), 0, 1))
        mapping = interp1d(src, dst, kind="linear", fill_value="extrapolate")
        x_old = np.linspace(0, 1, n)
        x_new = np.clip(mapping(x_old), 0, 1)
        warped = np.zeros_like(seq)
        for dim in range(seq.shape[1]):
            warped[:, dim] = np.interp(np.linspace(0, 1, n), x_new, seq[:, dim])
        return warped

    @staticmethod
    def spatial_flip(seq):
        flipped = seq.copy()
        for i in range(NUM_LANDMARKS):
            flipped[:, i * 3] = 1.0 - flipped[:, i * 3]
        return flipped

    @staticmethod
    def random_mask(seq, max_landmarks=5):
        masked = seq.copy()
        n_mask = random.randint(1, max_landmarks)
        for lid in random.sample(range(NUM_LANDMARKS), n_mask):
            masked[:, lid * 3 : lid * 3 + 3] = 0
        return masked

    @staticmethod
    def frame_dropout(seq, max_drop_ratio=0.2):
        n = len(seq)
        n_drop = max(1, int(n * random.uniform(0.05, max_drop_ratio)))
        drop_ids = sorted(random.sample(range(1, n - 1), min(n_drop, n - 2)))
        result = seq.copy()
        for idx in drop_ids:
            result[idx] = (result[idx - 1] + result[idx + 1]) / 2.0
        return result


class GestureDataset(Dataset):
    """PyTorch dataset wrapping pre-extracted keypoint sequences."""

    def __init__(self, samples, labels, augmentor=None, normalize_stats=None):
        self.samples = samples
        self.labels = labels
        self.augmentor = augmentor
        self.mean = normalize_stats["mean"] if normalize_stats else None
        self.std = normalize_stats["std"] if normalize_stats else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx].copy()
        label = self.labels[idx]

        if self.augmentor is not None:
            seq = self.augmentor(seq)

        if self.mean is not None and self.std is not None:
            seq = (seq - self.mean) / (self.std + 1e-8)

        # Transpose to (channels=63, length=30) for Conv1d
        seq = seq.T
        return torch.FloatTensor(seq), torch.LongTensor([label]).squeeze()


class GestureCNN1D(nn.Module):
    """1D-CNN temporal classifier.
    Input shape : (batch, 63, 30)
    Output shape: (batch, num_classes)
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(FEATURE_DIM, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classification head (target of L1 pruning)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_dataset(data_dir, detector, cache_path=None):
    """Walk the class sub-directories, extract landmarks from every video
    and produce fixed-length samples."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached data: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        raw_samples = cache["samples"]
        raw_labels = cache["labels"]
        # Ensure every sample is a proper float32 array
        samples = [np.array(s, dtype=np.float32) for s in raw_samples]
        labels = [int(l) for l in raw_labels]
        return samples, labels

    samples, labels = [], []
    class_counts = defaultdict(int)
    data_path = Path(data_dir)

    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        label_idx = CLASS_TO_IDX[class_name]
        video_files = sorted(
            f for f in class_dir.iterdir()
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
        )
        logger.info(f"  Class [{class_name}]: found {len(video_files)} videos")

        for vf in tqdm(video_files, desc=f"  {class_name}", leave=False):
            landmarks_list, total_frames = extract_landmarks_from_video(vf, detector)
            if total_frames == 0:
                continue
            for s in generate_samples_from_video(landmarks_list, total_frames):
                samples.append(s.astype(np.float32))
                labels.append(label_idx)
                class_counts[class_name] += 1

    logger.info(f"Dataset statistics ({data_dir}):")
    for cn in CLASS_NAMES:
        logger.info(f"  {cn:>15s}: {class_counts[cn]:>5d} samples")
    logger.info(f"  {'Total':>15s}: {len(samples):>5d} samples")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            samples=np.array(samples, dtype=object),
            labels=np.array(labels, dtype=np.int64),
        )
        logger.info(f"Cache saved: {cache_path}")

    return samples, labels


def compute_normalize_stats(samples):
    """Compute per-feature mean and std from the training set."""
    all_data = np.array([s.astype(np.float32) for s in samples], dtype=np.float32)
    return {
        "mean": all_data.mean(axis=(0, 1)).astype(np.float32),
        "std": all_data.std(axis=(0, 1)).astype(np.float32),
    }


def create_weighted_sampler(labels):
    """Build a WeightedRandomSampler to counter class imbalance."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bx.size(0)
        correct += (logits.argmax(1) == by).sum().item()
        total += bx.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        loss = criterion(logits, by)
        total_loss += loss.item() * bx.size(0)
        preds = logits.argmax(1)
        correct += (preds == by).sum().item()
        total += bx.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(by.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def apply_l1_pruning(model, amount=PRUNE_AMOUNT):
    """Apply L1 unstructured pruning to all Linear layers inside
    model.classifier. Convolutional layers are left intact."""
    logger.info(f"Applying L1 unstructured pruning to FC layers (ratio={amount:.0%})")
    pruned = 0
    for name, module in model.classifier.named_modules():
        if isinstance(module, nn.Linear):
            n_total = module.weight.nelement()
            z_before = (module.weight == 0).sum().item()
            prune.l1_unstructured(module, name="weight", amount=amount)
            z_after = (module.weight == 0).sum().item()
            logger.info(
                f"  classifier.{name}: params={n_total}, "
                f"zeros_before={z_before}, zeros_after={z_after} "
                f"({z_after / n_total:.1%})"
            )
            pruned += 1
    logger.info(f"Pruned {pruned} FC layer(s)")
    return model


def make_pruning_permanent(model):
    """Remove the pruning re-parametrisation hooks so the masks are
    baked into the weight tensors permanently."""
    for _, module in model.classifier.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    logger.info("Pruning masks made permanent")


def print_sparsity_report(model):
    logger.info("Sparsity report:")
    logger.info(f"{'Layer':>35s} | {'Params':>10s} | {'Zeros':>10s} | {'Sparsity':>8s}")
    logger.info("-" * 75)
    tp, tz = 0, 0
    for name, param in model.named_parameters():
        if "weight" in name:
            n_p = param.nelement()
            n_z = (param == 0).sum().item()
            tp += n_p
            tz += n_z
            logger.info(f"{name:>35s} | {n_p:>10d} | {n_z:>10d} | {n_z / n_p:>7.1%}")
    logger.info("-" * 75)
    logger.info(f"{'Total':>35s} | {tp:>10d} | {tz:>10d} | {tz / tp:>7.1%}")


def save_model(model, normalize_stats, save_dir="checkpoints"):
    """Persist the pruned model weights and a JSON config file."""
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "gesture_cnn1d_pruned.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Pruned model saved: {model_path}")

    config = {
        "class_names": CLASS_NAMES,
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "num_classes": NUM_CLASSES,
        "num_landmarks": NUM_LANDMARKS,
        "prune_amount": PRUNE_AMOUNT,
        "normalize_mean": normalize_stats["mean"].tolist(),
        "normalize_std": normalize_stats["std"].tolist(),
    }
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Gesture 1D-CNN Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--no_cache", action="store_true", help="Force re-extraction of landmarks")
    parser.add_argument("--prune_amount", type=float, default=PRUNE_AMOUNT)
    args = parser.parse_args()

    logger.info("Gesture Recognition 1D-CNN Training Pipeline")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Seq length: {SEQ_LEN}, Feature dim: {FEATURE_DIM}")
    logger.info(f"Classes: {NUM_CLASSES}, Epochs: {args.epochs}")
    logger.info(f"Prune ratio: {args.prune_amount:.0%}")

    # Phase 1: landmark extraction
    logger.info("[Phase 1] Extracting hand landmarks...")
    detector = HandDetector(static_mode=True, max_hands=1, min_detection_conf=0.5)

    train_dir = os.path.join(args.data_dir, "Train")
    test_dir = os.path.join(args.data_dir, "Test")
    cache_dir = os.path.join(args.save_dir, "cache")
    train_cache = None if args.no_cache else os.path.join(cache_dir, "train.npz")
    test_cache = None if args.no_cache else os.path.join(cache_dir, "test.npz")

    logger.info("Processing training set...")
    train_samples, train_labels = load_dataset(train_dir, detector, train_cache)
    logger.info("Processing test set...")
    test_samples, test_labels = load_dataset(test_dir, detector, test_cache)
    detector.close()

    if len(train_samples) == 0:
        logger.error("Training set is empty. Check the data directory.")
        sys.exit(1)

    # Phase 2: normalisation statistics
    logger.info("[Phase 2] Computing normalisation statistics...")
    norm_stats = compute_normalize_stats(train_samples)
    logger.info(
        f"  Mean range: [{norm_stats['mean'].min():.4f}, {norm_stats['mean'].max():.4f}]"
    )
    logger.info(
        f"  Std  range: [{norm_stats['std'].min():.4f}, {norm_stats['std'].max():.4f}]"
    )

    # Phase 3: data loaders
    logger.info("[Phase 3] Building data loaders...")
    augmentor = GestureAugmentor(p=0.5)
    train_dataset = GestureDataset(train_samples, train_labels, augmentor=augmentor, normalize_stats=norm_stats)
    test_dataset = GestureDataset(test_samples, test_labels, augmentor=None, normalize_stats=norm_stats)

    sampler = create_weighted_sampler(train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    logger.info(f"  Test : {len(test_dataset)} samples, {len(test_loader)} batches/epoch")

    # Phase 4: model
    logger.info("[Phase 4] Initialising model...")
    model = GestureCNN1D(num_classes=NUM_CLASSES).to(DEVICE)
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # Phase 5: training
    logger.info("[Phase 5] Training...")
    best_acc, best_epoch = 0.0, 0
    patience, patience_ctr = 25, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | "
            f"Test  Loss:{te_loss:.4f} Acc:{te_acc:.4f} | LR:{lr_now:.2e}"
        )

        if te_acc > best_acc:
            best_acc, best_epoch, patience_ctr = te_acc, epoch, 0
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, "gesture_cnn1d_best.pth"))
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            logger.info(f"Early stopping triggered (patience={patience})")
            break

    logger.info(f"Training complete. Best test accuracy: {best_acc:.4f} (epoch {best_epoch})")

    # Phase 6: evaluate best model
    logger.info("[Phase 6] Loading best model for evaluation...")
    model.load_state_dict(
        torch.load(os.path.join(args.save_dir, "gesture_cnn1d_best.pth"), map_location=DEVICE, weights_only=True)
    )
    _, te_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    logger.info(f"Best model test accuracy: {te_acc:.4f}")
    logger.info("Classification report (before pruning):\n" + classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    ))

    # Phase 7: L1 pruning
    logger.info("[Phase 7] L1 unstructured pruning on FC layers...")
    model = apply_l1_pruning(model, amount=args.prune_amount)

    _, te_acc_p, preds_p, labels_p = evaluate(model, test_loader, criterion, DEVICE)
    logger.info(f"Post-pruning test accuracy: {te_acc_p:.4f} (delta: {te_acc_p - te_acc:+.4f})")
    logger.info("Classification report (after pruning):\n" + classification_report(
        labels_p, preds_p, target_names=CLASS_NAMES, digits=4, zero_division=0
    ))

    make_pruning_permanent(model)
    print_sparsity_report(model)

    # Phase 8: save
    logger.info("[Phase 8] Saving final model...")
    save_model(model, norm_stats, save_dir=args.save_dir)
    logger.info(f"Pipeline finished. Model: {args.save_dir}/gesture_cnn1d_pruned.pth")


if __name__ == "__main__":
    main()
