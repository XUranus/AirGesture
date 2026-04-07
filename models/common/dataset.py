"""
Dataset and data loading utilities.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .constants import SEQ_LEN, NUM_CLASSES, CLASS_NAMES
from .utils import resample, to_raw_sequence
from .features import compute_features
from .augmentation import add_jitter, time_warp


class GestureDataset(Dataset):
    """
    Dataset for gesture recognition.

    Handles:
    - Raw landmark sequences
    - Feature computation
    - Normalization
    - Data augmentation (optional)
    """

    def __init__(self, raw_samples, labels, norm_stats=None, augment: bool = False):
        """
        Initialize dataset.

        Args:
            raw_samples: List of raw landmark sequences
            labels: List of labels
            norm_stats: Dict with 'mean' and 'std' for normalization
            augment: Whether to apply data augmentation
        """
        self.raw_samples = raw_samples
        self.labels = labels
        self.norm_stats = norm_stats
        self.augment = augment

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        raw = self.raw_samples[idx].copy()
        label = self.labels[idx]

        # Apply augmentation
        if self.augment and random.random() < 0.5:
            raw = add_jitter(raw, sigma=0.002)
        if self.augment and random.random() < 0.3:
            raw = time_warp(raw)
            raw = resample(raw, SEQ_LEN)

        # Compute features
        feat = compute_features(raw)

        # Normalize
        if self.norm_stats is not None:
            feat = (feat - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-8)

        # Transpose to (feat_dim, seq_len) for Conv1d
        x = torch.FloatTensor(feat.T)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced dataset.

    Args:
        labels: List of labels

    Returns:
        Tensor of class weights
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = w / w.sum() * NUM_CLASSES
    return torch.FloatTensor(w)


def make_sampler(labels):
    """
    Create a WeightedRandomSampler for imbalanced dataset.

    Args:
        labels: List of labels

    Returns:
        WeightedRandomSampler
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts = np.maximum(counts, 1.0)
    sw = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)


def load_cache(cache_path):
    """
    Load data from cache file.

    Args:
        cache_path: Path to the .npz cache file

    Returns:
        Tuple of (samples, labels)
    """
    samples = np.load(cache_path, allow_pickle=True)["samples"]
    labels = np.load(cache_path, allow_pickle=True)["labels"]

    out_samples = [np.ascontiguousarray(s, dtype=np.float32) for s in samples]
    out_labels = labels.astype(np.int64).tolist()

    return out_samples, out_labels
