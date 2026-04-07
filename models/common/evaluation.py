"""
Evaluation utilities for gesture recognition.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        loader: DataLoader
        criterion: Loss function
        device: Device to run on

    Returns:
        Tuple of (avg_loss, accuracy, predictions, labels)
    """
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


@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    Evaluate model and return metrics.

    Args:
        model: PyTorch model
        loader: DataLoader
        device: Device to run on

    Returns:
        Dict with accuracy, f1_score, predictions, and labels
    """
    model.eval()
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "predictions": all_preds,
        "labels": all_labels,
    }
