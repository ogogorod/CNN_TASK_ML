"""Small shared helpers used across the pipeline."""
import os
import random
import json

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make runs reproducible (as much as cuDNN allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for an imbalanced CrossEntropyLoss.
    weight_c = N / (num_classes * count_c)
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1  # avoid div by zero for any missing class
    n = counts.sum()
    weights = n / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def ensure_dirs(*paths) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class EarlyStopper:
    """Stops training when validation metric stops improving."""

    def __init__(self, patience: int = 6, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if `value` is the new best score."""
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
