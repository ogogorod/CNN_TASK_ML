"""
Dataset discovery, label extraction, and stratified splitting.

Label source priority:
  1. The annotation JSON's tags[0]["name"]   (most reliable, ground truth)
  2. Fallback: the first "_"-separated token of the filename
     (e.g. "knife_ABbframe00430_box1.jpg" -> "knife")
"""
import os
import json
import glob

from collections import Counter
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import config


def _label_from_json(json_path: str) -> str | None:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        tags = data.get("tags", [])
        if tags:
            return tags[0].get("name")
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    return None


def _label_from_filename(filename: str) -> str:
    return filename.split("_")[0]


def discover_samples(img_dir: str = config.IMG_DIR,
                      ann_dir: str = config.ANN_DIR) -> List[Tuple[str, int]]:
    """
    Walk the image directory, pair each image with its annotation JSON,
    resolve a label, and return a list of (image_path, class_idx).
    Files whose label can't be resolved to a known class are skipped
    (and reported) rather than silently mislabeled.
    """
    image_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.jpeg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )

    samples = []
    skipped = 0
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        ann_path = os.path.join(ann_dir, fname + ".json")

        label = _label_from_json(ann_path) if os.path.exists(ann_path) else None
        if label is None:
            label = _label_from_filename(fname)

        if label not in config.CLASS_TO_IDX:
            skipped += 1
            continue

        samples.append((img_path, config.CLASS_TO_IDX[label]))

    if skipped:
        print(f"[discover_samples] Skipped {skipped} files with unresolved labels.")
    return samples


def split_samples(samples: List[Tuple[str, int]]):
    """Stratified 70/15/15 split (ratios from config) by class label."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels,
        train_size=config.TRAIN_FRAC,
        stratify=labels,
        random_state=config.SEED,
    )
    val_size = config.VAL_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels,
        train_size=val_size,
        stratify=rest_labels,
        random_state=config.SEED,
    )

    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    test = list(zip(test_paths, test_labels))
    return train, val, test


def print_split_summary(name: str, split: List[Tuple[str, int]]) -> None:
    counts = Counter(label for _, label in split)
    print(f"\n{name} set: {len(split)} samples")
    for idx in range(config.NUM_CLASSES):
        cls = config.IDX_TO_CLASS[idx]
        print(f"  {cls:12s}: {counts.get(idx, 0)}")


class WeaponDataset(Dataset):
    """Thin Dataset wrapper: (image_path, label) -> (tensor, label)."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
