"""
Create dataset class-distribution summaries.

This script summarizes:

1. Raw dataset discovered by dataset.py:
    data/raw/img
    data/raw/ann

2. Recreated train/val/test split from the raw dataset:
    same config.SEED and split logic as training

3. Exported balanced test dataset:
    data/test/labels.csv

Outputs:
    outputs/reports/dataset_summary.csv
    outputs/reports/dataset_summary.json

    outputs/figures/dataset_summary/raw_class_distribution.png
    outputs/figures/dataset_summary/train_val_test_distribution.png
    outputs/figures/dataset_summary/exported_test_class_distribution.png

This script does NOT train anything.
It only reads dataset labels and creates summaries.

Run from project root:
    cd /Users/oleg/Desktop/Claude_cnn/weapon_classifier
    python src/dataset_summary.py
"""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from dataset import discover_samples, split_samples


def make_output_dirs():
    """
    Create output folders.
    """
    reports_dir = Path(config.REPORT_DIR)
    figures_dir = Path(config.OUTPUT_DIR) / "figures" / "dataset_summary"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return reports_dir, figures_dir


def count_labels_from_samples(samples) -> Counter:
    """
    Count class labels from samples returned by dataset.py.

    samples format:
        [(image_path, label_idx), ...]
    """
    counts = Counter()

    for _, label_idx in samples:
        class_name = config.IDX_TO_CLASS[label_idx]
        counts[class_name] += 1

    return counts


def count_exported_test_labels() -> Counter:
    """
    Count class labels from data/test/labels.csv.
    """
    labels_path = Path(config.PROJECT_ROOT) / "data" / "test" / "labels.csv"

    counts = Counter()

    if not labels_path.exists():
        print(f"Exported test labels not found, skipping: {labels_path}")
        return counts

    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label_name = row.get("label_name")

            if label_name in config.CLASS_NAMES:
                counts[label_name] += 1

    return counts


def counts_to_rows(split_name: str, counts: Counter) -> List[Dict[str, Any]]:
    """
    Convert class counts into table rows.
    """
    total = sum(counts.values())

    rows = []

    for class_name in config.CLASS_NAMES:
        count = counts.get(class_name, 0)

        if total > 0:
            percentage = count / total * 100
        else:
            percentage = 0.0

        rows.append({
            "split": split_name,
            "class_name": class_name,
            "count": int(count),
            "percentage": float(percentage),
        })

    return rows


def save_summary_files(all_rows: List[Dict[str, Any]], reports_dir: Path):
    """
    Save dataset summary as CSV and JSON.
    """
    summary_df = pd.DataFrame(all_rows)

    csv_path = reports_dir / "dataset_summary.csv"
    json_path = reports_dir / "dataset_summary.json"

    summary_df.to_csv(csv_path, index=False)

    grouped = {}

    for split_name in summary_df["split"].unique():
        split_df = summary_df[summary_df["split"] == split_name]
        grouped[split_name] = {
            row["class_name"]: {
                "count": int(row["count"]),
                "percentage": float(row["percentage"]),
            }
            for _, row in split_df.iterrows()
        }

    with open(json_path, "w") as f:
        json.dump(grouped, f, indent=2)

    print()
    print("Dataset summary:")
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved JSON to: {json_path}")


def plot_single_distribution(
    counts: Counter,
    title: str,
    save_path: Path,
):
    """
    Plot one class distribution bar chart.
    """
    class_names = config.CLASS_NAMES
    values = [counts.get(class_name, 0) for class_name in class_names]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(class_names, values)

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_train_val_test_distribution(
    train_counts: Counter,
    val_counts: Counter,
    test_counts: Counter,
    save_path: Path,
):
    """
    Plot grouped class distribution for train/val/test split.
    """
    class_names = config.CLASS_NAMES
    x = np.arange(len(class_names))
    width = 0.25

    train_values = [train_counts.get(class_name, 0) for class_name in class_names]
    val_values = [val_counts.get(class_name, 0) for class_name in class_names]
    test_values = [test_counts.get(class_name, 0) for class_name in class_names]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, train_values, width, label="Train")
    ax.bar(x, val_values, width, label="Validation")
    ax.bar(x + width, test_values, width, label="Test")

    ax.set_title("Train / Validation / Test Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    reports_dir, figures_dir = make_output_dirs()

    print("Discovering raw dataset samples...")
    samples = discover_samples()

    print(f"Discovered labeled samples: {len(samples)}")

    train_split, val_split, test_split = split_samples(samples)

    raw_counts = count_labels_from_samples(samples)
    train_counts = count_labels_from_samples(train_split)
    val_counts = count_labels_from_samples(val_split)
    original_test_counts = count_labels_from_samples(test_split)
    exported_test_counts = count_exported_test_labels()

    all_rows = []
    all_rows.extend(counts_to_rows("raw_all", raw_counts))
    all_rows.extend(counts_to_rows("train", train_counts))
    all_rows.extend(counts_to_rows("validation", val_counts))
    all_rows.extend(counts_to_rows("original_test_split", original_test_counts))

    if sum(exported_test_counts.values()) > 0:
        all_rows.extend(counts_to_rows("exported_balanced_test", exported_test_counts))

    save_summary_files(all_rows, reports_dir)

    plot_single_distribution(
        counts=raw_counts,
        title="Raw Dataset Class Distribution",
        save_path=figures_dir / "raw_class_distribution.png",
    )

    plot_train_val_test_distribution(
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=original_test_counts,
        save_path=figures_dir / "train_val_test_distribution.png",
    )

    if sum(exported_test_counts.values()) > 0:
        plot_single_distribution(
            counts=exported_test_counts,
            title="Exported Balanced Test Dataset Class Distribution",
            save_path=figures_dir / "exported_test_class_distribution.png",
        )

    print()
    print("Dataset summary finished.")
    print(f"Figure output folder: {figures_dir}")
    print("This script did not train or overwrite any checkpoint.")


if __name__ == "__main__":
    main()
