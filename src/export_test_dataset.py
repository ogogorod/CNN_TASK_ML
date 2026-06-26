"""
Export a balanced version of the official held-out test split into data/test.

This script:
    1. Uses your existing dataset.py logic:
        - discover_samples()
        - split_samples()
    2. Recreates the same train/val/test split using config.SEED.
    3. Takes ONLY the official 15% test split.
    4. Balances it by taking the same number of images from each class.
       By default, it uses the smallest class count from the test split.
    5. Copies images and annotations into:

        data/test/
            img/
            ann/
            labels.csv
            summary.txt

Important:
    This does NOT train anything.
    It only copies files.

Run from project root:
    cd /Users/oleg/Desktop/ML_CNN/weapon_classifier
    python src/export_test_dataset.py

Optional:
    python src/export_test_dataset.py --samples-per-class 50
    python src/export_test_dataset.py --overwrite
"""

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import config
from dataset import discover_samples, split_samples


def get_project_root() -> Path:
    """
    Get project root from config.

    In the updated config.py, config.PROJECT_ROOT is a string path.
    """
    return Path(config.PROJECT_ROOT)


def get_output_root() -> Path:
    """
    Exported test dataset location:
        data/test/
    """
    return get_project_root() / "data" / "test"


def ensure_clean_output_folder(output_root: Path, overwrite: bool):
    """
    Create output folder.

    If the folder already exists:
        - stop by default
        - delete/recreate only if --overwrite is passed
    """
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {output_root}\n"
                "Use --overwrite if you want to recreate it."
            )
        shutil.rmtree(output_root)

    (output_root / "img").mkdir(parents=True, exist_ok=True)
    (output_root / "ann").mkdir(parents=True, exist_ok=True)


def get_annotation_path(image_path: Path) -> Path:
    """
    Your project expects annotation files as:
        ANN_DIR / image_filename + ".json"

    Example:
        image: pistol_001.jpg
        ann:   pistol_001.jpg.json
    """
    return Path(config.ANN_DIR) / f"{image_path.name}.json"


def load_or_create_annotation(source_ann_path: Path, target_label: str) -> dict:
    """
    Load existing annotation if available.

    Also rewrites the first tag name to target_label so the exported dataset
    has clean labels matching config.CLASS_NAMES.

    If the annotation is missing or broken, create a minimal annotation.
    """
    data = None

    if source_ann_path.exists():
        try:
            with open(source_ann_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = None

    if data is None:
        data = {
            "tags": [
                {
                    "name": target_label,
                    "value_type": "none",
                    "classes": [],
                }
            ]
        }
    else:
        if "tags" not in data or not data["tags"]:
            data["tags"] = [
                {
                    "name": target_label,
                    "value_type": "none",
                    "classes": [],
                }
            ]
        else:
            data["tags"][0]["name"] = target_label

    return data


def safe_filename(filename: str) -> str:
    """
    Make filename safe for copying.
    """
    result = []

    for char in filename:
        if char.isalnum() or char in ("-", "_", "."):
            result.append(char)
        else:
            result.append("_")

    return "".join(result)


def group_by_class(samples):
    """
    Convert list of (image_path, label_idx) into:
        class_idx -> list[(image_path, label_idx)]
    """
    grouped = defaultdict(list)

    for image_path, label_idx in samples:
        grouped[label_idx].append((image_path, label_idx))

    return grouped


def choose_balanced_samples(test_split, samples_per_class: int | None):
    """
    Select equal number of samples from each class.

    By default:
        use the smallest class count in the test split.

    If --samples-per-class is provided:
        use that number, but fail if some class has fewer images.
    """
    grouped = group_by_class(test_split)

    counts = {
        class_idx: len(grouped.get(class_idx, []))
        for class_idx in range(config.NUM_CLASSES)
    }

    missing_classes = [
        config.IDX_TO_CLASS[class_idx]
        for class_idx, count in counts.items()
        if count == 0
    ]

    if missing_classes:
        raise ValueError(
            "Some classes are missing from the test split, cannot create balanced dataset:\n"
            + ", ".join(missing_classes)
        )

    min_count = min(counts.values())

    if samples_per_class is None:
        n_per_class = min_count
    else:
        n_per_class = samples_per_class

        too_small = [
            f"{config.IDX_TO_CLASS[class_idx]} has only {count}"
            for class_idx, count in counts.items()
            if count < n_per_class
        ]

        if too_small:
            raise ValueError(
                f"Requested {n_per_class} samples per class, but some classes have fewer:\n"
                + "\n".join(too_small)
            )

    rng = random.Random(config.SEED)

    selected = []

    for class_idx in range(config.NUM_CLASSES):
        class_samples = grouped[class_idx]
        class_samples = sorted(class_samples, key=lambda x: x[0])
        chosen = rng.sample(class_samples, n_per_class)
        selected.extend(chosen)

    selected = sorted(selected, key=lambda x: (x[1], x[0]))

    return selected, counts, n_per_class


def export_samples(selected_samples, output_root: Path):
    """
    Copy selected images and annotations to output folder.
    """
    output_img_dir = output_root / "img"
    output_ann_dir = output_root / "ann"

    rows = []

    for image_path_str, label_idx in selected_samples:
        image_path = Path(image_path_str)
        label_name = config.IDX_TO_CLASS[label_idx]

        # Prefix with class name. This makes manual checking easier.
        new_image_name = f"{label_name}__test__{image_path.name}"
        new_image_name = safe_filename(new_image_name)

        destination_image_path = output_img_dir / new_image_name

        # Safety: avoid accidental overwrite if filenames collide.
        counter = 1
        while destination_image_path.exists():
            stem = destination_image_path.stem
            suffix = destination_image_path.suffix
            destination_image_path = output_img_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        shutil.copy2(image_path, destination_image_path)

        source_ann_path = get_annotation_path(image_path)

        annotation_data = load_or_create_annotation(
            source_ann_path=source_ann_path,
            target_label=label_name,
        )

        # Annotation filename must match the copied image filename + ".json"
        destination_ann_path = output_ann_dir / f"{destination_image_path.name}.json"

        with open(destination_ann_path, "w") as f:
            json.dump(annotation_data, f, indent=2)

        rows.append({
            "original_image_path": str(image_path),
            "original_ann_path": str(source_ann_path),
            "label_idx": label_idx,
            "label_name": label_name,
            "exported_image_path": str(destination_image_path),
            "exported_ann_path": str(destination_ann_path),
            "exported_image_filename": destination_image_path.name,
        })

    return rows


def save_labels_csv(rows, output_root: Path):
    """
    Save labels.csv.
    """
    csv_path = output_root / "labels.csv"

    fieldnames = [
        "original_image_path",
        "original_ann_path",
        "label_idx",
        "label_name",
        "exported_image_path",
        "exported_ann_path",
        "exported_image_filename",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows, original_test_counts, n_per_class, output_root: Path):
    """
    Save readable summary.txt.
    """
    exported_counts = Counter(row["label_name"] for row in rows)

    lines = []
    lines.append("Balanced held-out test dataset summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"DATA_ROOT: {config.DATA_ROOT}")
    lines.append(f"IMG_DIR:   {config.IMG_DIR}")
    lines.append(f"ANN_DIR:   {config.ANN_DIR}")
    lines.append("")
    lines.append("Split settings:")
    lines.append(f"  SEED:       {config.SEED}")
    lines.append(f"  TRAIN_FRAC: {config.TRAIN_FRAC}")
    lines.append(f"  VAL_FRAC:   {config.VAL_FRAC}")
    lines.append(f"  TEST_FRAC:  {config.TEST_FRAC}")
    lines.append("")
    lines.append("Original test split class counts:")
    for class_idx in range(config.NUM_CLASSES):
        class_name = config.IDX_TO_CLASS[class_idx]
        count = original_test_counts.get(class_idx, 0)
        lines.append(f"  {class_name}: {count}")
    lines.append("")
    lines.append(f"Balanced samples per class: {n_per_class}")
    lines.append(f"Total exported images: {len(rows)}")
    lines.append("")
    lines.append("Exported class counts:")
    for class_name in config.CLASS_NAMES:
        lines.append(f"  {class_name}: {exported_counts.get(class_name, 0)}")
    lines.append("")
    lines.append("Output:")
    lines.append(f"  {output_root}")
    lines.append(f"  {output_root / 'img'}")
    lines.append(f"  {output_root / 'ann'}")
    lines.append(f"  {output_root / 'labels.csv'}")

    summary_path = output_root / "summary.txt"

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print()
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Optional fixed number of samples per class. Default uses smallest test class count.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate data/test if it already exists.",
    )

    args = parser.parse_args()

    output_root = get_output_root()

    print("Recreating original train/val/test split...")
    samples = discover_samples()
    train_split, val_split, test_split = split_samples(samples)

    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_split)}")
    print(f"Val samples:   {len(val_split)}")
    print(f"Test samples:  {len(test_split)}")

    selected_samples, original_test_counts, n_per_class = choose_balanced_samples(
        test_split=test_split,
        samples_per_class=args.samples_per_class,
    )

    ensure_clean_output_folder(
        output_root=output_root,
        overwrite=args.overwrite,
    )

    rows = export_samples(
        selected_samples=selected_samples,
        output_root=output_root,
    )

    save_labels_csv(rows, output_root)
    save_summary(
        rows=rows,
        original_test_counts=original_test_counts,
        n_per_class=n_per_class,
        output_root=output_root,
    )

    print()
    print("Done.")
    print("This exported dataset comes only from the official held-out test split.")
    print("No training images are copied if the original split settings are unchanged.")


if __name__ == "__main__":
    main()
