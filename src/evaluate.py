"""
Evaluate trained weapon-classification models on data/test.

Models evaluated:
    - scratch
    - transfer
    - mobilenetv2

Input:
    data/test/labels.csv
    data/test/img/

Output folders:
    outputs/reports/
    outputs/predictions/
    outputs/figures/confusion_matrices/
    outputs/figures/roc_curves/
    outputs/figures/precision_recall_curves/

This script does NOT train anything.
It only loads existing checkpoints and runs inference.

Run from project root:
    cd /Users/oleg/Desktop/Claude_cnn/weapon_classifier
    python src/evaluate.py

Optional quick test:
    python src/evaluate.py --limit 30

Optional selected models:
    python src/evaluate.py --models scratch transfer
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

import config
from infer import load_model, predict_image, get_checkpoint_path
from utils import get_device


MODEL_NAMES = ["scratch", "transfer", "mobilenetv2"]
BINARY_LABELS = ["non_weapon", "weapon"]


# ---------------------------------------------------------------------------
# Paths and folders
# ---------------------------------------------------------------------------

def make_output_dirs() -> Dict[str, Path]:
    """
    Create output folders for reports, predictions, and figures.
    """
    output_root = Path(config.OUTPUT_DIR)

    folders = {
        "reports": Path(config.REPORT_DIR),
        "predictions": output_root / "predictions",
        "confusion_matrices": output_root / "figures" / "confusion_matrices",
        "roc_curves": output_root / "figures" / "roc_curves",
        "pr_curves": output_root / "figures" / "precision_recall_curves",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def get_test_labels_path() -> Path:
    """
    labels.csv created by export_test_dataset.py.
    """
    return Path(config.PROJECT_ROOT) / "data" / "test" / "labels.csv"


def get_test_img_dir() -> Path:
    """
    Image folder created by export_test_dataset.py.
    """
    return Path(config.PROJECT_ROOT) / "data" / "test" / "img"


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def class_to_binary(class_name: str) -> str:
    """
    Convert multiclass label to binary label.
    """
    if class_name in config.WEAPON_CLASSES:
        return "weapon"
    return "non_weapon"


def binary_to_index(binary_label: str) -> int:
    """
    Convert binary label to numeric index.
    non_weapon -> 0
    weapon     -> 1
    """
    if binary_label == "weapon":
        return 1
    return 0


# ---------------------------------------------------------------------------
# Test dataset loading
# ---------------------------------------------------------------------------

def load_test_rows(limit: int | None = None) -> List[Dict[str, Any]]:
    """
    Load exported test dataset rows from data/test/labels.csv.

    This function reconstructs the image path from:
        data/test/img/<exported_image_filename>

    This avoids problems if labels.csv contains old absolute paths.
    """
    labels_path = get_test_labels_path()
    image_dir = get_test_img_dir()

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing test labels file: {labels_path}\n"
            "Run first:\n"
            "  python src/export_test_dataset.py"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing test image folder: {image_dir}\n"
            "Run first:\n"
            "  python src/export_test_dataset.py"
        )

    rows = []

    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_filename = row.get("exported_image_filename")
            true_class = row.get("label_name")

            if not image_filename or not true_class:
                raise ValueError(
                    "labels.csv must contain columns: "
                    "exported_image_filename and label_name"
                )

            if true_class not in config.CLASS_NAMES:
                raise ValueError(f"Unknown class in labels.csv: {true_class}")

            image_path = image_dir / image_filename

            if not image_path.exists():
                raise FileNotFoundError(f"Image listed in labels.csv not found: {image_path}")

            rows.append({
                "image_path": str(image_path),
                "image_filename": image_filename,
                "true_class": true_class,
                "true_index": config.CLASS_TO_IDX[true_class],
                "true_binary": class_to_binary(true_class),
            })

            if limit is not None and len(rows) >= limit:
                break

    return rows


# ---------------------------------------------------------------------------
# Timing and efficiency helpers
# ---------------------------------------------------------------------------

def synchronize_device(device) -> None:
    """
    Synchronize CUDA or MPS for more accurate timing.
    CPU does not need synchronization.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def count_parameters(model) -> int:
    """
    Count total model parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def get_model_size_mb(model_name: str) -> float:
    """
    Checkpoint file size in MB.
    """
    checkpoint_path = get_checkpoint_path(model_name)

    if not os.path.exists(checkpoint_path):
        return float("nan")

    size_bytes = os.path.getsize(checkpoint_path)
    return size_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_one_model(
    model_name: str,
    test_rows: List[Dict[str, Any]],
    device,
    folders: Dict[str, Path],
) -> Dict[str, Any]:
    """
    Evaluate one model and save all model-specific outputs.
    """
    print()
    print("=" * 80)
    print(f"Evaluating model: {model_name}")
    print("=" * 80)

    load_start = time.perf_counter()
    model = load_model(model_name, device)
    synchronize_device(device)
    load_end = time.perf_counter()

    model_load_time_seconds = load_end - load_start
    parameter_count = count_parameters(model)
    model_size_mb = get_model_size_mb(model_name)

    prediction_rows = []

    for row in test_rows:
        image_path = row["image_path"]

        synchronize_device(device)
        start = time.perf_counter()

        result = predict_image(model, image_path, device)

        synchronize_device(device)
        end = time.perf_counter()

        inference_time_ms = (end - start) * 1000

        predicted_class = result["predicted_class"]
        predicted_binary = class_to_binary(predicted_class)

        prediction_row = {
            "model": model_name,
            "image_path": image_path,
            "image_filename": row["image_filename"],

            "true_class": row["true_class"],
            "predicted_class": predicted_class,
            "correct": row["true_class"] == predicted_class,

            "true_index": row["true_index"],
            "predicted_index": result["predicted_index"],

            "confidence": result["confidence"],

            "true_binary": row["true_binary"],
            "predicted_binary": predicted_binary,
            "binary_correct": row["true_binary"] == predicted_binary,

            "weapon_score": result["weapon_score"],
            "inference_time_ms": inference_time_ms,
        }

        for class_name in config.CLASS_NAMES:
            prediction_row[f"{class_name}_prob"] = result["class_probs"][class_name]

        prediction_rows.append(prediction_row)

    predictions_df = pd.DataFrame(prediction_rows)

    metrics = compute_metrics(
        model_name=model_name,
        predictions_df=predictions_df,
        model_load_time_seconds=model_load_time_seconds,
        parameter_count=parameter_count,
        model_size_mb=model_size_mb,
    )

    save_prediction_files(model_name, predictions_df, folders)
    save_classification_report(model_name, predictions_df, folders)
    save_confusion_matrices(model_name, predictions_df, folders)
    save_roc_curve(model_name, predictions_df, folders)
    save_precision_recall_curve(model_name, predictions_df, folders)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Binary ROC-AUC: {metrics['binary_roc_auc']:.4f}")
    print(f"Binary PR-AUC: {metrics['binary_pr_auc']:.4f}")
    print(f"Average inference time: {metrics['avg_inference_time_ms']:.2f} ms/image")
    print(f"Saved outputs for: {model_name}")

    return metrics


def compute_metrics(
    model_name: str,
    predictions_df: pd.DataFrame,
    model_load_time_seconds: float,
    parameter_count: int,
    model_size_mb: float,
) -> Dict[str, Any]:
    """
    Compute multiclass, binary weapon/no-weapon, error, and efficiency metrics.
    """
    y_true = predictions_df["true_class"].tolist()
    y_pred = predictions_df["predicted_class"].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=config.CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=config.CLASS_NAMES,
        average="weighted",
        zero_division=0,
    )

    # Binary weapon / non-weapon metrics
    y_true_binary = predictions_df["true_binary"].tolist()
    y_pred_binary = predictions_df["predicted_binary"].tolist()

    y_true_binary_idx = [binary_to_index(label) for label in y_true_binary]
    y_pred_binary_idx = [binary_to_index(label) for label in y_pred_binary]

    binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)

    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        labels=BINARY_LABELS,
        average="binary",
        pos_label="weapon",
        zero_division=0,
    )

    weapon_scores = predictions_df["weapon_score"].tolist()

    try:
        binary_roc_auc = roc_auc_score(y_true_binary_idx, weapon_scores)
    except ValueError:
        binary_roc_auc = float("nan")

    try:
        binary_pr_auc = average_precision_score(y_true_binary_idx, weapon_scores)
    except ValueError:
        binary_pr_auc = float("nan")

    # Errors
    total_images = len(predictions_df)
    total_errors = int((predictions_df["correct"] == False).sum())
    multiclass_error_rate = total_errors / total_images if total_images > 0 else float("nan")

    binary_errors = int((predictions_df["binary_correct"] == False).sum())

    false_positives = int(
        (
            (predictions_df["true_binary"] == "non_weapon")
            & (predictions_df["predicted_binary"] == "weapon")
        ).sum()
    )

    false_negatives = int(
        (
            (predictions_df["true_binary"] == "weapon")
            & (predictions_df["predicted_binary"] == "non_weapon")
        ).sum()
    )

    # Efficiency
    total_inference_time_seconds = predictions_df["inference_time_ms"].sum() / 1000
    avg_inference_time_ms = predictions_df["inference_time_ms"].mean()

    if total_inference_time_seconds > 0:
        images_per_second = total_images / total_inference_time_seconds
    else:
        images_per_second = float("nan")

    # Per-class report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=config.CLASS_NAMES,
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}

    for class_name in config.CLASS_NAMES:
        per_class[class_name] = {
            "precision": float(report_dict[class_name]["precision"]),
            "recall": float(report_dict[class_name]["recall"]),
            "f1_score": float(report_dict[class_name]["f1-score"]),
            "support": int(report_dict[class_name]["support"]),
        }

    # Confusion matrices for JSON
    multiclass_cm = confusion_matrix(
        y_true,
        y_pred,
        labels=config.CLASS_NAMES,
    )

    binary_cm = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=BINARY_LABELS,
    )

    metrics = {
        "model": model_name,

        # Multiclass metrics
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),

        # Binary weapon/no-weapon metrics
        "binary_accuracy": float(binary_accuracy),
        "binary_precision": float(binary_precision),
        "binary_recall": float(binary_recall),
        "binary_f1": float(binary_f1),
        "binary_roc_auc": float(binary_roc_auc),
        "binary_pr_auc": float(binary_pr_auc),

        # Error metrics
        "total_images": int(total_images),
        "total_errors": int(total_errors),
        "multiclass_error_rate": float(multiclass_error_rate),
        "binary_errors": int(binary_errors),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),

        # Efficiency metrics
        "model_load_time_seconds": float(model_load_time_seconds),
        "total_inference_time_seconds": float(total_inference_time_seconds),
        "avg_inference_time_ms": float(avg_inference_time_ms),
        "images_per_second": float(images_per_second),
        "model_size_mb": float(model_size_mb),
        "parameter_count": int(parameter_count),

        # Detailed nested outputs
        "per_class": per_class,
        "multiclass_confusion_matrix": multiclass_cm.tolist(),
        "binary_confusion_matrix": binary_cm.tolist(),
    }

    return metrics


# ---------------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------------

def save_prediction_files(
    model_name: str,
    predictions_df: pd.DataFrame,
    folders: Dict[str, Path],
) -> None:
    """
    Save prediction-level outputs for error analysis and Grad-CAM selection.
    """
    predictions_path = folders["predictions"] / f"predictions_{model_name}.csv"
    errors_path = folders["predictions"] / f"errors_{model_name}.csv"
    false_positives_path = folders["predictions"] / f"false_positives_{model_name}.csv"
    false_negatives_path = folders["predictions"] / f"false_negatives_{model_name}.csv"

    predictions_df.to_csv(predictions_path, index=False)

    errors_df = predictions_df[predictions_df["correct"] == False].copy()
    errors_df.to_csv(errors_path, index=False)

    false_positives_df = predictions_df[
        (predictions_df["true_binary"] == "non_weapon")
        & (predictions_df["predicted_binary"] == "weapon")
    ].copy()

    false_negatives_df = predictions_df[
        (predictions_df["true_binary"] == "weapon")
        & (predictions_df["predicted_binary"] == "non_weapon")
    ].copy()

    false_positives_df.to_csv(false_positives_path, index=False)
    false_negatives_df.to_csv(false_negatives_path, index=False)


def save_classification_report(
    model_name: str,
    predictions_df: pd.DataFrame,
    folders: Dict[str, Path],
) -> None:
    """
    Save sklearn-style classification report as CSV.
    """
    y_true = predictions_df["true_class"].tolist()
    y_pred = predictions_df["predicted_class"].tolist()

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=config.CLASS_NAMES,
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_path = folders["reports"] / f"classification_report_{model_name}.csv"
    report_df.to_csv(report_path)


# ---------------------------------------------------------------------------
# Figure outputs
# ---------------------------------------------------------------------------

def save_confusion_matrices(
    model_name: str,
    predictions_df: pd.DataFrame,
    folders: Dict[str, Path],
) -> None:
    """
    Save multiclass and binary confusion matrix figures.
    """
    y_true = predictions_df["true_class"].tolist()
    y_pred = predictions_df["predicted_class"].tolist()

    multiclass_cm = confusion_matrix(
        y_true,
        y_pred,
        labels=config.CLASS_NAMES,
    )

    multiclass_path = folders["confusion_matrices"] / f"confusion_matrix_{model_name}.png"

    plot_confusion_matrix(
        cm=multiclass_cm,
        labels=config.CLASS_NAMES,
        title=f"Multiclass Confusion Matrix - {model_name}",
        save_path=multiclass_path,
    )

    y_true_binary = predictions_df["true_binary"].tolist()
    y_pred_binary = predictions_df["predicted_binary"].tolist()

    binary_cm = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=BINARY_LABELS,
    )

    binary_path = folders["confusion_matrices"] / f"binary_confusion_matrix_{model_name}.png"

    plot_confusion_matrix(
        cm=binary_cm,
        labels=BINARY_LABELS,
        title=f"Binary Weapon Confusion Matrix - {model_name}",
        save_path=binary_path,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Path,
) -> None:
    """
    Plot and save a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_roc_curve(
    model_name: str,
    predictions_df: pd.DataFrame,
    folders: Dict[str, Path],
) -> None:
    """
    Save ROC curve for binary weapon detection.

    Positive class:
        weapon

    Score:
        weapon_score = P(knife) + P(pistol)
    """
    y_true_binary_idx = [
        binary_to_index(label)
        for label in predictions_df["true_binary"].tolist()
    ]

    weapon_scores = predictions_df["weapon_score"].tolist()

    try:
        fpr, tpr, thresholds = roc_curve(y_true_binary_idx, weapon_scores)
        auc_value = roc_auc_score(y_true_binary_idx, weapon_scores)
    except ValueError:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

    ax.set_title(f"ROC Curve - {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    fig.tight_layout()

    save_path = folders["roc_curves"] / f"roc_curve_{model_name}.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    curve_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    })

    curve_csv_path = folders["reports"] / f"roc_curve_points_{model_name}.csv"
    curve_df.to_csv(curve_csv_path, index=False)


def save_precision_recall_curve(
    model_name: str,
    predictions_df: pd.DataFrame,
    folders: Dict[str, Path],
) -> None:
    """
    Save precision-recall curve for binary weapon detection.

    Positive class:
        weapon
    """
    y_true_binary_idx = [
        binary_to_index(label)
        for label in predictions_df["true_binary"].tolist()
    ]

    weapon_scores = predictions_df["weapon_score"].tolist()

    try:
        precision, recall, thresholds = precision_recall_curve(
            y_true_binary_idx,
            weapon_scores,
        )
        ap_value = average_precision_score(y_true_binary_idx, weapon_scores)
    except ValueError:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(recall, precision, label=f"AP = {ap_value:.4f}")

    ax.set_title(f"Precision-Recall Curve - {model_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    fig.tight_layout()

    save_path = folders["pr_curves"] / f"pr_curve_{model_name}.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    # precision and recall are one element longer than thresholds
    curve_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": list(thresholds) + [np.nan],
    })

    curve_csv_path = folders["reports"] / f"pr_curve_points_{model_name}.csv"
    curve_df.to_csv(curve_csv_path, index=False)


# ---------------------------------------------------------------------------
# Final summary outputs
# ---------------------------------------------------------------------------

def save_final_results(
    all_metrics: List[Dict[str, Any]],
    folders: Dict[str, Path],
) -> None:
    """
    Save final JSON and CSV comparison tables.
    """
    results_path = folders["reports"] / "evaluation_results.json"

    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    comparison_rows = []

    for metrics in all_metrics:
        row = {
            key: value
            for key, value in metrics.items()
            if key not in [
                "per_class",
                "multiclass_confusion_matrix",
                "binary_confusion_matrix",
            ]
        }

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    preferred_order = [
        "model",
        "total_images",

        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",

        "binary_accuracy",
        "binary_precision",
        "binary_recall",
        "binary_f1",
        "binary_roc_auc",
        "binary_pr_auc",

        "total_errors",
        "multiclass_error_rate",
        "binary_errors",
        "false_positives",
        "false_negatives",

        "avg_inference_time_ms",
        "images_per_second",
        "total_inference_time_seconds",
        "model_load_time_seconds",
        "model_size_mb",
        "parameter_count",
    ]

    existing_columns = [
        column for column in preferred_order
        if column in comparison_df.columns
    ]

    comparison_df = comparison_df[existing_columns]

    comparison_path = folders["reports"] / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    efficiency_columns = [
        "model",
        "avg_inference_time_ms",
        "images_per_second",
        "total_inference_time_seconds",
        "model_load_time_seconds",
        "model_size_mb",
        "parameter_count",
    ]

    efficiency_df = comparison_df[efficiency_columns]
    efficiency_path = folders["reports"] / "efficiency_metrics.csv"
    efficiency_df.to_csv(efficiency_path, index=False)

    print()
    print("=" * 80)
    print("Final model comparison")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    print()
    print(f"Saved final results to: {results_path}")
    print(f"Saved model comparison to: {comparison_path}")
    print(f"Saved efficiency metrics to: {efficiency_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Models to evaluate.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing. Example: --limit 30",
    )

    args = parser.parse_args()

    folders = make_output_dirs()

    device = get_device()
    print(f"Using device: {device}")

    test_rows = load_test_rows(limit=args.limit)

    print(f"Loaded test images: {len(test_rows)}")
    print(f"Test labels file: {get_test_labels_path()}")
    print(f"Test image folder: {get_test_img_dir()}")

    all_metrics = []

    for model_name in args.models:
        metrics = evaluate_one_model(
            model_name=model_name,
            test_rows=test_rows,
            device=device,
            folders=folders,
        )

        all_metrics.append(metrics)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_final_results(all_metrics, folders)

    print()
    print("Evaluation finished.")
    print("This script did not train or overwrite any checkpoint.")


if __name__ == "__main__":
    main()
