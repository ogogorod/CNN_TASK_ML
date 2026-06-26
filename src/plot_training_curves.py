"""
Plot training and validation curves for all trained models.

This script reads history JSON files from:
    outputs/reports/
        scratch_history.json
        transfer_history.json
        mobilenetv2_history.json

It saves figures into:
    outputs/figures/training_curves/
        scratch_accuracy_curve.png
        scratch_loss_curve.png
        transfer_accuracy_curve.png
        transfer_loss_curve.png
        mobilenetv2_accuracy_curve.png
        mobilenetv2_loss_curve.png

It also saves combined comparison curves:
    outputs/figures/training_curves/all_models_val_accuracy.png
    outputs/figures/training_curves/all_models_val_loss.png

This script does NOT train anything.
It only reads saved history files and creates plots.

Run from project root:
    cd /Users/oleg/Desktop/Claude_cnn/weapon_classifier
    python src/plot_training_curves.py
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

import config


MODEL_NAMES = ["scratch", "transfer", "mobilenetv2"]


def get_history_path(model_name: str) -> Path:
    """
    Return path to model history JSON.
    """
    return Path(config.REPORT_DIR) / f"{model_name}_history.json"


def get_output_dir() -> Path:
    """
    Return output directory for training curve figures.
    """
    output_dir = Path(config.OUTPUT_DIR) / "figures" / "training_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_history(model_name: str) -> pd.DataFrame:
    """
    Load training history JSON as a pandas DataFrame.

    Expected columns:
        epoch
        train_loss
        train_acc
        val_loss
        val_acc
        lr
    """
    history_path = get_history_path(model_name)

    if not history_path.exists():
        raise FileNotFoundError(
            f"Missing history file for {model_name}: {history_path}\n"
            "Make sure the file exists in outputs/reports/."
        )

    with open(history_path, "r") as f:
        history = json.load(f)

    if not history:
        raise ValueError(f"History file is empty: {history_path}")

    df = pd.DataFrame(history)

    required_columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"History file {history_path} is missing required column: {column}"
            )

    return df


def plot_accuracy_curve(model_name: str, history_df: pd.DataFrame, output_dir: Path):
    """
    Save train vs validation accuracy curve for one model.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(history_df["epoch"], history_df["train_acc"], marker="o", label="Train accuracy")
    ax.plot(history_df["epoch"], history_df["val_acc"], marker="o", label="Validation accuracy")

    ax.set_title(f"Accuracy Curve - {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    save_path = output_dir / f"{model_name}_accuracy_curve.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss_curve(model_name: str, history_df: pd.DataFrame, output_dir: Path):
    """
    Save train vs validation loss curve for one model.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation loss")

    ax.set_title(f"Loss Curve - {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    save_path = output_dir / f"{model_name}_loss_curve.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_combined_validation_accuracy(histories: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Save one plot comparing validation accuracy across models.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, history_df in histories.items():
        ax.plot(
            history_df["epoch"],
            history_df["val_acc"],
            marker="o",
            label=model_name,
        )

    ax.set_title("Validation Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    save_path = output_dir / "all_models_val_accuracy.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_combined_validation_loss(histories: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Save one plot comparing validation loss across models.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, history_df in histories.items():
        ax.plot(
            history_df["epoch"],
            history_df["val_loss"],
            marker="o",
            label=model_name,
        )

    ax.set_title("Validation Loss Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    save_path = output_dir / "all_models_val_loss.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_training_summary(histories: Dict[str, pd.DataFrame]):
    """
    Save a small CSV summary with best validation metrics.
    """
    rows = []

    for model_name, history_df in histories.items():
        best_val_acc_idx = history_df["val_acc"].idxmax()
        best_val_loss_idx = history_df["val_loss"].idxmin()

        best_val_acc_row = history_df.loc[best_val_acc_idx]
        best_val_loss_row = history_df.loc[best_val_loss_idx]

        rows.append({
            "model": model_name,
            "epochs_recorded": int(len(history_df)),
            "best_val_acc": float(best_val_acc_row["val_acc"]),
            "best_val_acc_epoch": int(best_val_acc_row["epoch"]),
            "best_val_loss": float(best_val_loss_row["val_loss"]),
            "best_val_loss_epoch": int(best_val_loss_row["epoch"]),
            "final_train_acc": float(history_df.iloc[-1]["train_acc"]),
            "final_val_acc": float(history_df.iloc[-1]["val_acc"]),
            "final_train_loss": float(history_df.iloc[-1]["train_loss"]),
            "final_val_loss": float(history_df.iloc[-1]["val_loss"]),
        })

    summary_df = pd.DataFrame(rows)

    reports_dir = Path(config.REPORT_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_dir / "training_curve_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print()
    print("Training curve summary:")
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved summary to: {summary_path}")


def main():
    output_dir = get_output_dir()

    histories = {}

    for model_name in MODEL_NAMES:
        print(f"Loading history for: {model_name}")

        history_df = load_history(model_name)
        histories[model_name] = history_df

        plot_accuracy_curve(model_name, history_df, output_dir)
        plot_loss_curve(model_name, history_df, output_dir)

        print(f"Saved curves for: {model_name}")

    plot_combined_validation_accuracy(histories, output_dir)
    plot_combined_validation_loss(histories, output_dir)
    save_training_summary(histories)

    print()
    print("Training curve plots finished.")
    print(f"Output folder: {output_dir}")
    print("This script did not train or overwrite any checkpoint.")


if __name__ == "__main__":
    main()
