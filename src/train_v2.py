"""
Train one model and save the best checkpoint.

This is a safer unified training script for:
    - scratch
    - transfer
    - mobilenetv2

Important safety behavior:
    If a checkpoint already exists, this script will NOT overwrite it
    unless you pass --overwrite.

Examples:
    python train_v2.py --model scratch
    python train_v2.py --model transfer
    python train_v2.py --model mobilenetv2

Force retraining and overwrite old checkpoint:
    python train_v2.py --model mobilenetv2 --overwrite

This script imports architectures from models_v2.py.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import discover_samples, split_samples, print_split_summary, WeaponDataset
from transforms import train_transform, eval_transform
from models_v2 import get_model
from utils import (
    set_seed,
    get_device,
    compute_class_weights,
    ensure_dirs,
    save_json,
    EarlyStopper,
)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
    Run one training or validation epoch.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_seen += images.size(0)

    avg_loss = total_loss / total_seen
    accuracy = total_correct / total_seen

    return avg_loss, accuracy


def get_default_epochs(model_kind: str) -> int:
    """
    Return default epoch count for each model.
    """
    if model_kind == "scratch":
        return config.EPOCHS_SCRATCH

    return config.EPOCHS_TRANSFER


def get_default_lr(model_kind: str) -> float:
    """
    Return default learning rate for each model.
    """
    if model_kind == "scratch":
        return config.LR_SCRATCH

    return config.LR_TRANSFER


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["scratch", "transfer", "mobilenetv2"],
        required=True,
        help="Which model to train.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional custom number of epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional custom learning rate.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing checkpoint.",
    )

    args = parser.parse_args()

    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    ensure_dirs(config.CHECKPOINT_DIR, config.REPORT_DIR)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{args.model}_best.pt")
    history_path = os.path.join(config.REPORT_DIR, f"{args.model}_history.json")

    if os.path.exists(ckpt_path) and not args.overwrite:
        print()
        print("Checkpoint already exists:")
        print(f"  {ckpt_path}")
        print()
        print("Training was stopped to avoid overwriting saved weights.")
        print("Use --overwrite only if you really want to retrain this model.")
        return

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    samples = discover_samples()
    print(f"Discovered {len(samples)} labeled samples.")

    train_split, val_split, test_split = split_samples(samples)

    print_split_summary("Train", train_split)
    print_split_summary("Val", val_split)
    print_split_summary("Test", test_split)

    train_ds = WeaponDataset(train_split, transform=train_transform)
    val_ds = WeaponDataset(val_split, transform=eval_transform)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------
    # Model / loss / optimizer
    # ------------------------------------------------------------------
    model = get_model(args.model).to(device)

    train_labels = [label for _, label in train_split]
    class_weights = compute_class_weights(train_labels, config.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = args.lr if args.lr is not None else get_default_lr(args.model)
    epochs = args.epochs if args.epochs is not None else get_default_epochs(args.model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    stopper = EarlyStopper(
        patience=config.EARLY_STOP_PATIENCE,
        mode="max",
    )

    history = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
        )

        scheduler.step(val_acc)

        elapsed = time.time() - start_time

        print(
            f"[{args.model}] epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        is_best = stopper.step(val_acc)

        if is_best:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kind": args.model,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

        if stopper.should_stop:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)."
            )
            break

    save_json(history, history_path)

    print()
    print(f"Best val_acc for {args.model}: {stopper.best:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
