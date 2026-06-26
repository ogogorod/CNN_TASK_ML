"""
Inference utilities for the weapon classification project.

This file can be used in two ways:

1. Command-line inference for manual testing:
    python src/infer.py --model scratch --image path/to/image.jpg
    python src/infer.py --model transfer --image path/to/image.jpg
    python src/infer.py --model mobilenetv2 --image path/to/image.jpg

    Folder / glob example:
    python src/infer.py --model mobilenetv2 --image "data/test/img/*.jpg"

2. Importable functions for evaluation scripts:
    from infer import load_model, predict_image, predict_many

    model = load_model("mobilenetv2", device)
    result = predict_image(model, "path/to/image.jpg", device)

This script does NOT train anything.
It only loads saved checkpoints and runs forward passes.

Expected checkpoints:
    outputs/checkpoints/scratch_best.pt
    outputs/checkpoints/transfer_best.pt
    outputs/checkpoints/mobilenetv2_best.pt
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
from PIL import Image

import config
from transforms import eval_transform
from models_v2 import get_model
from utils import get_device


MODEL_NAMES = ["scratch", "transfer", "mobilenetv2"]


def get_checkpoint_path(model_kind: str) -> str:
    """
    Return checkpoint path for a model.

    Example:
        scratch -> outputs/checkpoints/scratch_best.pt
    """
    if model_kind not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model kind: {model_kind!r}. "
            f"Use one of: {MODEL_NAMES}"
        )

    return os.path.join(config.CHECKPOINT_DIR, f"{model_kind}_best.pt")


def load_model(model_kind: str, device=None):
    """
    Load a trained model checkpoint.

    Args:
        model_kind:
            "scratch", "transfer", or "mobilenetv2"

        device:
            torch.device. If None, utils.get_device() is used.

    Returns:
        model in eval mode

    Important:
        This function does NOT train the model.
    """
    if device is None:
        device = get_device()

    checkpoint_path = get_checkpoint_path(model_kind)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Make sure the checkpoint exists in outputs/checkpoints/."
        )

    model = get_model(model_kind).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model


def load_image_tensor(image_path: str, device) -> torch.Tensor:
    """
    Load one image and apply the same evaluation transform used by the project.

    Returns:
        Tensor with shape [1, 3, IMG_SIZE, IMG_SIZE]
    """
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)
    return tensor


@torch.no_grad()
def predict_image(model, image_path: str, device=None) -> Dict[str, Any]:
    """
    Predict one image.

    Args:
        model:
            Loaded PyTorch model.

        image_path:
            Path to image file.

        device:
            torch.device. If None, utils.get_device() is used.

    Returns:
        Dictionary with:
            image
            predicted_class
            predicted_index
            confidence
            is_weapon
            weapon_score
            class_probs
            class_prob_list
    """
    if device is None:
        device = get_device()

    tensor = load_image_tensor(image_path, device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    pred_idx = int(probs.argmax().item())
    pred_class = config.IDX_TO_CLASS[pred_idx]

    weapon_indices = sorted(config.WEAPON_IDXS)
    weapon_score = float(probs[weapon_indices].sum().item())

    class_probs = {
        config.IDX_TO_CLASS[i]: float(probs[i].item())
        for i in range(config.NUM_CLASSES)
    }

    result = {
        "image": image_path,
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx].item()),
        "is_weapon": pred_class in config.WEAPON_CLASSES,
        "weapon_score": weapon_score,
        "class_probs": class_probs,
        "class_prob_list": [float(p.item()) for p in probs],
    }

    return result


def predict_many(model, image_paths: List[str], device=None) -> List[Dict[str, Any]]:
    """
    Predict many images with one loaded model.

    This is intentionally simple and readable.
    Evaluation scripts can use this function, or loop over predict_image()
    themselves if they need timing per image.
    """
    if device is None:
        device = get_device()

    results = []

    for image_path in image_paths:
        result = predict_image(model, image_path, device)
        results.append(result)

    return results


def collect_image_paths(image_arg: str) -> List[str]:
    """
    Accept either:
        - one image path
        - glob pattern, e.g. "data/test/img/*.jpg"
        - folder path

    Returns:
        sorted list of image paths
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    path = Path(image_arg)

    if path.is_dir():
        image_paths = [
            str(p)
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
        return sorted(image_paths)

    if any(char in image_arg for char in "*?[]"):
        return sorted(glob.glob(image_arg))

    return [image_arg]


def print_prediction(result: Dict[str, Any], show_probs: bool = True) -> None:
    """
    Print one prediction in a readable way.
    """
    verdict = "WEAPON" if result["is_weapon"] else "not weapon"

    print()
    print(f"Image: {result['image']}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Binary decision: {verdict}")
    print(f"Weapon score: {result['weapon_score']:.4f}")

    if show_probs:
        print("Class probabilities:")
        for class_name in config.CLASS_NAMES:
            prob = result["class_probs"][class_name]
            print(f"  {class_name:12s}: {prob:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        required=True,
        help="Model checkpoint to use.",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Image path, folder path, or glob pattern.",
    )

    parser.add_argument(
        "--hide-probs",
        action="store_true",
        help="Do not print class probabilities.",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    image_paths = collect_image_paths(args.image)

    if not image_paths:
        print(f"No images found for: {args.image}")
        return

    print(f"Found images: {len(image_paths)}")

    model = load_model(args.model, device)

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Skipping missing file: {image_path}")
            continue

        result = predict_image(model, image_path, device)
        print_prediction(result, show_probs=not args.hide_probs)


if __name__ == "__main__":
    main()
