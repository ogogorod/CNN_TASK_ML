"""
Generate Grad-CAM visualizations for trained weapon-classification models.

Models supported:
    - scratch
    - transfer
    - mobilenetv2

This script reads prediction CSVs created by evaluate.py:
    outputs/predictions/predictions_<model>.csv
    outputs/predictions/false_positives_<model>.csv
    outputs/predictions/false_negatives_<model>.csv

It saves Grad-CAM images into:
    outputs/gradcam/<model>/correct/
    outputs/gradcam/<model>/false_positives/
    outputs/gradcam/<model>/false_negatives/

This script does NOT train anything.
It only loads checkpoints and creates visualization images.

Run from project root:
    cd /Users/oleg/Desktop/Claude_cnn/weapon_classifier
    python src/gradcam.py

Optional:
    python src/gradcam.py --models scratch transfer mobilenetv2
    python src/gradcam.py --num-images 5
    python src/gradcam.py --target predicted
    python src/gradcam.py --target true

Recommended for report:
    python src/gradcam.py --num-images 3
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import config
from infer import load_model
from utils import get_device


MODEL_NAMES = ["scratch", "transfer", "mobilenetv2"]
GROUPS = ["correct", "false_positives", "false_negatives"]


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Simple Grad-CAM implementation.

    It stores:
        - activations from a target convolutional layer
        - gradients from the same layer

    Then it creates a heatmap showing which image regions influenced
    the selected class score.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input_tensor, output_tensor):
        self.activations = output_tensor.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """
        Remove hooks after Grad-CAM is finished.
        """
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, input_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
        """
        Generate one Grad-CAM heatmap.

        Args:
            input_tensor:
                Tensor with shape [1, 3, H, W]

            target_class_idx:
                Class index to explain.

        Returns:
            Heatmap as numpy array with values in [0, 1].
        """
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(input_tensor)
        class_score = outputs[:, target_class_idx].sum()

        class_score.backward()

        if self.activations is None:
            raise RuntimeError("No activations captured. Check target layer.")

        if self.gradients is None:
            raise RuntimeError("No gradients captured. Check target layer.")

        # activations: [1, C, H, W]
        # gradients:   [1, C, H, W]
        activations = self.activations[0]
        gradients = self.gradients[0]

        # Global average pooling of gradients -> channel weights
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=activations.dtype,
            device=activations.device,
        )

        for channel_idx, weight in enumerate(weights):
            cam += weight * activations[channel_idx]

        # Keep only positive influence
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


# ---------------------------------------------------------------------------
# Target layer selection
# ---------------------------------------------------------------------------

def get_target_layer(model, model_name: str):
    """
    Choose the last convolutional block for each model.

    This is the standard target for Grad-CAM because it still contains
    spatial information but is close to the final decision.
    """

    if model_name == "transfer":
        # ResNet18 from torchvision
        if hasattr(model, "layer4"):
            return model.layer4[-1]

    if model_name == "mobilenetv2":
        # MobileNetV2 from torchvision
        if hasattr(model, "features"):
            return model.features[-1]

    if model_name == "scratch":
        # Custom ScratchCNN usually has self.features as nn.Sequential.
        if hasattr(model, "features"):
            return model.features[-1]

    # Fallback: find the last Conv2d layer in the model
    last_conv = None

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError(f"Could not find a convolutional layer for {model_name}.")

    return last_conv


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def build_eval_transform():
    """
    Same basic preprocessing idea as evaluation:
        resize -> tensor -> ImageNet normalization

    This matches common transfer-learning preprocessing and should also work
    for the scratch model if it was trained with the same eval_transform.
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image_for_gradcam(image_path: str, device):
    """
    Load image both as:
        - original RGB numpy image for overlay
        - normalized tensor for model
    """
    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    transform = build_eval_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    return image, original, tensor


def create_overlay(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    """
    Create Grad-CAM overlay.

    Args:
        original_rgb:
            Original image as RGB numpy array.

        heatmap:
            Grad-CAM heatmap in [0, 1].

        alpha:
            Heatmap intensity over original image.

    Returns:
        RGB overlay image as uint8 numpy array.
    """
    original_resized = cv2.resize(
        original_rgb,
        (config.IMG_SIZE, config.IMG_SIZE),
    )

    heatmap_resized = cv2.resize(
        heatmap,
        (config.IMG_SIZE, config.IMG_SIZE),
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_color_rgb = cv2.cvtColor(
        heatmap_color_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay = (
        (1 - alpha) * original_resized.astype(np.float32)
        + alpha * heatmap_color_rgb.astype(np.float32)
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def save_gradcam_image(
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    save_path: Path,
    title: str,
):
    """
    Save side-by-side original image and Grad-CAM overlay.
    """
    import matplotlib.pyplot as plt

    original_resized = cv2.resize(
        original_rgb,
        (config.IMG_SIZE, config.IMG_SIZE),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV selection helpers
# ---------------------------------------------------------------------------

def get_predictions_dir() -> Path:
    return Path(config.OUTPUT_DIR) / "predictions"


def get_gradcam_output_dir() -> Path:
    return Path(config.OUTPUT_DIR) / "gradcam"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """
    Read CSV if it exists; otherwise return an empty dataframe.
    """
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def choose_examples(
    model_name: str,
    group_name: str,
    num_images: int,
) -> pd.DataFrame:
    """
    Select examples for one Grad-CAM group.

    Groups:
        correct:
            from predictions_<model>.csv where correct == True

        false_positives:
            from false_positives_<model>.csv

        false_negatives:
            from false_negatives_<model>.csv

    If false positives or false negatives do not exist, it falls back to
    multiclass mistakes so that you still get visual examples.
    """
    predictions_dir = get_predictions_dir()

    predictions_path = predictions_dir / f"predictions_{model_name}.csv"
    errors_path = predictions_dir / f"errors_{model_name}.csv"
    false_positives_path = predictions_dir / f"false_positives_{model_name}.csv"
    false_negatives_path = predictions_dir / f"false_negatives_{model_name}.csv"

    predictions_df = read_csv_if_exists(predictions_path)
    errors_df = read_csv_if_exists(errors_path)
    false_positives_df = read_csv_if_exists(false_positives_path)
    false_negatives_df = read_csv_if_exists(false_negatives_path)

    if predictions_df.empty:
        raise FileNotFoundError(
            f"No predictions found for {model_name}.\n"
            f"Expected: {predictions_path}\n"
            "Run first:\n"
            "  python src/evaluate.py"
        )

    if group_name == "correct":
        selected = predictions_df[predictions_df["correct"] == True].copy()

    elif group_name == "false_positives":
        selected = false_positives_df.copy()

        if selected.empty:
            print(
                f"No binary false positives for {model_name}. "
                "Falling back to multiclass errors."
            )
            selected = errors_df.copy()

    elif group_name == "false_negatives":
        selected = false_negatives_df.copy()

        if selected.empty:
            print(
                f"No binary false negatives for {model_name}. "
                "Falling back to multiclass errors."
            )
            selected = errors_df.copy()

    else:
        raise ValueError(f"Unknown group: {group_name}")

    if selected.empty:
        print(f"No examples available for {model_name} / {group_name}.")
        return selected

    # Sort by confidence descending.
    # For correct predictions, this gives clear successful examples.
    # For mistakes, this gives confident errors, which are useful for analysis.
    if "confidence" in selected.columns:
        selected = selected.sort_values("confidence", ascending=False)

    selected = selected.head(num_images)

    return selected


# ---------------------------------------------------------------------------
# Main Grad-CAM generation
# ---------------------------------------------------------------------------

def get_target_class_idx(row, target_mode: str) -> int:
    """
    Decide which class Grad-CAM should explain.

    target_mode:
        predicted:
            explain the class that the model predicted.
            This is the default and usually best for understanding model behavior.

        true:
            explain the ground-truth class.
            Useful especially for false negatives to ask:
            "Where would the model look for the true weapon class?"
    """
    if target_mode == "predicted":
        class_name = row["predicted_class"]

    elif target_mode == "true":
        class_name = row["true_class"]

    else:
        raise ValueError(f"Unknown target mode: {target_mode}")

    return config.CLASS_TO_IDX[class_name]


def safe_filename(text: str) -> str:
    """
    Make a safe filename component.
    """
    safe_chars = []

    for char in str(text):
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    return "".join(safe_chars)


def generate_gradcams_for_model(
    model_name: str,
    num_images: int,
    target_mode: str,
    device,
):
    """
    Generate Grad-CAM images for one model.
    """
    print()
    print("=" * 80)
    print(f"Generating Grad-CAMs for model: {model_name}")
    print("=" * 80)

    model = load_model(model_name, device)
    model.eval()

    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    output_root = get_gradcam_output_dir() / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        for group_name in GROUPS:
            group_output_dir = output_root / group_name
            group_output_dir.mkdir(parents=True, exist_ok=True)

            examples = choose_examples(
                model_name=model_name,
                group_name=group_name,
                num_images=num_images,
            )

            if examples.empty:
                continue

            for idx, row in examples.iterrows():
                image_path = row["image_path"]

                if not os.path.exists(image_path):
                    print(f"Skipping missing image: {image_path}")
                    continue

                pil_image, original_rgb, input_tensor = load_image_for_gradcam(
                    image_path=image_path,
                    device=device,
                )

                target_class_idx = get_target_class_idx(
                    row=row,
                    target_mode=target_mode,
                )

                heatmap = gradcam.generate(
                    input_tensor=input_tensor,
                    target_class_idx=target_class_idx,
                )

                overlay_rgb = create_overlay(
                    original_rgb=original_rgb,
                    heatmap=heatmap,
                )

                true_class = row["true_class"]
                predicted_class = row["predicted_class"]
                confidence = row.get("confidence", 0.0)

                title = (
                    f"{model_name} | {group_name} | "
                    f"true={true_class}, pred={predicted_class}, "
                    f"target={config.IDX_TO_CLASS[target_class_idx]}, "
                    f"conf={confidence:.3f}"
                )

                image_filename = Path(image_path).name
                save_name = (
                    f"{group_name}__"
                    f"true-{safe_filename(true_class)}__"
                    f"pred-{safe_filename(predicted_class)}__"
                    f"{safe_filename(image_filename)}.png"
                )

                save_path = group_output_dir / save_name

                save_gradcam_image(
                    original_rgb=original_rgb,
                    overlay_rgb=overlay_rgb,
                    save_path=save_path,
                    title=title,
                )

                print(f"Saved: {save_path}")

    finally:
        gradcam.remove_hooks()

    print(f"Finished Grad-CAMs for: {model_name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Models for which Grad-CAMs should be generated.",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Number of images per group: correct, false positives, false negatives.",
    )

    parser.add_argument(
        "--target",
        choices=["predicted", "true"],
        default="predicted",
        help=(
            "Which class to explain. "
            "'predicted' explains the model decision. "
            "'true' explains the ground-truth class."
        ),
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    predictions_dir = get_predictions_dir()

    if not predictions_dir.exists():
        raise FileNotFoundError(
            f"Predictions folder not found: {predictions_dir}\n"
            "Run first:\n"
            "  python src/evaluate.py"
        )

    for model_name in args.models:
        generate_gradcams_for_model(
            model_name=model_name,
            num_images=args.num_images,
            target_mode=args.target,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("Grad-CAM generation finished.")
    print(f"Output folder: {get_gradcam_output_dir()}")
    print("This script did not train or overwrite any checkpoint.")


if __name__ == "__main__":
    main()
