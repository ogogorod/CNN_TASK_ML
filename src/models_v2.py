"""
Model definitions for the weapon classification project.

This file centralizes all model architectures in one place.

Models:
    - scratch: custom CNN trained from random initialization
    - transfer: ResNet18 pretrained on ImageNet
    - mobilenetv2: MobileNetV2 pretrained on ImageNet

All models output config.NUM_CLASSES logits and are compatible with:
    outputs/checkpoints/scratch_best.pt
    outputs/checkpoints/transfer_best.pt
    outputs/checkpoints/mobilenetv2_best.pt
"""

import torch.nn as nn
from torchvision import models

import config


class ScratchCNN(nn.Module):
    """
    Compact CNN trained from scratch.

    This architecture matches the original ScratchCNN used during training.
    Do not change it if you want to load the existing scratch_best.pt checkpoint.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),     # 224 -> 112
            block(32, 64),    # 112 -> 56
            block(64, 128),   # 56  -> 28
            block(128, 256),  # 28  -> 14
            block(256, 256),  # 14  -> 7
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_transfer_model(
    num_classes: int = config.NUM_CLASSES,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build ResNet18 transfer-learning model.

    This architecture matches the original transfer model used during training.
    Do not change it if you want to load transfer_best.pt.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenetv2(
    num_classes: int = config.NUM_CLASSES,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build MobileNetV2 transfer-learning model.

    This architecture matches the MobileNetV2 architecture used in
    train_mobilenetv2.py when mobilenetv2_best.pt was created.
    Do not change it if you want to load mobilenetv2_best.pt.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def get_model(kind: str) -> nn.Module:
    """
    Return model architecture by name.

    Supported:
        scratch
        transfer
        mobilenetv2
    """
    if kind == "scratch":
        return ScratchCNN()

    if kind == "transfer":
        return build_transfer_model()

    if kind == "mobilenetv2":
        return build_mobilenetv2()

    raise ValueError(
        f"Unknown model kind: {kind!r}. "
        "Use 'scratch', 'transfer', or 'mobilenetv2'."
    )
