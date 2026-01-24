# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models as tv_models

CLASS_NAMES_6: Tuple[str, ...] = ("angry", "disgust", "fear", "happy", "sad", "surprise")
NUM_CLASSES_DEFAULT = 6


@dataclass
class ModelSpec:
    model: nn.Module
    input_size: int
    num_classes: int
    class_names: Tuple[str, ...]


def _resnet18_from_scratch(num_classes: int) -> nn.Module:
    model = tv_models.resnet18(weights=None)  # from scratch
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES_DEFAULT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


def build_model(
    name: str = "resnet18",
    num_classes: int = NUM_CLASSES_DEFAULT,
    *,
    class_names: Tuple[str, ...] = CLASS_NAMES_6,
) -> ModelSpec:
    name = name.lower()

    if len(class_names) != num_classes:
        raise ValueError(f"class_names length ({len(class_names)}) != num_classes ({num_classes})")

    if name == "resnet18":
        return ModelSpec(
            model=_resnet18_from_scratch(num_classes),
            input_size=64,
            num_classes=num_classes,
            class_names=class_names,
        )

    if name == "simple_cnn":
        return ModelSpec(
            model=SimpleCNN(num_classes),
            input_size=64,
            num_classes=num_classes,
            class_names=class_names,
        )

    raise ValueError(f"Unknown model name: {name}")


def save_checkpoint(
    path: str,
    model: nn.Module,
    *,
    model_name: str,
    num_classes: int,
    class_names: Tuple[str, ...],
    epoch: int,
    best_val_acc: float,
    extra: Optional[Dict] = None,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(model: nn.Module, ckpt_path: str, *, map_location="cpu") -> Dict:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return ckpt
    model.load_state_dict(ckpt, strict=True)
    return {}
