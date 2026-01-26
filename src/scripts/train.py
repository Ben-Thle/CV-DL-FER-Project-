# scripts/train.py
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.models.model import build_model


@dataclass
class Metrics:
    loss: float
    acc: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, dl, criterion, optimizer, device) -> Metrics:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in tqdm(dl, leave=False):
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

    return Metrics(loss=total_loss / total, acc=total_correct / total)


def save_ckpt(path: str, model: nn.Module, epoch: int, best_val_acc: float, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            **meta,
        },
        path,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/split_data")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--num-classes", type=int, default=6)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="checkpoints/best.pt")
    p.add_argument("--small-input", action="store_true", help="recommended for 64x64")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "val")

    # Minimal augmentation for faces
    train_tf = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # safe even if already 64
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=val_tf)

    # Ensure class count matches folder structure
    if len(train_ds.classes) != args.num_classes:
        raise ValueError(f"Found {len(train_ds.classes)} classes in folders, but --num-classes={args.num_classes}. "
                         f"Folders: {train_ds.classes}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=max(256, args.batch), shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    

    model = build_model(args.model, num_classes=args.num_classes, small_input=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = -1.0
    meta = {
        "model_name": args.model,
        "num_classes": args.num_classes,
        "class_names": train_ds.classes,
        "small_input": bool(args.small_input),
    }

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, criterion, optimizer, device)
        va = run_epoch(model, val_dl, criterion, None, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train acc {tr.acc:.3f} loss {tr.loss:.3f} | "
            f"val acc {va.acc:.3f} loss {va.loss:.3f}"
        )

        if va.acc > best_val:
            best_val = va.acc
            save_ckpt(args.out, model, epoch, best_val, meta)
            print(f"  ✓ saved best checkpoint to {args.out} (val acc {best_val:.3f})")


if __name__ == "__main__":
    main()