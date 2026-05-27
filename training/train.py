#!/usr/bin/env python3
"""
Train the HARCNN on the UCI HAR dataset.

Usage
-----
python training/train.py \\
    --data-dir "data/UCI HAR Dataset" \\
    --epochs 50 \\
    --batch-size 64 \\
    --lr 1e-3 \\
    --output-dir models/

Outputs
-------
models/har_cnn_best.pt          — best checkpoint (by test accuracy)
models/har_cnn_last.pt          — final epoch checkpoint
models/normalization_params.json — mean/std per channel (6 values)
models/training_history.json    — per-epoch loss/accuracy
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local imports — ensure script can be run from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import (
    ACTIVITY_LABELS, N_CLASSES, get_dataloaders, save_norm_params
)
from training.model import HARCNN


# ── Training helpers ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model. Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HARCNN on UCI HAR")
    p.add_argument("--data-dir",    required=True,  help="Path to 'UCI HAR Dataset'")
    p.add_argument("--output-dir",  default="models/", help="Where to save checkpoints")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--dropout",     type=float, default=0.5)
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="auto",
                   help="'cpu', 'cuda', 'mps', or 'auto'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility
    torch.manual_seed(args.seed)

    # ── Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── Data
    print("Loading UCI HAR dataset ...")
    train_loader, test_loader, norm_params = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
    )
    save_norm_params(norm_params, str(out_dir / "normalization_params.json"))

    print(f"  Train: {len(train_loader.dataset):,} windows")
    print(f"  Test:  {len(test_loader.dataset):,}  windows")
    print(f"  Channels: {norm_params['channels']}")
    print(f"  Mean:  {[f'{v:.4f}' for v in norm_params['mean']]}")
    print(f"  Std:   {[f'{v:.4f}' for v in norm_params['std']]}")

    # ── Model
    model = HARCNN(n_channels=6, n_classes=N_CLASSES, dropout=args.dropout)
    model = model.to(device)
    print(f"\nModel: {model.num_parameters:,} parameters")

    # ── Optimizer + scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop
    best_acc = 0.0
    history: List[Dict] = []

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Test Loss':>9}  {'Test Acc':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc, 5),
            "test_loss":  round(test_loss, 5),
            "test_acc":   round(test_acc, 5),
            "lr":         lr_now,
        })

        marker = " *" if test_acc > best_acc else ""
        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{test_loss:>9.4f}  {test_acc:>7.2%}  {lr_now:>8.2e}  "
            f"{elapsed:>5.1f}s{marker}"
        )

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_acc": test_acc,
                    "norm_params": norm_params,
                    "args": vars(args),
                },
                out_dir / "har_cnn_best.pt",
            )

    # Save final checkpoint
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
            "norm_params": norm_params,
            "args": vars(args),
        },
        out_dir / "har_cnn_last.pt",
    )

    # Save training history
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.2%}")
    print(f"Checkpoints saved to: {out_dir}")
    print(f"  Best model: {out_dir / 'har_cnn_best.pt'}")
    print(f"  Norm params: {out_dir / 'normalization_params.json'}")
    print("\nNext: python training/evaluate.py --checkpoint models/har_cnn_best.pt ...")


if __name__ == "__main__":
    main()
