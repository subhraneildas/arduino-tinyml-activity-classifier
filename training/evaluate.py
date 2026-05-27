#!/usr/bin/env python3
"""
Evaluate a trained HARCNN checkpoint on the UCI HAR test split.

Usage
-----
python training/evaluate.py \\
    --data-dir  "data/UCI HAR Dataset" \\
    --checkpoint models/har_cnn_best.pt \\
    --output-dir models/

Outputs
-------
models/confusion_matrix.png    — normalised confusion matrix heatmap
models/eval_report.json        — per-class precision/recall/F1 + overall accuracy

Console output
--------------
Per-class classification report, overall accuracy, and model size info.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import ACTIVITY_LABELS, N_CLASSES, load_uci_har
from training.model import HARCNN


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model and normalization params from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_params = ckpt.get("norm_params")

    model = HARCNN(n_channels=6, n_classes=N_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Trained for {ckpt.get('epoch', '?')} epochs")
    print(f"  Saved test acc: {ckpt.get('test_acc', 0):.2%}")
    return model, norm_params


@torch.no_grad()
def predict_all(
    model: HARCNN,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple:
    """Run model on full dataset in batches. Returns (y_pred, y_prob)."""
    X_t = torch.from_numpy(X)   # (n, 6, 128)
    all_probs = []

    for i in range(0, len(X_t), batch_size):
        batch = X_t[i : i + batch_size].to(device)
        logits = model(batch)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    y_prob = np.concatenate(all_probs, axis=0)   # (n, 6)
    y_pred = y_prob.argmax(axis=-1)               # (n,)
    return y_pred, y_prob


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    output_path: str,
) -> None:
    """Save a normalised confusion matrix as a PNG."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("UCI HAR — Normalised Confusion Matrix", fontsize=13)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HARCNN on UCI HAR test split")
    p.add_argument("--data-dir",    required=True, help="Path to 'UCI HAR Dataset'")
    p.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    p.add_argument("--output-dir",  default="models/")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--device",      default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load model
    model, norm_params = load_checkpoint(args.checkpoint, device)
    print(f"Model parameters: {model.num_parameters:,}")

    # ── Load test data (apply same normalization as training)
    _, _, X_test, y_test, norm_params = load_uci_har(
        args.data_dir, normalize=True, norm_params=norm_params
    )
    print(f"\nTest samples: {len(y_test):,}")

    # ── Inference
    y_pred, y_prob = predict_all(model, X_test, args.batch_size, device)
    accuracy = (y_pred == y_test).mean()

    # ── Report
    print(f"\nOverall Test Accuracy: {accuracy:.2%}")
    print("\nPer-class Report:")
    report_str = classification_report(
        y_test, y_pred,
        target_names=ACTIVITY_LABELS,
        digits=4,
    )
    print(report_str)

    # ── Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        labels=[lbl[:12] for lbl in ACTIVITY_LABELS],   # shorten for display
        output_path=str(out_dir / "confusion_matrix.png"),
    )

    # ── Save JSON report
    from sklearn.metrics import classification_report as cr_dict
    report_dict = classification_report(
        y_test, y_pred,
        target_names=ACTIVITY_LABELS,
        output_dict=True,
    )
    report_dict["overall_accuracy"] = float(accuracy)
    report_dict["model_parameters"] = model.num_parameters

    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved eval report -> {out_dir / 'eval_report.json'}")

    # ── Per-class confidence stats
    print("\nPer-class mean confidence (correct predictions):")
    for cls_id, cls_name in enumerate(ACTIVITY_LABELS):
        mask = (y_test == cls_id) & (y_pred == cls_id)
        if mask.sum() > 0:
            conf = y_prob[mask, cls_id].mean()
            print(f"  {cls_name:<22} {conf:.3f}")


if __name__ == "__main__":
    main()
