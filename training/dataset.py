"""
UCI HAR dataset loader for CNN-based Human Activity Recognition.

The raw inertial signals in the dataset are already segmented into
128-sample windows at 50 Hz (2.56 s per window, 50% overlap between
consecutive windows during the original capture).

We load 6 channels:
  [0] body_acc_x  — linear acceleration X (gravity removed), g
  [1] body_acc_y  — linear acceleration Y (gravity removed), g
  [2] body_acc_z  — linear acceleration Z (gravity removed), g
  [3] body_gyro_x — angular velocity X, rad/s
  [4] body_gyro_y — angular velocity Y, rad/s
  [5] body_gyro_z — angular velocity Z, rad/s

(Note: UCI HAR also provides total_acc_{x,y,z} which includes gravity —
 we do not use those to match the on-device gravity-subtracted signal.)

Activity labels (0-indexed):
  0 WALKING  1 WALKING_UPSTAIRS  2 WALKING_DOWNSTAIRS
  3 SITTING  4 STANDING          5 LAYING
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── Constants ──────────────────────────────────────────────────────────────────
ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

N_CLASSES = 6
N_CHANNELS = 6   # body_acc (3) + body_gyro (3)
SEQ_LEN = 128    # timesteps per window @ 50 Hz = 2.56 s
SAMPLE_RATE_HZ = 50


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_raw_signals(data_dir: Path, split: str) -> np.ndarray:
    """
    Load 6-channel inertial signals for `split` ('train' or 'test').

    Returns
    -------
    ndarray of shape (n_windows, SEQ_LEN, N_CHANNELS) — float32
        Channels: [body_acc_x, _y, _z, body_gyro_x, _y, _z]
    """
    sig_dir = data_dir / split / "Inertial Signals"
    if not sig_dir.exists():
        raise FileNotFoundError(
            f"Inertial Signals directory not found: {sig_dir}\n"
            f"Run 'python data/download_uci_har.py' first."
        )

    channels = []
    # Body acceleration — gravity component removed via Butterworth LP filter
    for axis in ("x", "y", "z"):
        fname = sig_dir / f"body_acc_{axis}_{split}.txt"
        channels.append(np.loadtxt(fname, dtype=np.float32))   # (n, 128)

    # Angular velocity in rad/s
    for axis in ("x", "y", "z"):
        fname = sig_dir / f"body_gyro_{axis}_{split}.txt"
        channels.append(np.loadtxt(fname, dtype=np.float32))   # (n, 128)

    # Stack -> (n, 128, 6)
    signals = np.stack(channels, axis=-1)
    return signals


def _load_labels(data_dir: Path, split: str) -> np.ndarray:
    """Load labels, converting from 1-indexed (1..6) to 0-indexed (0..5)."""
    path = data_dir / split / f"y_{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    return np.loadtxt(path, dtype=np.int64) - 1


# ── Public API ─────────────────────────────────────────────────────────────────

def load_uci_har(
    data_dir: str,
    normalize: bool = True,
    norm_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load UCI HAR raw inertial signals and optionally z-score normalize.

    Parameters
    ----------
    data_dir : str
        Path to the 'UCI HAR Dataset' directory.
    normalize : bool
        Apply per-channel z-score normalization.
    norm_params : dict or None
        Pre-computed {'mean': [...], 'std': [...]} (shape 6 each).
        If None and normalize=True, computed from the training split.

    Returns
    -------
    X_train : ndarray (7352, 6, 128) — channels first, for PyTorch Conv1d
    y_train : ndarray (7352,)        — int64, 0..5
    X_test  : ndarray (2947, 6, 128)
    y_test  : ndarray (2947,)
    norm_params : dict  {'mean', 'std', 'channels', 'units'}
    """
    root = Path(data_dir)

    X_train_raw = _load_raw_signals(root, "train")   # (7352, 128, 6)
    y_train = _load_labels(root, "train")             # (7352,)
    X_test_raw  = _load_raw_signals(root, "test")    # (2947, 128, 6)
    y_test  = _load_labels(root, "test")              # (2947,)

    if normalize:
        if norm_params is None:
            # Compute from training set: collapse n & time dims -> shape (6,)
            flat = X_train_raw.reshape(-1, N_CHANNELS)
            mean = flat.mean(axis=0)
            std  = flat.std(axis=0)
            std  = np.where(std < 1e-8, 1.0, std)    # guard zero channels
            norm_params = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "channels": [
                    "body_acc_x", "body_acc_y", "body_acc_z",
                    "body_gyro_x", "body_gyro_y", "body_gyro_z",
                ],
                "units": ["g", "g", "g", "rad/s", "rad/s", "rad/s"],
            }

        mean = np.array(norm_params["mean"], dtype=np.float32)   # (6,)
        std  = np.array(norm_params["std"],  dtype=np.float32)   # (6,)

        # Broadcast: (n, 128, 6) - (6,) / (6,) -> (n, 128, 6)
        X_train_raw = (X_train_raw - mean) / std
        X_test_raw  = (X_test_raw  - mean) / std
    else:
        if norm_params is None:
            norm_params = {"mean": [0.0] * 6, "std": [1.0] * 6,
                           "channels": [], "units": []}

    # Transpose to channels-first: (n, 128, 6) -> (n, 6, 128)
    X_train = X_train_raw.transpose(0, 2, 1)
    X_test  = X_test_raw.transpose(0, 2, 1)

    return X_train, y_train, X_test, y_test, norm_params


def save_norm_params(norm_params: Dict, path: str) -> None:
    """Persist normalization params to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved normalization params -> {path}")


def load_norm_params(path: str) -> Dict:
    """Load normalization params from JSON."""
    with open(path) as f:
        return json.load(f)


# ── PyTorch Dataset / DataLoader ───────────────────────────────────────────────

class HARDataset(Dataset):
    """Lightweight PyTorch Dataset wrapping UCI HAR numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)   # (n, 6, 128) float32
        self.y = torch.from_numpy(y)   # (n,)        int64

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    normalize: bool = True,
    norm_params: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train/test DataLoaders for UCI HAR.

    Returns
    -------
    train_loader, test_loader, norm_params
    """
    X_train, y_train, X_test, y_test, norm_params = load_uci_har(
        data_dir, normalize=normalize, norm_params=norm_params
    )

    train_ds = HARDataset(X_train, y_train)
    test_ds  = HARDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, norm_params


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/UCI HAR Dataset"
    tr, te, nparams = get_dataloaders(data_dir, batch_size=32, num_workers=0)
    X_b, y_b = next(iter(tr))
    print(f"Batch X: {X_b.shape} {X_b.dtype}")  # (32, 6, 128) float32
    print(f"Batch y: {y_b.shape} {y_b.dtype}")   # (32,)        int64
    print(f"Train batches: {len(tr)}, Test batches: {len(te)}")
    print(f"Norm params mean[:3]: {nparams['mean'][:3]}")
