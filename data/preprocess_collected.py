#!/usr/bin/env python3
"""
preprocess_collected.py — Convert raw collected IMU CSV data into
windowed numpy arrays ready for training/fine-tuning the HARCNN.

Pipeline (per trial CSV file)
------------------------------
1. Load raw 9-axis CSV (from har_data_collector.ino / collect_data.py).
2. Subtract static bias  (from session metadata.json calibration values).
3. Gravity removal:
     gravity = Butterworth LP (fc=0.3 Hz, order=3)  applied to total_acc
     body_acc = bias_corrected_total_acc - gravity
   Matches the UCI HAR preprocessing exactly.
4. Convert gyroscope deg/s -> rad/s.
5. Segment into non-overlapping / 50%-overlapping windows of 128 samples.
6. Z-score normalization:
     - If --norm-params given: use UCI HAR training stats (for fine-tuning).
     - Otherwise: compute from this dataset's training split.
7. Optional: merge windows with UCI HAR dataset (training/dataset.py).
8. Save as numpy arrays + normalization_params.json.

Output channels (must match training/dataset.py):
  [0] body_acc_x   [1] body_acc_y   [2] body_acc_z   (g, gravity removed)
  [3] body_gyro_x  [4] body_gyro_y  [5] body_gyro_z   (rad/s)

Note: magnetometer data is preserved in a separate *_9axis.npy file for
      future 9-axis model research, but not used in the current 6-axis CNN.

Usage
-----
# Preprocess one session, standalone normalization:
python data/preprocess_collected.py \\
    --session-dir data/collected/session_20240127_143022/ \\
    --output-dir  data/collected_processed/

# Preprocess + merge with UCI HAR + use UCI HAR normalization stats:
python data/preprocess_collected.py \\
    --session-dir data/collected/session_20240127_143022/ \\
    --merge-uci   "data/UCI HAR Dataset" \\
    --norm-params models/normalization_params.json \\
    --output-dir  data/collected_processed/

# Multiple sessions at once:
python data/preprocess_collected.py \\
    --session-dir data/collected/session_*/  \\
    --output-dir  data/collected_processed/

Outputs
-------
data/collected_processed/
    X_train.npy           (n_train, 6, 128)  float32, channels-first
    y_train.npy           (n_train,)          int64,   0..5
    X_test.npy            (n_test,  6, 128)
    y_test.npy            (n_test,)
    X_train_9axis.npy     (n_train, 9, 128)  float32  (acc+gyro+mag, norm)
    X_test_9axis.npy      (n_test,  9, 128)
    normalization_params.json
    preprocessing_report.json
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

# Local import for UCI HAR merge
sys.path.insert(0, str(Path(__file__).parent.parent))

ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
LABEL_TO_ID = {lbl: i for i, lbl in enumerate(ACTIVITY_LABELS)}

WINDOW_SIZE   = 128   # samples per window
WINDOW_STRIDE = 64    # 50% overlap
N_CHANNELS    = 6     # body_acc (3) + body_gyro (3)
SAMPLE_RATE   = 50.0  # Hz

# Butterworth filter parameters (match UCI HAR gravity removal)
BUTTER_ORDER  = 3
GRAVITY_FC_HZ = 0.3   # cutoff frequency for gravity component


# ─── Signal processing ────────────────────────────────────────────────────────

def butter_lowpass(data: np.ndarray, fc: float = GRAVITY_FC_HZ,
                   fs: float = SAMPLE_RATE, order: int = BUTTER_ORDER) -> np.ndarray:
    """
    Zero-phase 3rd-order Butterworth low-pass filter.
    Matches the gravity-removal filter used in UCI HAR data collection.

    data: (n_samples, n_channels) or (n_samples,)
    Returns same shape.
    """
    nyq = 0.5 * fs
    Wn  = fc / nyq
    b, a = butter(order, Wn, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def remove_gravity_butterworth(total_acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate gravity from body acceleration using Butterworth LP filter.

    Parameters
    ----------
    total_acc : (n, 3)  total acceleration [g], bias already subtracted

    Returns
    -------
    body_acc : (n, 3)  linear body acceleration [g], gravity removed
    gravity  : (n, 3)  estimated gravity component [g]
    """
    gravity  = butter_lowpass(total_acc)          # low-freq = gravity
    body_acc = total_acc - gravity                  # high-freq = body motion
    return body_acc, gravity


# ─── CSV loading and preprocessing ───────────────────────────────────────────

def load_trial_csv(csv_path: Path, bias: Dict) -> Optional[pd.DataFrame]:
    """
    Load one trial CSV file and apply bias subtraction.

    Returns a DataFrame with columns:
      body_acc_x/y/z [g], body_gyro_x/y/z [rad/s], mag_x/y/z [uT]
    Returns None if file has fewer than WINDOW_SIZE samples.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  WARN: Could not read {csv_path.name}: {e}")
        return None

    if len(df) < WINDOW_SIZE:
        print(f"  WARN: {csv_path.name} has only {len(df)} samples "
              f"(need >= {WINDOW_SIZE}) — skipping.")
        return None

    # Extract raw columns
    total_acc  = df[["total_ax_g", "total_ay_g", "total_az_g"]].values.copy()
    gyro_degs  = df[["gyro_x_degs", "gyro_y_degs", "gyro_z_degs"]].values.copy()
    mag        = df[["mag_x_uT", "mag_y_uT", "mag_z_uT"]].values.copy()

    # 1. Subtract static bias
    total_acc[:, 0] -= bias.get("bias_ax_g",    0.0)
    total_acc[:, 1] -= bias.get("bias_ay_g",    0.0)
    total_acc[:, 2] -= bias.get("bias_az_g",    0.0)
    gyro_degs[:, 0] -= bias.get("bias_gx_degs", 0.0)
    gyro_degs[:, 1] -= bias.get("bias_gy_degs", 0.0)
    gyro_degs[:, 2] -= bias.get("bias_gz_degs", 0.0)

    # 2. Gravity removal via Butterworth LP
    body_acc, _ = remove_gravity_butterworth(total_acc)

    # 3. Gyroscope deg/s -> rad/s
    body_gyro = gyro_degs * (np.pi / 180.0)

    # Assemble 9-axis frame
    result = pd.DataFrame({
        "body_acc_x":   body_acc[:, 0],
        "body_acc_y":   body_acc[:, 1],
        "body_acc_z":   body_acc[:, 2],
        "body_gyro_x":  body_gyro[:, 0],
        "body_gyro_y":  body_gyro[:, 1],
        "body_gyro_z":  body_gyro[:, 2],
        "mag_x_uT":     mag[:, 0],
        "mag_y_uT":     mag[:, 1],
        "mag_z_uT":     mag[:, 2],
    })
    return result


# ─── Windowing ────────────────────────────────────────────────────────────────

def segment_windows(
    data_6ch: np.ndarray,
    label_id: int,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window across the signal and extract fixed-size segments.

    Parameters
    ----------
    data_6ch : (n_samples, 6)  body_acc XYZ + body_gyro XYZ
    label_id : int              activity class 0..5

    Returns
    -------
    X : (n_windows, 6, window_size)  channels-first for PyTorch Conv1d
    y : (n_windows,)                 int64
    """
    n = len(data_6ch)
    starts = range(0, n - window_size + 1, stride)
    windows = []
    for s in starts:
        w = data_6ch[s : s + window_size]   # (window_size, 6)
        windows.append(w.T)                  # (6, window_size)

    if not windows:
        return np.empty((0, 6, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(windows, axis=0).astype(np.float32)   # (n_windows, 6, window_size)
    y = np.full(len(windows), label_id, dtype=np.int64)
    return X, y


def segment_windows_9axis(
    data_9ch: np.ndarray,
    label_id: int,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as segment_windows but for all 9 channels (acc+gyro+mag)."""
    n = len(data_9ch)
    starts = range(0, n - window_size + 1, stride)
    windows = [data_9ch[s : s + window_size].T for s in starts]
    if not windows:
        return np.empty((0, 9, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.full(len(windows), label_id, dtype=np.int64)
    return X, y


# ─── Session loading ──────────────────────────────────────────────────────────

def load_session(session_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all trial CSV files in a session directory.

    Returns
    -------
    X6    : (n_windows, 6, 128)  6-axis windows (channels-first)
    y     : (n_windows,)          labels 0..5
    X9    : (n_windows, 9, 128)  9-axis windows
    y9    : (n_windows,)
    """
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {session_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    bias = meta.get("bias", {})

    all_X6, all_y6 = [], []
    all_X9, all_y9 = [], []
    trial_counts: Dict[str, int] = {}

    for lbl in ACTIVITY_LABELS:
        csv_files = sorted(session_dir.glob(f"{lbl}_trial_*.csv"))
        if not csv_files:
            continue

        trial_counts[lbl] = 0
        for csv_path in csv_files:
            df = load_trial_csv(csv_path, bias)
            if df is None:
                continue

            lbl_id = LABEL_TO_ID[lbl]
            data_6ch = df[[
                "body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z",
            ]].values   # (n, 6)

            data_9ch = df[[
                "body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "mag_x_uT", "mag_y_uT", "mag_z_uT",
            ]].values   # (n, 9)

            X6, y6 = segment_windows(data_6ch, lbl_id)
            X9, y9 = segment_windows_9axis(data_9ch, lbl_id)

            if len(X6) == 0:
                print(f"  WARN: No windows from {csv_path.name}")
                continue

            all_X6.append(X6); all_y6.append(y6)
            all_X9.append(X9); all_y9.append(y9)
            trial_counts[lbl] = trial_counts.get(lbl, 0) + len(X6)
            print(f"  {csv_path.name}: {len(df):,} samples -> {len(X6)} windows [{lbl}]")

    if not all_X6:
        raise ValueError(f"No valid data found in session {session_dir}")

    X6 = np.concatenate(all_X6, axis=0)
    y6 = np.concatenate(all_y6, axis=0)
    X9 = np.concatenate(all_X9, axis=0)
    y9 = np.concatenate(all_y9, axis=0)

    print(f"\n  Session total: {len(X6)} windows across "
          f"{len(trial_counts)} activities")
    for lbl, cnt in trial_counts.items():
        print(f"    {lbl:<25} {cnt:4d} windows")

    return X6, y6, X9, y9


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_6axis(
    X: np.ndarray,
    norm_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Z-score normalize (n, 6, 128) array.

    If norm_params given (from UCI HAR training): apply those stats.
    Otherwise: compute from X (per-channel, across all samples & timesteps).
    """
    if norm_params is not None:
        mean = np.array(norm_params["mean"], dtype=np.float32)   # (6,)
        std  = np.array(norm_params["std"],  dtype=np.float32)   # (6,)
    else:
        # X: (n, 6, 128) -> reshape to (n*128, 6) for stats
        flat = X.transpose(0, 2, 1).reshape(-1, 6)
        mean = flat.mean(axis=0).astype(np.float32)
        std  = flat.std(axis=0).astype(np.float32)
        std  = np.where(std < 1e-8, 1.0, std)
        norm_params = {
            "mean": mean.tolist(),
            "std":  std.tolist(),
            "channels": [
                "body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z",
            ],
            "units": ["g", "g", "g", "rad/s", "rad/s", "rad/s"],
            "source": "custom_dataset",
        }

    # Broadcast: (n, 6, 128) - (6,1) / (6,1)
    X_norm = (X - mean[:, None]) / std[:, None]
    return X_norm.astype(np.float32), norm_params


def normalize_9axis(X9: np.ndarray) -> np.ndarray:
    """Per-channel z-score for 9-axis data (acc+gyro from UCI params, mag standalone)."""
    flat = X9.transpose(0, 2, 1).reshape(-1, 9)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    return ((X9 - mean[:, None]) / std[:, None]).astype(np.float32)


# ─── UCI HAR merge ────────────────────────────────────────────────────────────

def merge_with_uci_har(
    X_custom: np.ndarray, y_custom: np.ndarray,
    uci_har_dir: str,
    norm_params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge custom dataset windows with UCI HAR dataset.
    Uses the same normalization params for both.
    """
    from training.dataset import load_uci_har

    print("\nMerging with UCI HAR dataset ...")
    X_tr, y_tr, X_te, y_te, _ = load_uci_har(
        uci_har_dir, normalize=True, norm_params=norm_params
    )

    X_all = np.concatenate([X_tr, X_te, X_custom], axis=0)
    y_all = np.concatenate([y_tr, y_te, y_custom], axis=0)

    print(f"  UCI HAR:  {len(X_tr) + len(X_te):,} windows")
    print(f"  Custom:   {len(X_custom):,} windows")
    print(f"  Combined: {len(X_all):,} windows")
    return X_all, y_all


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess collected IMU data into training-ready numpy arrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--session-dir", nargs="+", required=True,
        help="One or more session directories (supports glob: data/collected/session_*/)",
    )
    p.add_argument("--output-dir",  default="data/collected_processed/")
    p.add_argument(
        "--merge-uci",  default="",
        help="Path to 'UCI HAR Dataset' to merge with collected data",
    )
    p.add_argument(
        "--norm-params", default="",
        help="Path to normalization_params.json (use UCI HAR stats for fine-tuning). "
             "If omitted, stats are computed from this dataset.",
    )
    p.add_argument(
        "--test-split", type=float, default=0.2,
        help="Fraction of custom windows to reserve for test set",
    )
    p.add_argument(
        "--stride", type=int, default=WINDOW_STRIDE,
        help=f"Window stride in samples (default {WINDOW_STRIDE} = 50%% overlap)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Expand glob patterns in session-dir
    session_dirs: List[Path] = []
    for pattern in args.session_dir:
        matches = sorted(glob.glob(pattern))
        if matches:
            session_dirs.extend(Path(m) for m in matches if Path(m).is_dir())
        elif Path(pattern).is_dir():
            session_dirs.append(Path(pattern))

    if not session_dirs:
        print("ERROR: No session directories found.")
        sys.exit(1)

    print(f"Processing {len(session_dirs)} session(s):")
    for d in session_dirs:
        print(f"  {d}")

    # ── Load and preprocess all sessions
    all_X6, all_y6 = [], []
    all_X9, all_y9 = [], []

    for sd in session_dirs:
        print(f"\n{'─'*60}")
        print(f"Session: {sd.name}")
        print('─'*60)
        X6, y6, X9, y9 = load_session(sd)
        all_X6.append(X6); all_y6.append(y6)
        all_X9.append(X9); all_y9.append(y9)

    X6 = np.concatenate(all_X6, axis=0)
    y6 = np.concatenate(all_y6, axis=0)
    X9 = np.concatenate(all_X9, axis=0)

    print(f"\n{'─'*60}")
    print(f"Total windows before normalization: {len(X6):,}")
    print(f"Label distribution:")
    for lbl_id, lbl in enumerate(ACTIVITY_LABELS):
        cnt = (y6 == lbl_id).sum()
        if cnt > 0:
            print(f"  {lbl:<25} {cnt:4d} windows")

    # ── Load pre-computed norm params (optional)
    norm_params = None
    if args.norm_params and Path(args.norm_params).exists():
        with open(args.norm_params) as f:
            norm_params = json.load(f)
        print(f"\nUsing normalization params from: {args.norm_params}")
        print(f"  mean={[f'{v:.4f}' for v in norm_params['mean']]}")
        print(f"  std ={[f'{v:.4f}' for v in norm_params['std']]}")

    # ── Train/test split (stratified)
    X6_tr, X6_te, y6_tr, y6_te = train_test_split(
        X6, y6, test_size=args.test_split,
        random_state=args.seed, stratify=y6,
    )
    X9_tr, X9_te, _, _ = train_test_split(
        X9, y6, test_size=args.test_split,
        random_state=args.seed, stratify=y6,
    )

    # ── Normalize 6-axis
    X6_tr_norm, norm_params = normalize_6axis(X6_tr, norm_params)
    X6_te_norm, _           = normalize_6axis(X6_te, norm_params)

    # ── Normalize 9-axis (standalone stats, always)
    X9_tr_norm = normalize_9axis(X9_tr)
    X9_te_norm = normalize_9axis(X9_te)

    # ── Merge with UCI HAR (optional)
    if args.merge_uci and Path(args.merge_uci).exists():
        X6_tr_norm, y6_tr = merge_with_uci_har(
            X6_tr_norm, y6_tr, args.merge_uci, norm_params
        )
        # Shuffle merged set
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(X6_tr_norm))
        X6_tr_norm, y6_tr = X6_tr_norm[idx], y6_tr[idx]

    # ── Save 6-axis arrays
    np.save(out_dir / "X_train.npy", X6_tr_norm)
    np.save(out_dir / "y_train.npy", y6_tr)
    np.save(out_dir / "X_test.npy",  X6_te_norm)
    np.save(out_dir / "y_test.npy",  y6_te)

    # ── Save 9-axis arrays
    np.save(out_dir / "X_train_9axis.npy", X9_tr_norm)
    np.save(out_dir / "X_test_9axis.npy",  X9_te_norm)

    # ── Save normalization params
    with open(out_dir / "normalization_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)

    # ── Report
    report = {
        "timestamp":       datetime.now().isoformat(),
        "n_sessions":      len(session_dirs),
        "session_dirs":    [str(d) for d in session_dirs],
        "window_size":     WINDOW_SIZE,
        "window_stride":   args.stride,
        "sample_rate_hz":  SAMPLE_RATE,
        "n_channels_6ax":  N_CHANNELS,
        "n_channels_9ax":  9,
        "train_windows":   int(len(X6_tr_norm)),
        "test_windows":    int(len(X6_te_norm)),
        "merged_uci_har":  bool(args.merge_uci),
        "norm_source":     norm_params.get("source", "custom_dataset"),
        "label_distribution_train": {
            ACTIVITY_LABELS[i]: int((y6_tr == i).sum())
            for i in range(6)
            if (y6_tr == i).sum() > 0
        },
        "label_distribution_test": {
            ACTIVITY_LABELS[i]: int((y6_te == i).sum())
            for i in range(6)
            if (y6_te == i).sum() > 0
        },
        "output_files": {
            "X_train": "X_train.npy  (n_train, 6, 128) float32",
            "y_train": "y_train.npy  (n_train,)         int64",
            "X_test":  "X_test.npy   (n_test,  6, 128) float32",
            "y_test":  "y_test.npy   (n_test,)          int64",
            "X_train_9axis": "X_train_9axis.npy  (n_train, 9, 128)",
            "X_test_9axis":  "X_test_9axis.npy   (n_test,  9, 128)",
        },
    }
    with open(out_dir / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print('='*60)
    print(f"  Train windows : {len(X6_tr_norm):,}  shape {X6_tr_norm.shape}")
    print(f"  Test  windows : {len(X6_te_norm):,}  shape {X6_te_norm.shape}")
    print(f"  9-axis train  : {X9_tr_norm.shape}")
    print(f"\n  Saved to: {out_dir}/")
    for fname in ["X_train.npy","y_train.npy","X_test.npy","y_test.npy",
                  "X_train_9axis.npy","X_test_9axis.npy",
                  "normalization_params.json","preprocessing_report.json"]:
        p = out_dir / fname
        kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"    {fname:<30} {kb:7.1f} KB")

    print(f"\nTo fine-tune on this data:")
    print(f"  python training/train.py \\")
    print(f"      --data-dir {out_dir} \\")
    print(f"      --data-format numpy \\")
    print(f"      --checkpoint models/har_cnn_best.pt \\")
    print(f"      --epochs 20 --lr 1e-4")
    print(f"\nTo inspect the data:")
    print(f"  python data/inspect_data.py --session-dir {session_dirs[0]}")


if __name__ == "__main__":
    main()
