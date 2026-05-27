#!/usr/bin/env python3
"""
inspect_data.py — Visualize raw collected IMU session data.

Generates diagnostic plots to verify data quality before preprocessing:
  1. Time-series plots for each activity trial (acc + gyro + mag)
  2. Spectrogram per channel (frequency content check)
  3. Label distribution bar chart across all sessions
  4. Gravity estimation overlay (Butterworth LP filter result)
  5. Per-channel statistics table (mean, std, min, max)

Usage
-----
# Inspect a single session
python data/inspect_data.py --session-dir data/collected/session_20240127_143022/

# Inspect a specific trial CSV
python data/inspect_data.py --csv data/collected/session_XXX/WALKING_trial_001.csv

# Inspect multiple sessions
python data/inspect_data.py --session-dir data/collected/session_*/

# Save all plots to a directory (no display)
python data/inspect_data.py --session-dir data/collected/session_XXX/ --output-dir plots/

Requirements
------------
pip install matplotlib scipy numpy pandas
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, spectrogram

ACTIVITY_LABELS = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]
SAMPLE_RATE = 50.0   # Hz

# Color palette for activities
ACT_COLORS = [
    "#2196F3", "#FF9800", "#4CAF50",
    "#9C27B0", "#F44336", "#795548",
]
CHANNEL_COLORS = {
    "acc_x": "#E53935", "acc_y": "#43A047", "acc_z": "#1E88E5",
    "gyro_x": "#FB8C00", "gyro_y": "#8E24AA", "gyro_z": "#00ACC1",
    "mag_x": "#6D4C41", "mag_y": "#546E7A", "mag_z": "#37474F",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def butter_lowpass_gravity(data: np.ndarray, fc: float = 0.3, fs: float = 50.0) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(3, fc / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)


def load_csv_with_meta(csv_path: Path, meta: Dict) -> Optional[pd.DataFrame]:
    """Load a trial CSV and attach bias-subtracted columns for visualization."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"WARN: {csv_path.name}: {e}")
        return None
    if len(df) < 10:
        return None

    # Time axis in seconds
    df["time_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0

    # Bias subtraction (for visualization of the corrected signal)
    bias = meta.get("bias", {})
    df["body_acc_x_raw"] = df["total_ax_g"]   - bias.get("bias_ax_g",    0)
    df["body_acc_y_raw"] = df["total_ay_g"]   - bias.get("bias_ay_g",    0)
    df["body_acc_z_raw"] = df["total_az_g"]   - bias.get("bias_az_g",    0)
    df["body_gyro_x"]    = (df["gyro_x_degs"] - bias.get("bias_gx_degs", 0)) * np.pi / 180
    df["body_gyro_y"]    = (df["gyro_y_degs"] - bias.get("bias_gy_degs", 0)) * np.pi / 180
    df["body_gyro_z"]    = (df["gyro_z_degs"] - bias.get("bias_gz_degs", 0)) * np.pi / 180

    # Gravity estimate + body acc
    total = df[["body_acc_x_raw", "body_acc_y_raw", "body_acc_z_raw"]].values
    gravity = butter_lowpass_gravity(total)
    body    = total - gravity
    df["body_acc_x"] = body[:, 0]
    df["body_acc_y"] = body[:, 1]
    df["body_acc_z"] = body[:, 2]
    df["gravity_x"]  = gravity[:, 0]
    df["gravity_y"]  = gravity[:, 1]
    df["gravity_z"]  = gravity[:, 2]

    return df


# ─── Plot 1: Time-series for one trial ───────────────────────────────────────

def plot_trial_timeseries(
    df: pd.DataFrame,
    title: str,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    4-row plot: total acc | body acc + gravity overlay | gyro | mag
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    t = df["time_s"].values

    # ── Row 1: Total acceleration (raw, bias subtracted)
    ax = axes[0]
    ax.plot(t, df["body_acc_x_raw"], color="#E53935", lw=0.8, label="total_ax")
    ax.plot(t, df["body_acc_y_raw"], color="#43A047", lw=0.8, label="total_ay")
    ax.plot(t, df["body_acc_z_raw"], color="#1E88E5", lw=0.8, label="total_az")
    ax.set_ylabel("Total Acc [g]\n(bias subtracted)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.grid(True, alpha=0.3)

    # ── Row 2: Body acceleration + gravity overlay
    ax = axes[1]
    ax.plot(t, df["body_acc_x"], color="#E53935", lw=0.8, label="body_ax")
    ax.plot(t, df["body_acc_y"], color="#43A047", lw=0.8, label="body_ay")
    ax.plot(t, df["body_acc_z"], color="#1E88E5", lw=0.8, label="body_az")
    ax.plot(t, df["gravity_z"],  color="#1E88E5", lw=1.2, ls="--", alpha=0.5, label="gravity_z")
    ax.set_ylabel("Body Acc [g]\n(gravity removed)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.grid(True, alpha=0.3)

    # ── Row 3: Gyroscope (rad/s)
    ax = axes[2]
    ax.plot(t, df["body_gyro_x"], color="#FB8C00", lw=0.8, label="gyro_x")
    ax.plot(t, df["body_gyro_y"], color="#8E24AA", lw=0.8, label="gyro_y")
    ax.plot(t, df["body_gyro_z"], color="#00ACC1", lw=0.8, label="gyro_z")
    ax.set_ylabel("Body Gyro [rad/s]", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.grid(True, alpha=0.3)

    # ── Row 4: Magnetometer
    ax = axes[3]
    ax.plot(t, df["mag_x_uT"], color="#6D4C41", lw=0.8, label="mag_x")
    ax.plot(t, df["mag_y_uT"], color="#546E7A", lw=0.8, label="mag_y")
    ax.plot(t, df["mag_z_uT"], color="#37474F", lw=0.8, label="mag_z")
    ax.set_ylabel("Magnetometer [µT]", fontsize=8)
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


# ─── Plot 2: Spectrograms ─────────────────────────────────────────────────────

def plot_spectrograms(
    df: pd.DataFrame,
    title: str,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Spectrogram for each of the 6 body-motion channels."""
    channels = [
        ("body_acc_x",  "Body Acc X [g]"),
        ("body_acc_y",  "Body Acc Y [g]"),
        ("body_acc_z",  "Body Acc Z [g]"),
        ("body_gyro_x", "Body Gyro X [rad/s]"),
        ("body_gyro_y", "Body Gyro Y [rad/s]"),
        ("body_gyro_z", "Body Gyro Z [rad/s]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(f"Spectrograms — {title}", fontsize=12, fontweight="bold")

    for ax, (col, lbl) in zip(axes.flat, channels):
        sig = df[col].values
        f, t_s, Sxx = spectrogram(sig, fs=SAMPLE_RATE, nperseg=64, noverlap=32)
        # Only show 0–10 Hz (activity frequencies)
        freq_mask = f <= 10
        im = ax.pcolormesh(
            t_s, f[freq_mask],
            10 * np.log10(Sxx[freq_mask] + 1e-12),
            shading="gouraud", cmap="viridis",
        )
        ax.set_ylabel("Freq [Hz]", fontsize=8)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_title(lbl, fontsize=9)
        plt.colorbar(im, ax=ax, label="dB", pad=0.02)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


# ─── Plot 3: Session label distribution ──────────────────────────────────────

def plot_label_distribution(
    session_dirs: List[Path],
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of window counts per activity across all sessions."""
    # Count trial files per activity per session
    counts: Dict[str, List[int]] = {lbl: [] for lbl in ACTIVITY_LABELS}

    for sd in session_dirs:
        for lbl in ACTIVITY_LABELS:
            csvs = list(sd.glob(f"{lbl}_trial_*.csv"))
            total_rows = 0
            for csv_path in csvs:
                try:
                    df = pd.read_csv(csv_path)
                    total_rows += len(df)
                except Exception:
                    pass
            windows = max(0, (total_rows - 128) // 64 + 1) if total_rows >= 128 else 0
            counts[lbl].append(windows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Overview", fontsize=13, fontweight="bold")

    # Left: stacked bar (per session breakdown)
    ax = axes[0]
    x = np.arange(len(ACTIVITY_LABELS))
    width = 0.6 / max(len(session_dirs), 1)
    session_names = [sd.name for sd in session_dirs]
    bar_colors = plt.cm.Set2(np.linspace(0, 1, len(session_dirs)))
    for i, (sname, color) in enumerate(zip(session_names, bar_colors)):
        vals = [counts[lbl][i] if i < len(counts[lbl]) else 0 for lbl in ACTIVITY_LABELS]
        offset = (i - len(session_dirs) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=sname, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in ACTIVITY_LABELS], fontsize=8)
    ax.set_ylabel("Estimated windows (128/64)")
    ax.set_title("Windows per Activity per Session")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Right: pie chart of total windows
    ax = axes[1]
    totals = [sum(counts[lbl]) for lbl in ACTIVITY_LABELS]
    non_zero = [(lbl, t) for lbl, t in zip(ACTIVITY_LABELS, totals) if t > 0]
    if non_zero:
        lbls, vals = zip(*non_zero)
        ax.pie(
            vals,
            labels=[l.replace("_", "\n") for l in lbls],
            colors=[ACT_COLORS[ACTIVITY_LABELS.index(l)] for l in lbls],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 8},
        )
        ax.set_title("Total Dataset Composition")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    return fig


# ─── Plot 4: Per-channel statistics ──────────────────────────────────────────

def print_channel_stats(df: pd.DataFrame, title: str) -> None:
    """Print a table of per-channel statistics."""
    cols = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "mag_x_uT", "mag_y_uT", "mag_z_uT",
    ]
    units = ["g"]*3 + ["rad/s"]*3 + ["µT"]*3

    print(f"\n{'─'*70}")
    print(f"Channel statistics — {title}")
    print(f"{'─'*70}")
    print(f"{'Channel':<18} {'Unit':<7} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"{'─'*70}")
    for col, unit in zip(cols, units):
        if col not in df.columns:
            continue
        v = df[col].values
        print(f"{col:<18} {unit:<7} "
              f"{v.mean():>9.4f} {v.std():>9.4f} "
              f"{v.min():>9.4f} {v.max():>9.4f}")
    print(f"{'─'*70}")
    print(f"Samples: {len(df):,}  |  Duration: {df['time_s'].iloc[-1]:.2f} s  "
          f"|  Effective rate: {len(df)/df['time_s'].iloc[-1]:.1f} Hz")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect collected IMU session data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--session-dir", nargs="+",
                     help="One or more session directories")
    grp.add_argument("--csv", help="Single trial CSV file to inspect")

    p.add_argument("--output-dir", default="",
                   help="Directory to save plots (empty = show interactively)")
    p.add_argument("--no-show", action="store_true",
                   help="Don't display plots (useful with --output-dir)")
    p.add_argument("--max-trials", type=int, default=3,
                   help="Max trials per activity to plot time-series for")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show or args.output_dir:
        matplotlib.use("Agg")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Single CSV mode
    if args.csv:
        csv_path = Path(args.csv)
        # Try to load metadata from parent session dir
        meta_path = csv_path.parent / "metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        df = load_csv_with_meta(csv_path, meta)
        if df is None:
            print("ERROR: Could not load CSV or too few samples.")
            sys.exit(1)

        title = csv_path.stem
        print_channel_stats(df, title)

        fig1 = plot_trial_timeseries(
            df, title,
            out_path=out_dir / f"{title}_timeseries.png" if out_dir else None,
        )
        fig2 = plot_spectrograms(
            df, title,
            out_path=out_dir / f"{title}_spectrogram.png" if out_dir else None,
        )
        if not args.no_show:
            plt.show()
        return

    # ─── Session directory mode
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

    print(f"Inspecting {len(session_dirs)} session(s):")
    for sd in session_dirs:
        print(f"  {sd}")

    # ── 1. Label distribution across all sessions
    print("\nGenerating label distribution plot ...")
    plot_label_distribution(
        session_dirs,
        out_path=out_dir / "label_distribution.png" if out_dir else None,
    )

    # ── 2. Time-series + spectrograms per trial (limited to max_trials)
    for sd in session_dirs:
        meta_path = sd / "metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        print(f"\nSession: {sd.name}")

        for lbl in ACTIVITY_LABELS:
            csv_files = sorted(sd.glob(f"{lbl}_trial_*.csv"))[: args.max_trials]
            for csv_path in csv_files:
                title = f"{sd.name} / {csv_path.stem}"
                print(f"  Plotting: {csv_path.name} ...")

                df = load_csv_with_meta(csv_path, meta)
                if df is None:
                    continue

                print_channel_stats(df, csv_path.stem)

                safe_name = csv_path.stem.replace(" ", "_")

                plot_trial_timeseries(
                    df, title,
                    out_path=out_dir / f"{safe_name}_timeseries.png" if out_dir else None,
                )
                plot_spectrograms(
                    df, title,
                    out_path=out_dir / f"{safe_name}_spectrogram.png" if out_dir else None,
                )

    if out_dir:
        print(f"\nAll plots saved to: {out_dir}/")
    elif not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
