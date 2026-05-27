#!/usr/bin/env python3
"""
Download and extract the UCI Human Activity Recognition dataset.

Usage:
    python data/download_uci_har.py [--output-dir data/]

Dataset URL:
    https://archive.ics.uci.edu/ml/machine-learning-databases/00240/
    UCI HAR Dataset.zip  (62 MB)

After extraction you will have:
    data/UCI HAR Dataset/
        train/
            Inertial Signals/
                body_acc_x_train.txt   shape (7352, 128)
                body_acc_y_train.txt   shape (7352, 128)
                body_acc_z_train.txt   shape (7352, 128)
                body_gyro_x_train.txt  shape (7352, 128)
                body_gyro_y_train.txt  shape (7352, 128)
                body_gyro_z_train.txt  shape (7352, 128)
                total_acc_x_train.txt  ...
                ...
            y_train.txt                shape (7352,) labels 1-6
        test/
            ...same structure...
            y_test.txt                 shape (2947,) labels 1-6
        activity_labels.txt
        features.txt
"""

import argparse
import hashlib
import io
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00240/UCI%20HAR%20Dataset.zip"
)
EXPECTED_MD5 = "53e099237392e0b9602f592c672fd1f2"  # for verification


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream-download url → dest with a progress bar."""
    print(f"Downloading {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI HAR dataset")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to download and extract into (default: data/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    zip_path = out_dir / "UCI_HAR_Dataset.zip"
    dataset_dir = out_dir / "UCI HAR Dataset"

    if dataset_dir.exists():
        print(f"Dataset already present at: {dataset_dir}")
        return

    # Download
    if not zip_path.exists():
        download(UCI_HAR_URL, zip_path)
    else:
        print(f"Using cached zip: {zip_path}")

    # Verify MD5
    print("Verifying checksum ...", end=" ", flush=True)
    actual_md5 = md5_file(zip_path)
    if actual_md5 != EXPECTED_MD5:
        print(f"MISMATCH — expected {EXPECTED_MD5}, got {actual_md5}")
        print("Deleting corrupt file. Re-run to download again.")
        zip_path.unlink()
        raise SystemExit(1)
    print("OK")

    # Extract
    print(f"Extracting to {out_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Done. Dataset is at: {dataset_dir}")

    # Quick sanity check
    required = [
        dataset_dir / "train" / "Inertial Signals" / "body_acc_x_train.txt",
        dataset_dir / "train" / "y_train.txt",
        dataset_dir / "test"  / "Inertial Signals" / "body_acc_x_test.txt",
        dataset_dir / "test"  / "y_test.txt",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Extraction seems incomplete. Missing:\n" + "\n".join(str(m) for m in missing)
        )
    print("Sanity check passed — all required files present.")


if __name__ == "__main__":
    main()
