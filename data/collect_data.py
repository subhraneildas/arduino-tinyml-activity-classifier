#!/usr/bin/env python3
"""
collect_data.py — Interactive PC-side HAR data collector.

Communicates with har_data_collector.ino over USB-Serial to capture
9-axis IMU data (acc + gyro + mag) for building a custom HAR dataset.

Usage
-----
python data/collect_data.py --port /dev/ttyACM0
python data/collect_data.py --port COM3              # Windows
python data/collect_data.py --port auto              # auto-detect Nano 33 BLE

Output
------
data/collected/session_YYYYMMDD_HHMMSS/
    metadata.json                 — session info + calibration bias values
    WALKING_trial_001.csv         — raw 9-axis data, one row per sample
    WALKING_trial_002.csv
    WALKING_UPSTAIRS_trial_001.csv
    ...

CSV column order (matches Arduino har_data_collector.ino output):
    timestamp_ms, total_ax_g, total_ay_g, total_az_g,
    gyro_x_degs, gyro_y_degs, gyro_z_degs,
    mag_x_uT, mag_y_uT, mag_z_uT

Requirements
------------
    pip install pyserial
"""

import argparse
import csv
import json
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

# ─── Activity definitions (matching UCI HAR) ──────────────────────────────────
ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# Default collection parameters
DEFAULT_BAUD         = 115200
DEFAULT_TRIAL_SECS   = 30     # seconds per trial
DEFAULT_N_TRIALS     = 3      # trials per activity (for dataset variety)
DEFAULT_CALIB_N      = 100    # calibration samples (~2 s at 50 Hz)
DEFAULT_COUNTDOWN    = 3      # countdown seconds before recording
DEFAULT_PAUSE_SECS   = 3      # rest between trials

# CSV columns emitted by the Arduino sketch
CSV_COLUMNS = [
    "timestamp_ms",
    "total_ax_g", "total_ay_g", "total_az_g",
    "gyro_x_degs", "gyro_y_degs", "gyro_z_degs",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
]

SEPARATOR = "─" * 60


# ─── Serial reader thread ─────────────────────────────────────────────────────

class SerialReader(threading.Thread):
    """
    Background thread that reads lines from Serial and puts them into a queue.
    Separates data lines ("D,...") from control/status lines.
    """

    def __init__(self, ser: serial.Serial) -> None:
        super().__init__(daemon=True)
        self.ser      = ser
        self.data_q   : queue.Queue = queue.Queue()   # raw data line strings
        self.ctrl_q   : queue.Queue = queue.Queue()   # control/status strings
        self._stop    = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("D,"):
                    self.data_q.put(line)
                else:
                    self.ctrl_q.put(line)
            except (serial.SerialException, OSError):
                break

    def stop(self) -> None:
        self._stop.set()

    def wait_for(self, prefix: str, timeout: float = 10.0) -> Optional[str]:
        """Block until a control line starting with `prefix` arrives."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.ctrl_q.get(timeout=0.1)
                if line.startswith(prefix):
                    return line
                # put non-matching lines back
                self.ctrl_q.put(line)
            except queue.Empty:
                pass
        return None

    def flush_data(self) -> None:
        """Discard any buffered data lines."""
        while not self.data_q.empty():
            try:
                self.data_q.get_nowait()
            except queue.Empty:
                break


# ─── Port auto-detection ──────────────────────────────────────────────────────

def auto_detect_port() -> str:
    """Find the first USB-serial port that looks like a Nano 33 BLE."""
    candidates = []
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        vid  = port.vid
        pid  = port.pid
        # Arduino Nano 33 BLE uses VID=0x2341 PID=0x805a (bootloader) or 0x8057
        if vid == 0x2341 or "arduino" in desc or "usbmodem" in port.device.lower():
            candidates.append(port.device)
    if candidates:
        return candidates[0]
    # Fallback: first available port
    ports = [p.device for p in serial.tools.list_ports.comports()]
    if ports:
        return ports[0]
    return ""


def list_available_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  (no serial ports found)")
        return
    for p in ports:
        print(f"  {p.device:<20} {p.description}")


# ─── Connection helpers ───────────────────────────────────────────────────────

def connect(port: str, baud: int) -> Tuple[serial.Serial, SerialReader]:
    """Open serial port and wait for Arduino 'READY' message."""
    print(f"\nConnecting to {port} @ {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=2.0)
    except serial.SerialException as e:
        print(f"ERROR: Could not open port: {e}")
        print("Available ports:")
        list_available_ports()
        sys.exit(1)

    reader = SerialReader(ser)
    reader.start()

    # Arduino resets on serial open — wait for READY
    print("Waiting for Arduino to boot (up to 10 s) ...")
    ready = reader.wait_for("READY", timeout=10.0)
    if ready is None:
        print("ERROR: Timed out waiting for READY. Check the board and port.")
        reader.stop()
        ser.close()
        sys.exit(1)

    print("Connected and ready.")
    return ser, reader


def send_cmd(ser: serial.Serial, cmd: str) -> None:
    """Send a command line to the Arduino."""
    ser.write((cmd + "\n").encode("ascii"))


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate(ser: serial.Serial, reader: SerialReader, n: int = DEFAULT_CALIB_N) -> Dict:
    """
    Run bias calibration on the device.
    Returns dict of bias values: {ax, ay, az, gx, gy, gz}.
    """
    print(f"\n{SEPARATOR}")
    print("BIAS CALIBRATION")
    print(SEPARATOR)
    print(f"Place the device flat and hold STILL for ~{n / 50:.0f} seconds.")
    input("Press ENTER when ready ...")

    send_cmd(ser, f"CALIB {n}")

    # Wait for CALIB_START
    reader.wait_for("CALIB_START", timeout=5.0)
    print(f"Calibrating ({n} samples @ 50 Hz) ", end="", flush=True)

    # Animate progress dots while waiting
    done_line = None
    deadline = time.time() + (n / 50.0) + 5.0
    while time.time() < deadline:
        try:
            line = reader.ctrl_q.get(timeout=0.3)
            if line.startswith("CALIB_DONE"):
                done_line = line
                break
            reader.ctrl_q.put(line)
        except queue.Empty:
            print(".", end="", flush=True)
    print()

    if done_line is None:
        print("ERROR: Calibration timed out.")
        sys.exit(1)

    # Parse: CALIB_DONE ax ay az gx gy gz
    parts = done_line.split()
    if len(parts) < 7:
        print(f"ERROR: Unexpected CALIB_DONE format: {done_line}")
        sys.exit(1)

    bias = {
        "bias_ax_g":    float(parts[1]),
        "bias_ay_g":    float(parts[2]),
        "bias_az_g":    float(parts[3]),
        "bias_gx_degs": float(parts[4]),
        "bias_gy_degs": float(parts[5]),
        "bias_gz_degs": float(parts[6]),
    }

    print("Bias calibration complete:")
    print(f"  acc  [g]     : ax={bias['bias_ax_g']:+.5f}  "
          f"ay={bias['bias_ay_g']:+.5f}  az={bias['bias_az_g']:+.5f}")
    print(f"  gyro [deg/s] : gx={bias['bias_gx_degs']:+.5f}  "
          f"gy={bias['bias_gy_degs']:+.5f}  gz={bias['bias_gz_degs']:+.5f}")
    return bias


# ─── Single trial recording ───────────────────────────────────────────────────

def record_trial(
    ser: serial.Serial,
    reader: SerialReader,
    activity: str,
    trial_num: int,
    duration_s: float,
    out_dir: Path,
) -> int:
    """
    Record one trial of an activity. Returns number of samples collected.
    Saves data to <out_dir>/<activity>_trial_<NNN>.csv.
    """
    print(f"\n  {SEPARATOR}")
    print(f"  {activity}  —  Trial {trial_num}")
    print(f"  {SEPARATOR}")
    print(f"  Prepare to perform: {activity.replace('_', ' ')}")

    # Countdown
    for i in range(DEFAULT_COUNTDOWN, 0, -1):
        print(f"  Starting in {i}...", end="\r", flush=True)
        time.sleep(1.0)
    print(f"  GO!                    ")

    # Flush stale data and start streaming
    reader.flush_data()
    send_cmd(ser, "START")
    reader.wait_for("STREAMING", timeout=5.0)

    t_start = time.time()
    rows: List[List[str]] = []

    # Progress bar characters
    bar_width = 30

    while True:
        elapsed = time.time() - t_start
        if elapsed >= duration_s:
            break

        # Drain all available data samples
        while not reader.data_q.empty():
            try:
                line = reader.data_q.get_nowait()
                # Strip "D," prefix and split
                cols = line[2:].split(",")
                if len(cols) == len(CSV_COLUMNS):
                    rows.append(cols)
            except queue.Empty:
                break

        # Progress bar
        frac = min(elapsed / duration_s, 1.0)
        filled = int(frac * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        n_samples = len(rows)
        print(f"  [{bar}] {elapsed:5.1f}/{duration_s:.0f} s  "
              f"{n_samples:5d} samples", end="\r", flush=True)
        time.sleep(0.05)

    # Stop streaming
    send_cmd(ser, "STOP")
    resp = reader.wait_for("STOPPED", timeout=5.0)
    arduino_count = int(resp.split()[1]) if resp else -1

    # Drain any remaining samples
    time.sleep(0.2)
    while not reader.data_q.empty():
        try:
            line = reader.data_q.get_nowait()
            cols = line[2:].split(",")
            if len(cols) == len(CSV_COLUMNS):
                rows.append(cols)
        except queue.Empty:
            break

    print()  # newline after progress bar

    # Save CSV
    fname = f"{activity}_trial_{trial_num:03d}.csv"
    fpath = out_dir / fname
    with open(fpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)

    n_saved = len(rows)
    actual_hz = n_saved / duration_s if duration_s > 0 else 0
    print(f"  Saved: {fname}  ({n_saved:,} samples, {actual_hz:.1f} Hz effective, "
          f"Arduino reported {arduino_count})")
    return n_saved


# ─── Session orchestration ────────────────────────────────────────────────────

def run_collection_session(
    ser: serial.Serial,
    reader: SerialReader,
    out_dir: Path,
    bias: Dict,
    args: argparse.Namespace,
) -> None:
    """Interactive loop: choose activities and record trials."""
    session_summary: List[Dict] = []

    while True:
        print(f"\n{SEPARATOR}")
        print("ACTIVITY SELECTION")
        print(SEPARATOR)
        print("Available activities:")
        for i, lbl in enumerate(ACTIVITY_LABELS):
            print(f"  {i}: {lbl}")
        print(f"  {len(ACTIVITY_LABELS)}: (done — end session)")

        # Activity selection
        while True:
            try:
                choice_str = input("\nEnter activity number: ").strip()
                choice = int(choice_str)
                if 0 <= choice <= len(ACTIVITY_LABELS):
                    break
                print("  Invalid choice.")
            except ValueError:
                print("  Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended by user.")
                return

        if choice == len(ACTIVITY_LABELS):
            print("Ending session.")
            break

        activity = ACTIVITY_LABELS[choice]

        # Number of trials
        try:
            n_trials_str = input(
                f"How many trials? [{args.n_trials}]: "
            ).strip()
            n_trials = int(n_trials_str) if n_trials_str else args.n_trials
        except (ValueError, EOFError, KeyboardInterrupt):
            n_trials = args.n_trials

        # Duration
        try:
            dur_str = input(
                f"Duration per trial (seconds)? [{args.duration}]: "
            ).strip()
            duration = float(dur_str) if dur_str else args.duration
        except (ValueError, EOFError, KeyboardInterrupt):
            duration = args.duration

        # Find next trial number for this activity (so we can resume sessions)
        existing = list(out_dir.glob(f"{activity}_trial_*.csv"))
        start_trial = len(existing) + 1

        total_samples = 0
        for t in range(start_trial, start_trial + n_trials):
            n = record_trial(ser, reader, activity, t, duration, out_dir)
            total_samples += n
            session_summary.append({
                "activity": activity,
                "trial": t,
                "samples": n,
                "duration_s": duration,
            })
            if t < start_trial + n_trials - 1:
                print(f"  Resting {DEFAULT_PAUSE_SECS} s before next trial ...")
                time.sleep(DEFAULT_PAUSE_SECS)

        print(f"\n  {activity}: {n_trials} trial(s), "
              f"{total_samples:,} total samples collected.")

    # Print session summary
    print(f"\n{SEPARATOR}")
    print("SESSION SUMMARY")
    print(SEPARATOR)
    activity_counts: Dict[str, int] = {}
    for entry in session_summary:
        act = entry["activity"]
        activity_counts[act] = activity_counts.get(act, 0) + entry["samples"]

    for act, cnt in activity_counts.items():
        windows = max(0, (cnt - 128) // 64 + 1) if cnt >= 128 else 0
        print(f"  {act:<25} {cnt:6,} samples  ~{windows:4d} windows (128/64)")

    # Update metadata with trial summary
    meta_path = out_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["trial_summary"] = session_summary
    meta["activity_sample_counts"] = activity_counts
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata updated: {meta_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive IMU data collector for HAR dataset building",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port",      default="auto",
                   help="Serial port (e.g. /dev/ttyACM0, COM3, or 'auto')")
    p.add_argument("--baud",      type=int, default=DEFAULT_BAUD)
    p.add_argument("--output-dir",default="data/collected",
                   help="Root directory for session output")
    p.add_argument("--calib-n",  type=int, default=DEFAULT_CALIB_N,
                   help="Number of calibration samples")
    p.add_argument("--duration",  type=float, default=DEFAULT_TRIAL_SECS,
                   help="Default recording duration per trial (s)")
    p.add_argument("--n-trials",  type=int, default=DEFAULT_N_TRIALS,
                   help="Default number of trials per activity")
    p.add_argument("--subject",   default="",
                   help="Optional subject identifier (for multi-subject datasets)")
    p.add_argument("--list-ports", action="store_true",
                   help="List available serial ports and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_ports:
        print("Available serial ports:")
        list_available_ports()
        return

    print("=" * 60)
    print("  nRF52840 HAR Data Collector")
    print("=" * 60)

    # ── Resolve port
    port = args.port
    if port == "auto":
        port = auto_detect_port()
        if not port:
            print("ERROR: No serial port found. Available ports:")
            list_available_ports()
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    # ── Connect
    ser, reader = connect(port, args.baud)

    # ── Create session directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"session_{ts}"
    out_dir = Path(args.output_dir) / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session directory: {out_dir}")

    # ── Calibrate
    bias = calibrate(ser, reader, n=args.calib_n)

    # ── Save session metadata
    meta = {
        "session_id":    session_name,
        "timestamp":     ts,
        "subject":       args.subject,
        "port":          port,
        "baud":          args.baud,
        "calib_n":       args.calib_n,
        "sample_rate_hz": 50,
        "bias":          bias,
        "csv_columns":   CSV_COLUMNS,
        "units": {
            "total_ax_g":    "g (raw total accel including gravity)",
            "total_ay_g":    "g",
            "total_az_g":    "g",
            "gyro_x_degs":   "deg/s (raw gyroscope)",
            "gyro_y_degs":   "deg/s",
            "gyro_z_degs":   "deg/s",
            "mag_x_uT":      "microtesla",
            "mag_y_uT":      "microtesla",
            "mag_z_uT":      "microtesla",
        },
        "preprocessing_note": (
            "Apply bias subtraction using bias.bias_a*_g / bias.bias_g*_degs. "
            "Then apply 3rd-order Butterworth LP at 0.3 Hz to get gravity, "
            "subtract to get body_acc. Convert gyro deg/s -> rad/s."
        ),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    # ── Interactive collection loop
    try:
        run_collection_session(ser, reader, out_dir, bias, args)
    except KeyboardInterrupt:
        print("\n\nInterrupted — sending STOP to Arduino ...")
        send_cmd(ser, "STOP")
        time.sleep(0.5)
    finally:
        reader.stop()
        ser.close()
        print(f"\nSession data saved to: {out_dir}")
        print("Next: python data/preprocess_collected.py "
              f"--session-dir {out_dir}")


if __name__ == "__main__":
    main()
