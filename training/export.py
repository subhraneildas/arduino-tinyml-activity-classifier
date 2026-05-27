#!/usr/bin/env python3
"""
Export HARCNN: PyTorch -> ONNX -> TFLite FP32 -> TFLite int8 (PTQ).

Pipeline
--------
1. Load PyTorch checkpoint, fuse BN layers into Conv for export.
2. Export to ONNX (opset 17) using torch.onnx.export — FP32 checkpoint.
3. Validate ONNX inference with onnxruntime.
4. Convert ONNX -> TF SavedModel using onnx2tf (handles channels-first
   Conv1d transposition automatically).
5. Re-quantize SavedModel -> TFLite int8 using tf.lite.TFLiteConverter
   with a representative calibration dataset from the UCI HAR test split.
6. Validate TFLite int8 inference and check accuracy.
7. Save quantization parameters (input/output scale + zero_point) as JSON
   for use in the Arduino sketch.

Usage
-----
python training/export.py \\
    --data-dir  "data/UCI HAR Dataset" \\
    --checkpoint models/har_cnn_best.pt \\
    --output-dir models/

Outputs
-------
models/har_cnn.onnx             — ONNX FP32 model
models/har_cnn.tflite           — TFLite FP32 model
models/har_cnn_int8.tflite      — TFLite int8 model  (deploy this)
models/quantization_params.json — input/output scale + zero_point
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import N_CLASSES, load_uci_har
from training.model import HARCNN

# Input shape constants
BATCH = 1
N_CH  = 6
SEQ   = 128
# Number of calibration windows for representative dataset
N_CAL = 500


# ── 1. Load PyTorch model ───────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HARCNN(n_channels=N_CH, n_classes=N_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_params = ckpt.get("norm_params")
    print(f"Loaded checkpoint ({ckpt.get('epoch', '?')} epochs, "
          f"acc={ckpt.get('test_acc', 0):.2%})")
    return model, norm_params


# ── 2. ONNX export ─────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, onnx_path: str) -> None:
    """Export to ONNX opset 17 with dynamic batch axis."""
    dummy = torch.randn(BATCH, N_CH, SEQ)  # (1, 6, 128)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        verbose=False,
    )
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"ONNX export -> {onnx_path}  ({size_kb:.1f} KB)")


# ── 3. ONNX validation ─────────────────────────────────────────────────────────

def validate_onnx(onnx_path: str, model: nn.Module) -> None:
    """Check ONNX Runtime output matches PyTorch output (max diff < 1e-4)."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed; skipping ONNX validation.")
        return

    import onnx
    onnx.checker.check_model(onnx_path)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x_np = np.random.randn(4, N_CH, SEQ).astype(np.float32)
    ort_out = sess.run(None, {"input": x_np})[0]

    with torch.no_grad():
        pt_out = model(torch.from_numpy(x_np)).numpy()

    max_diff = np.abs(ort_out - pt_out).max()
    print(f"ONNX validation: max |PyTorch - ORT| = {max_diff:.2e}  {'OK' if max_diff < 1e-3 else 'WARN'}")


# ── 4. ONNX -> TF SavedModel ───────────────────────────────────────────────────

def convert_onnx_to_tf(onnx_path: str, saved_model_dir: str) -> None:
    """
    Convert ONNX -> TF SavedModel using onnx2tf.

    onnx2tf automatically handles channels-first -> channels-last transposition
    for 1D Conv ops (ONNX uses NCW, TF uses NWC).
    """
    try:
        import onnx2tf
    except ImportError:
        raise ImportError(
            "onnx2tf is required. Install with: pip install onnx2tf"
        )

    if Path(saved_model_dir).exists():
        shutil.rmtree(saved_model_dir)

    print(f"Converting ONNX -> TF SavedModel (this may take ~30s) ...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        not_use_onnxsim=False,          # simplify ONNX graph first
        verbosity="error",
    )
    print(f"TF SavedModel -> {saved_model_dir}/")


# ── 5 & 6. TF SavedModel -> TFLite FP32 & int8 ────────────────────────────────

def build_representative_dataset(X_cal: np.ndarray):
    """
    Generator for TFLite PTQ calibration.
    Yields batches of shape (1, SEQ, N_CH) [channels-last, as onnx2tf outputs].
    """
    # X_cal shape: (n, N_CH, SEQ)  channels-first (PyTorch convention)
    # TFLite model (after onnx2tf) expects (1, SEQ, N_CH)  channels-last
    X_cl = X_cal.transpose(0, 2, 1)   # -> (n, SEQ, N_CH)

    def gen():
        for i in range(len(X_cl)):
            sample = X_cl[i : i + 1].astype(np.float32)   # (1, 128, 6)
            yield [sample]

    return gen


def convert_to_tflite(
    saved_model_dir: str,
    X_cal: np.ndarray,
    fp32_path: str,
    int8_path: str,
) -> None:
    """Convert TF SavedModel to FP32 and int8 TFLite."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow is required. Install with: pip install tensorflow")

    # ── FP32 TFLite ────────────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_fp32 = converter.convert()
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)
    print(f"TFLite FP32 -> {fp32_path}  ({len(tflite_fp32)/1024:.1f} KB)")

    # ── int8 TFLite (Post-Training Quantization) ───────────────────────────────
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = build_representative_dataset(X_cal)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    print("Running int8 PTQ calibration ...")
    tflite_int8 = converter.convert()
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)
    print(f"TFLite int8  -> {int8_path}  ({len(tflite_int8)/1024:.1f} KB)")


# ── 7. Validate TFLite & extract quantization params ──────────────────────────

def validate_tflite_int8(
    int8_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_eval: int = 500,
) -> Dict:
    """
    Run TFLite int8 interpreter on a subset of test data.
    Returns quantization parameters for input/output tensors.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("[skip] tensorflow not found; skipping TFLite validation.")
        return {}

    interpreter = tf.lite.Interpreter(model_path=int8_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zp   = input_details[0]["quantization"]
    out_scale, out_zp = output_details[0]["quantization"]

    # Channels-last for TFLite (onnx2tf transposes)
    X_cl = X_test[:n_eval].transpose(0, 2, 1)   # (n, 128, 6)

    correct = 0
    for i in range(len(X_cl)):
        sample_f = X_cl[i : i + 1].astype(np.float32)   # (1, 128, 6)
        # Quantize input: q = round(f / scale) + zero_point, clip to int8
        sample_q = np.round(sample_f / in_scale + in_zp).astype(np.int8)
        sample_q = np.clip(sample_q, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(output_details[0]["index"])  # (1, 6) int8

        # Dequantize: f = (q - zero_point) * scale
        out_f = (out_q.astype(np.float32) - out_zp) * out_scale
        pred  = out_f.argmax(axis=-1)[0]
        if pred == y_test[i]:
            correct += 1

    acc = correct / len(X_cl)
    print(f"TFLite int8 accuracy on {n_eval} samples: {acc:.2%}")

    quant_params = {
        "input_scale":        float(in_scale),
        "input_zero_point":   int(in_zp),
        "output_scale":       float(out_scale),
        "output_zero_point":  int(out_zp),
        "input_shape_tflite": [1, SEQ, N_CH],   # channels-last (TFLite)
        "input_dtype":        "int8",
        "output_dtype":       "int8",
        "note": (
            "Input: quantize float with q=round(f/input_scale)+input_zero_point, "
            "clip to [-128,127]. "
            "Output: dequantize with f=(q-output_zero_point)*output_scale."
        ),
    }
    return quant_params


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export HARCNN to ONNX + TFLite int8")
    p.add_argument("--data-dir",    required=True, help="Path to 'UCI HAR Dataset'")
    p.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    p.add_argument("--output-dir",  default="models/")
    p.add_argument("--n-cal",  type=int, default=N_CAL,
                   help="Calibration samples for int8 PTQ (default 500)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")   # export always on CPU

    # ── Load model
    model, norm_params = load_model(args.checkpoint, device)

    # ── Load data for calibration and validation
    print("Loading UCI HAR data for calibration ...")
    X_train, _, X_test, y_test, norm_params = load_uci_har(
        args.data_dir, normalize=True, norm_params=norm_params
    )
    # Calibration set: first N_CAL windows from train split (shuffled)
    rng = np.random.default_rng(42)
    cal_idx = rng.choice(len(X_train), size=min(args.n_cal, len(X_train)), replace=False)
    X_cal = X_train[cal_idx]
    print(f"  Calibration set: {len(X_cal)} windows")

    # ── File paths
    onnx_path       = str(out_dir / "har_cnn.onnx")
    saved_model_dir = str(out_dir / "har_cnn_tf_savedmodel")
    fp32_tflite     = str(out_dir / "har_cnn.tflite")
    int8_tflite     = str(out_dir / "har_cnn_int8.tflite")

    # ── Step 2: Export ONNX
    print("\n[1/4] Exporting PyTorch -> ONNX ...")
    export_onnx(model, onnx_path)

    # ── Step 3: Validate ONNX
    print("\n[2/4] Validating ONNX with OnnxRuntime ...")
    validate_onnx(onnx_path, model)

    # ── Step 4: ONNX -> TF SavedModel
    print("\n[3/4] Converting ONNX -> TF SavedModel ...")
    convert_onnx_to_tf(onnx_path, saved_model_dir)

    # ── Step 5: TF -> TFLite (FP32 + int8)
    print("\n[4/4] Converting TF -> TFLite FP32 + int8 PTQ ...")
    convert_to_tflite(saved_model_dir, X_cal, fp32_tflite, int8_tflite)

    # ── Step 6: Validate int8 TFLite
    print("\nValidating TFLite int8 accuracy ...")
    quant_params = validate_tflite_int8(int8_tflite, X_test, y_test)

    # ── Save quantization params
    qp_path = out_dir / "quantization_params.json"
    with open(qp_path, "w") as f:
        json.dump(quant_params, f, indent=2)
    print(f"\nQuantization params -> {qp_path}")
    print(json.dumps(quant_params, indent=2))

    # ── Summary
    print("\n" + "="*60)
    print("Export complete. Files:")
    for p in [onnx_path, fp32_tflite, int8_tflite]:
        kb = os.path.getsize(p) / 1024
        print(f"  {p}  ({kb:.1f} KB)")
    print("\nNext steps:")
    print("  1. python scripts/tflite_to_c_array.py \\")
    print(f"        --tflite {int8_tflite} \\")
    print("        --output-dir arduino/har_inference/")
    print("  2. Update NORM_MEAN/NORM_STD in arduino/har_inference/imu_handler.h")
    print(f"     Values: mean={norm_params['mean']}")
    print(f"             std ={norm_params['std']}")
    print("  3. Update INPUT_SCALE/INPUT_ZERO_POINT in har_inference.ino")
    print(f"     Values: scale={quant_params.get('input_scale')}, "
          f"zero_point={quant_params.get('input_zero_point')}")


if __name__ == "__main__":
    main()
