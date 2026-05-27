# CLAUDE.md — nRF52840 HAR TinyML Project

## Project Purpose
End-to-end TinyML pipeline for Human Activity Recognition (HAR) on the Arduino
Nano 33 BLE Sense (nRF52840). Trains a 1D CNN on UCI HAR dataset, quantizes to
int8 via the ONNX -> TFLite workflow, and deploys live inference on the MCU.

## Directory Layout
- `data/` — Dataset download script; actual data is gitignored
- `training/` — Python: data loading, model, training, evaluation, export
- `scripts/` — Utility: TFLite binary to C array header
- `arduino/har_inference/` — Arduino IDE sketch + IMU handler + model C array
- `models/` — Generated artifacts (gitignored)

## Key Files
- `training/model.py` — HARCNN class: 1D CNN with 4 conv blocks + GAP + FC
- `training/dataset.py` — UCI HAR loader: body_acc+body_gyro (6ch, 128 steps)
- `training/train.py` — CLI training loop; saves best checkpoint by val accuracy
- `training/export.py` — PyTorch -> ONNX -> TFLite FP32 -> TFLite int8
- `arduino/har_inference/har_inference.ino` — TFLite Micro inference sketch
- `arduino/har_inference/imu_handler.h` — LSM9DS1 read + bias correction + z-score
- `arduino/har_inference/model_data.h/.cpp` — Generated; do not hand-edit

## Python Stack
- PyTorch 2.x for training
- torch.onnx.export for ONNX checkpoint
- onnx2tf for ONNX -> TF SavedModel
- tensorflow.lite.TFLiteConverter for TFLite + int8 PTQ
- onnxruntime for ONNX validation

## Arduino Stack
- TensorFlowLite library (>=2.4.0-ALPHA) via Arduino Library Manager
- Arduino_LSM9DS1 library for IMU access
- Target: Arduino Nano 33 BLE (nRF52840, Cortex-M4F @ 64 MHz)

## Input/Output Conventions
- Model input shape: (1, 6, 128) in PyTorch (channels-first)
- TFLite model input shape: (1, 128, 6) (channels-last, after onnx2tf)
- UCI HAR labels: 0=WALKING, 1=WALKING_UPSTAIRS, 2=WALKING_DOWNSTAIRS,
                  3=SITTING, 4=STANDING, 5=LAYING
- Normalization: z-score, computed from train split, saved to
  models/normalization_params.json, hardcoded in imu_handler.h

## Quantization
- Type: Post-Training Integer Quantization (PTQ) with representative dataset
- Input/output dtype: int8
- Quantization params (scale, zero_point) saved to models/quantization_params.json
- Arduino sketch applies: q = round(float_val / input_scale) + input_zero_point

## Memory Budget (nRF52840)
- Flash: model ~95KB + sketch ~100KB < 1MB available
- RAM: tensor arena 48KB + stack/heap < 256KB available

## Common Commands
```bash
# Full pipeline
python data/download_uci_har.py
python training/train.py --data-dir "data/UCI HAR Dataset" --epochs 50
python training/evaluate.py --data-dir "data/UCI HAR Dataset" --checkpoint models/har_cnn_best.pt
python training/export.py --data-dir "data/UCI HAR Dataset" --checkpoint models/har_cnn_best.pt
python scripts/tflite_to_c_array.py --tflite models/har_cnn_int8.tflite --output-dir arduino/har_inference/
```
