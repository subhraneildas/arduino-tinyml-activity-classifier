# nRF52840 Human Activity Recognition — TinyML Pipeline

End-to-end TinyML pipeline: train a 1D CNN on the **UCI HAR** dataset, quantize to **int8**, and deploy live inference on the **Arduino Nano 33 BLE Sense** (nRF52840) reading accelerometer + gyroscope from the onboard LSM9DS1.

```
┌─────────────────────────────────────────────────────────────────┐
│  UCI HAR Dataset (6-axis: body_acc XYZ + body_gyro XYZ)        │
│         ↓  training/train.py                                    │
│  PyTorch 1D-CNN  (~95K params, ~93% accuracy)                   │
│         ↓  training/export.py  (torch.onnx.export)              │
│  ONNX model  (.onnx)                                            │
│         ↓  training/export.py  (onnx2tf + TFLiteConverter)      │
│  TFLite int8  (~95KB, calibrated quantization)                  │
│         ↓  scripts/tflite_to_c_array.py                         │
│  C array header  (model_data.h / model_data.cpp)                │
│         ↓  Arduino IDE 2.x                                      │
│  nRF52840 live inference @ 50Hz, 128-sample sliding window      │
└─────────────────────────────────────────────────────────────────┘
```

## Activity Classes (UCI HAR)

| ID | Label | Description |
|----|-------|-------------|
| 0 | WALKING | Level walking |
| 1 | WALKING_UPSTAIRS | Climbing stairs up |
| 2 | WALKING_DOWNSTAIRS | Descending stairs |
| 3 | SITTING | Seated static |
| 4 | STANDING | Upright static |
| 5 | LAYING | Lying down |

## Hardware

- **Board**: Arduino Nano 33 BLE Sense (Rev1 — LSM9DS1)
  *(Rev2 with LSM6DS3 also supported — see `imu_handler.h`)*
- **MCU**: Nordic nRF52840 — 1 MB Flash, 256 KB RAM, Cortex-M4F @ 64 MHz
- **IMU**: LSM9DS1 — accelerometer + gyroscope + magnetometer
- **Inference**: 6-axis (acc+gyro), magnetometer logged for future datasets

---

## Quick Start

### 1. Python Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download UCI HAR Dataset

```bash
python data/download_uci_har.py
# Downloads and extracts to: data/UCI HAR Dataset/
```

### 3. Train the Model

```bash
python training/train.py \
    --data-dir "data/UCI HAR Dataset" \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --output-dir models/
```

Training takes ~5 min on CPU, <1 min on GPU. Targets **>93% test accuracy**.

### 4. Evaluate + Confusion Matrix

```bash
python training/evaluate.py \
    --data-dir "data/UCI HAR Dataset" \
    --checkpoint models/har_cnn_best.pt \
    --output-dir models/
```

Saves `models/confusion_matrix.png` and prints per-class metrics.

### 5. Export: PyTorch → ONNX → TFLite int8

```bash
python training/export.py \
    --data-dir "data/UCI HAR Dataset" \
    --checkpoint models/har_cnn_best.pt \
    --output-dir models/
```

Produces:
- `models/har_cnn.onnx` — ONNX model (FP32 checkpoint)
- `models/har_cnn.tflite` — TFLite FP32
- `models/har_cnn_int8.tflite` — TFLite int8 (deploy this)
- `models/quantization_params.json` — input/output scale+zero_point

### 6. Generate C Array for Arduino

```bash
python scripts/tflite_to_c_array.py \
    --tflite models/har_cnn_int8.tflite \
    --output-dir arduino/har_inference/ \
    --var-name g_har_model
```

Overwrites `arduino/har_inference/model_data.h` and `model_data.cpp`.

### 7. Update Arduino Sketch Normalization Params

Open `models/normalization_params.json` and copy the values into
`arduino/har_inference/imu_handler.h` → `NORM_MEAN[]` and `NORM_STD[]`.

Also copy `input_scale`, `input_zero_point` from `models/quantization_params.json`
into `arduino/har_inference/har_inference.ino`.

### 8. Flash to Arduino Nano 33 BLE Sense

**Arduino IDE 2.x Libraries required** (install via Library Manager):
- `Arduino_TensorFlowLite` by TensorFlow (>= 2.4.0-ALPHA)
- `Arduino_LSM9DS1` by Arduino (Rev1) **or** `Arduino_LSM6DS3` + `Arduino_LIS2MDL` (Rev2)

Open `arduino/har_inference/har_inference.ino`, select:
- **Board**: Arduino Nano 33 BLE
- **Port**: your COM/tty port

Upload -> open Serial Monitor @ **115200 baud**.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── CLAUDE.md                          # Codebase documentation
├── data/
│   ├── download_uci_har.py            # Dataset downloader
│   └── UCI HAR Dataset/               # (gitignored, created by download script)
├── training/
│   ├── dataset.py                     # UCI HAR data loader + normalization
│   ├── model.py                       # 1D CNN architecture (PyTorch)
│   ├── train.py                       # Training loop with LR scheduling
│   ├── evaluate.py                    # Accuracy, confusion matrix, per-class metrics
│   └── export.py                      # PyTorch -> ONNX -> TFLite int8
├── models/                            # (gitignored) generated artifacts
│   ├── har_cnn_best.pt
│   ├── har_cnn.onnx
│   ├── har_cnn_int8.tflite
│   ├── normalization_params.json
│   └── quantization_params.json
├── scripts/
│   └── tflite_to_c_array.py          # TFLite binary -> C array header
└── arduino/
    └── har_inference/
        ├── har_inference.ino          # Main Arduino sketch
        ├── imu_handler.h              # IMU reading + bias correction + normalization
        ├── model_data.h               # Generated C array header (UPDATE after step 6)
        └── model_data.cpp             # Generated C array implementation
```

## Model Architecture

```
Input  (1, 6, 128)  <- batch=1, channels=6, timesteps=128
   |
Conv1D(32, k=5) + BN + ReLU          -> (1, 32, 128)
Conv1D(64, k=5) + BN + ReLU + Pool/2 -> (1, 64,  64)
Conv1D(128,k=3) + BN + ReLU + Pool/2 -> (1,128,  32)
Conv1D(128,k=3) + BN + ReLU + Pool/2 -> (1,128,  16)
GlobalAveragePool                     -> (1, 128)
Dense(64) + ReLU + Dropout(0.5)       -> (1,  64)
Dense(6)                              -> (1,   6)  <- logits
```

| Metric | Value |
|--------|-------|
| Parameters | ~95K |
| FP32 size | ~380 KB |
| **int8 size** | **~95 KB** (fits in 1 MB Flash) |
| Tensor arena | ~48 KB (fits in 256 KB RAM) |
| UCI HAR accuracy | >93% |
| Window size | 128 samples @ 50Hz = 2.56 s |
| Inference latency | ~15 ms on Cortex-M4F @ 64 MHz |

## Sensor Notes

The Arduino Nano 33 BLE Sense IMU returns:
- Accelerometer: **g** units (same as UCI HAR body_acc)
- Gyroscope: **deg/s** -> converted to **rad/s** in sketch
- Magnetometer: **uT** (logged to Serial, not used in model)

Gravity removal uses a first-order IIR low-pass filter (alpha=0.9, equivalent to
~0.8 Hz cutoff @ 50Hz) subtracted from total acceleration, approximating the
Butterworth filter used in UCI HAR preprocessing.
