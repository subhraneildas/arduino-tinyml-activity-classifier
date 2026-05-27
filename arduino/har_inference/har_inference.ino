/**
 * har_inference.ino — Human Activity Recognition on nRF52840
 *
 * Arduino Nano 33 BLE Sense — TFLite Micro inference sketch.
 *
 * ─── What it does ────────────────────────────────────────────────────────────
 *  1. Calibrates IMU bias at startup (hold still for ~2 s).
 *  2. Collects IMU samples in a sliding window (128 samples @ ~50 Hz).
 *  3. Fills the TFLite input tensor (int8, shape 1×128×6, channels-last).
 *  4. Runs inference on the 1D CNN model (loaded from Flash C array).
 *  5. Prints the predicted activity + confidence to Serial @ 115200 baud.
 *  6. Optionally logs magnetometer data (not used in model, useful for
 *     collecting a 9-axis dataset for future fine-tuning).
 *
 * ─── Memory usage (nRF52840) ─────────────────────────────────────────────────
 *  Flash : ~95 KB model + ~50 KB sketch code  < 1 MB available
 *  RAM   : TENSOR_ARENA_SIZE (48 KB) + globals < 256 KB available
 *
 * ─── Before flashing ─────────────────────────────────────────────────────────
 *  1. Run the full training + export pipeline (see README.md steps 1-6).
 *  2. Paste normalization values into imu_handler.h (NORM_MEAN / NORM_STD).
 *  3. Update INPUT_SCALE and INPUT_ZERO_POINT below from
 *     models/quantization_params.json.
 *  4. Flash model_data.h/.cpp from step 6 of the README.
 *
 * ─── Libraries required (Arduino Library Manager) ───────────────────────────
 *  • Arduino_TensorFlowLite  (>= 2.4.0-ALPHA)  by TensorFlow
 *  • Arduino_LSM9DS1         (Rev1)             by Arduino
 *    — or —
 *  • Arduino_LSM6DS3 + Arduino_LIS2MDL          (Rev2)
 *
 * ─── Serial output format ────────────────────────────────────────────────────
 *  [INF] Predicted: WALKING (62.4%)  | window 5 | latency 14 ms
 *  [MAG] mx=12.34 my=-4.56 mz=23.10 uT  (optional, if LOG_MAGNETOMETER=1)
 */

// ─── TFLite Micro includes ────────────────────────────────────────────────────
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ─── Project includes ─────────────────────────────────────────────────────────
#include "model_data.h"    // generated C array: g_har_model_data, g_har_model_len
#include "imu_handler.h"   // IMU read + bias + gravity removal + normalization

// ─── Configuration ────────────────────────────────────────────────────────────

// *** UPDATE THESE FROM models/quantization_params.json ***
// input_scale and input_zero_point from the TFLite int8 model.
// Formula: q = round(f_normalized / INPUT_SCALE) + INPUT_ZERO_POINT
static const float INPUT_SCALE      =  0.0784f;  // placeholder — update after export
static const int   INPUT_ZERO_POINT = -1;        // placeholder — update after export

// Output quantization (for dequantizing logits if needed)
static const float OUTPUT_SCALE      = 0.00390625f;  // placeholder
static const int   OUTPUT_ZERO_POINT = -128;          // placeholder

// Window / stride parameters (must match training)
static const int WINDOW_SIZE   = 128;   // timesteps per inference window
static const int WINDOW_STRIDE = 64;    // new samples between windows (50% overlap)
static const int N_CHANNELS    = 6;     // body_acc XYZ + body_gyro XYZ

// Target sample interval in microseconds (50 Hz = 20000 us)
static const unsigned long SAMPLE_INTERVAL_US = 20000UL;

// Tensor arena size — increase if TFLite reports "AllocateTensors() failed"
static const int TENSOR_ARENA_SIZE = 49152;   // 48 KB

// Log magnetometer to Serial (not used in model — useful for data collection)
static const bool LOG_MAGNETOMETER = true;

// Minimum confidence to report (below this prints "UNCERTAIN")
static const float MIN_CONFIDENCE = 0.50f;

// ─── Activity labels ──────────────────────────────────────────────────────────
static const char* ACTIVITY_LABELS[] = {
  "WALKING",
  "WALKING_UPSTAIRS",
  "WALKING_DOWNSTAIRS",
  "SITTING",
  "STANDING",
  "LAYING",
};
static const int N_CLASSES = 6;

// ─── TFLite Micro globals ─────────────────────────────────────────────────────
namespace {
  tflite::AllOpsResolver resolver;
  const tflite::Model*   model      = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor*          input_tensor  = nullptr;
  TfLiteTensor*          output_tensor = nullptr;

  // Tensor arena lives in RAM — Cortex-M4F needs 8-byte alignment
  alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

// ─── Sliding window buffer ────────────────────────────────────────────────────
// Shape: [WINDOW_SIZE][N_CHANNELS] — matches TFLite input (1, 128, 6) channels-last
static float window_buf[WINDOW_SIZE][N_CHANNELS];
static int   sample_count  = 0;   // total samples collected
static int   window_count  = 0;   // total inferences run

// ─── Forward declarations ─────────────────────────────────────────────────────
static void run_inference();
static void fill_input_tensor();
static void print_prediction(int pred_class, float confidence, unsigned long latency_ms);

// ─────────────────────────────────────────────────────────────────────────────
// setup()
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000);   // wait up to 5 s for Serial monitor

  Serial.println("=========================================");
  Serial.println(" nRF52840 HAR Inference — TFLite Micro  ");
  Serial.println("=========================================");

  // ── IMU init
  if (!imu_begin()) {
    Serial.println("FATAL: IMU initialization failed. Halting.");
    while (true) { digitalWrite(LED_BUILTIN, HIGH); delay(200);
                   digitalWrite(LED_BUILTIN, LOW);  delay(200); }
  }

  // ── TFLite Micro model init
  model = tflite::GetModel(g_har_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("FATAL: TFLite schema mismatch (model=");
    Serial.print(model->version());
    Serial.print(", runtime=");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.println("). Rebuild with matching TFLite version.");
    while (true);
  }

  interpreter = new tflite::MicroInterpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE
  );

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println("FATAL: AllocateTensors() failed.");
    Serial.print("  Increase TENSOR_ARENA_SIZE (currently ");
    Serial.print(TENSOR_ARENA_SIZE / 1024);
    Serial.println(" KB).");
    while (true);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Print tensor shapes for debugging
  Serial.print("[TF] Input  tensor: (");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.print(input_tensor->dims->data[i]);
    if (i < input_tensor->dims->size - 1) Serial.print(", ");
  }
  Serial.print(")  dtype=");
  Serial.println(input_tensor->type == kTfLiteInt8 ? "int8" : "other");

  Serial.print("[TF] Output tensor: (");
  for (int i = 0; i < output_tensor->dims->size; i++) {
    Serial.print(output_tensor->dims->data[i]);
    if (i < output_tensor->dims->size - 1) Serial.print(", ");
  }
  Serial.println(")");

  size_t used = interpreter->arena_used_bytes();
  Serial.print("[TF] Tensor arena used: ");
  Serial.print(used / 1024);
  Serial.print(" KB  (of ");
  Serial.print(TENSOR_ARENA_SIZE / 1024);
  Serial.println(" KB allocated)");

  // ── IMU bias calibration
  imu_calibrate_bias();

  Serial.println("\n[HAR] Starting inference. Open Serial Monitor @ 115200 baud.");
  Serial.println("[HAR] Window: 128 samples @ 50 Hz = 2.56 s per inference.");
  Serial.println("---");
}

// ─────────────────────────────────────────────────────────────────────────────
// loop()
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  static unsigned long last_sample_us = 0;
  unsigned long now_us = micros();

  // Pace sampling to ~50 Hz
  if (now_us - last_sample_us < SAMPLE_INTERVAL_US) return;
  last_sample_us = now_us;

  // ── Read IMU sample
  float body_acc[3], body_gyro[3];
  if (!imu_read_sample(body_acc, body_gyro)) return;

  // ── Fill sliding window  (ring buffer via modulo)
  int slot = sample_count % WINDOW_SIZE;
  window_buf[slot][0] = body_acc[0];
  window_buf[slot][1] = body_acc[1];
  window_buf[slot][2] = body_acc[2];
  window_buf[slot][3] = body_gyro[0];
  window_buf[slot][4] = body_gyro[1];
  window_buf[slot][5] = body_gyro[2];

  sample_count++;

  // ── Log magnetometer (every window, not every sample, to reduce output)
  if (LOG_MAGNETOMETER && (sample_count % WINDOW_SIZE == 0)) {
    float mx, my, mz;
    if (imu_read_magnetometer(mx, my, mz)) {
      Serial.print("[MAG] mx="); Serial.print(mx, 2);
      Serial.print(" my=");       Serial.print(my, 2);
      Serial.print(" mz=");       Serial.print(mz, 2);
      Serial.println(" uT");
    }
  }

  // ── Trigger inference: first window after WINDOW_SIZE samples,
  //    then every WINDOW_STRIDE samples (50% overlap)
  bool first_window  = (sample_count == WINDOW_SIZE);
  bool stride_window = (sample_count > WINDOW_SIZE) &&
                       ((sample_count - WINDOW_SIZE) % WINDOW_STRIDE == 0);

  if (first_window || stride_window) {
    run_inference();
    window_count++;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// run_inference()  — fill tensor, invoke, decode output
// ─────────────────────────────────────────────────────────────────────────────
static void run_inference() {
  fill_input_tensor();

  unsigned long t0 = millis();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long latency = millis() - t0;

  if (status != kTfLiteOk) {
    Serial.println("[ERR] interpreter->Invoke() failed!");
    return;
  }

  // ── Decode output (int8 -> float probabilities)
  const int8_t* out_data = output_tensor->data.int8;
  float probs[N_CLASSES];
  float sum_exp = 0.0f;

  // Dequantize and compute softmax
  for (int c = 0; c < N_CLASSES; c++) {
    float logit = ((float)out_data[c] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    probs[c] = logit;   // raw dequantized logit (softmax applied below)
  }

  // Numerically stable softmax
  float max_logit = probs[0];
  for (int c = 1; c < N_CLASSES; c++) {
    if (probs[c] > max_logit) max_logit = probs[c];
  }
  for (int c = 0; c < N_CLASSES; c++) {
    probs[c] = expf(probs[c] - max_logit);
    sum_exp += probs[c];
  }
  for (int c = 0; c < N_CLASSES; c++) {
    probs[c] /= sum_exp;
  }

  // Find argmax
  int   pred_class  = 0;
  float pred_conf   = probs[0];
  for (int c = 1; c < N_CLASSES; c++) {
    if (probs[c] > pred_conf) {
      pred_conf  = probs[c];
      pred_class = c;
    }
  }

  print_prediction(pred_class, pred_conf, latency);
}

// ─────────────────────────────────────────────────────────────────────────────
// fill_input_tensor() — copy window buffer into int8 input tensor
//
// The TFLite model (post onnx2tf) expects shape (1, 128, 6) — channels-last.
// window_buf is stored as window_buf[timestep][channel], matching this layout.
//
// Quantization: q = clip(round(f / INPUT_SCALE) + INPUT_ZERO_POINT, -128, 127)
// ─────────────────────────────────────────────────────────────────────────────
static void fill_input_tensor() {
  int8_t* inp = input_tensor->data.int8;

  // The ring buffer's oldest sample starts at (sample_count % WINDOW_SIZE)
  int oldest = sample_count % WINDOW_SIZE;

  for (int t = 0; t < WINDOW_SIZE; t++) {
    int row = (oldest + t) % WINDOW_SIZE;   // chronological order
    for (int ch = 0; ch < N_CHANNELS; ch++) {
      float f = window_buf[row][ch];

      // Quantize float -> int8
      float q_f = roundf(f / INPUT_SCALE) + (float)INPUT_ZERO_POINT;
      int q_i   = (int)q_f;
      if (q_i >  127) q_i =  127;
      if (q_i < -128) q_i = -128;

      // Tensor index: [batch=0, t, ch] -> flat index = t*N_CHANNELS + ch
      inp[t * N_CHANNELS + ch] = (int8_t)q_i;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// print_prediction() — formatted Serial output
// ─────────────────────────────────────────────────────────────────────────────
static void print_prediction(int pred_class, float confidence, unsigned long latency_ms) {
  Serial.print("[INF] Predicted: ");

  if (confidence >= MIN_CONFIDENCE) {
    Serial.print(ACTIVITY_LABELS[pred_class]);
  } else {
    Serial.print("UNCERTAIN (best: ");
    Serial.print(ACTIVITY_LABELS[pred_class]);
    Serial.print(")");
  }

  Serial.print(" (");
  Serial.print((int)(confidence * 100.0f));
  Serial.print("%)");

  Serial.print("  | window ");
  Serial.print(window_count + 1);

  Serial.print("  | latency ");
  Serial.print(latency_ms);
  Serial.println(" ms");
}
