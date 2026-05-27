/**
 * imu_handler.h — IMU reading, bias calibration, gravity removal,
 *                 and z-score normalization for HAR inference.
 *
 * Target: Arduino Nano 33 BLE Sense (LSM9DS1)
 *         Also supports Rev2 (LSM6DS3 + LIS2MDL) — see NANO33_REV2 flag.
 *
 * Coordinate system matches UCI HAR:
 *   Accelerometer: body_acc  = total_acc - gravity  [g units]
 *   Gyroscope:     body_gyro = raw gyro              [rad/s]
 *   (UCI HAR gyro is already in rad/s; LSM9DS1 returns deg/s -> converted here)
 *
 * *** IMPORTANT — Update after training ***
 * Copy the values from models/normalization_params.json into NORM_MEAN and
 * NORM_STD below before flashing.  The defaults are placeholders.
 *
 * Channel order (must match training/dataset.py):
 *   [0] body_acc_x   [1] body_acc_y   [2] body_acc_z
 *   [3] body_gyro_x  [4] body_gyro_y  [5] body_gyro_z
 */

#pragma once

// ─── Board selection ──────────────────────────────────────────────────────────
// Uncomment the line below for Arduino Nano 33 BLE Sense Rev2 (LSM6DS3+LIS2MDL)
// #define NANO33_REV2

#ifdef NANO33_REV2
  #include <Arduino_LSM6DS3.h>
  #include <Arduino_LIS2MDL.h>
  #define IMU_ACCEL_AVAIL()   IMU.accelerationAvailable()
  #define IMU_GYRO_AVAIL()    IMU.gyroscopeAvailable()
  #define IMU_READ_ACCEL(x,y,z) IMU.readAcceleration(x,y,z)
  #define IMU_READ_GYRO(x,y,z)  IMU.readGyroscope(x,y,z)
  #define MAG_AVAIL()         MAG.magneticFieldAvailable()
  #define MAG_READ(x,y,z)     MAG.readMagneticField(x,y,z)
  static bool imu_mag_ok = false;
#else
  #include <Arduino_LSM9DS1.h>
  #define IMU_ACCEL_AVAIL()   IMU.accelerationAvailable()
  #define IMU_GYRO_AVAIL()    IMU.gyroscopeAvailable()
  #define IMU_READ_ACCEL(x,y,z) IMU.readAcceleration(x,y,z)
  #define IMU_READ_GYRO(x,y,z)  IMU.readGyroscope(x,y,z)
  #define MAG_AVAIL()         IMU.magneticFieldAvailable()
  #define MAG_READ(x,y,z)     IMU.readMagneticField(x,y,z)
  static bool imu_mag_ok = true;   // LSM9DS1 has built-in mag
#endif

// ─── Normalization parameters ─────────────────────────────────────────────────
// *** REPLACE THESE WITH VALUES FROM models/normalization_params.json ***
//
// Format from training:
//   channels: [body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z]
//
static const float NORM_MEAN[6] = {
  0.0f,  // body_acc_x  [g]      <- replace with norm_params["mean"][0]
  0.0f,  // body_acc_y  [g]      <- replace with norm_params["mean"][1]
  0.0f,  // body_acc_z  [g]      <- replace with norm_params["mean"][2]
  0.0f,  // body_gyro_x [rad/s]  <- replace with norm_params["mean"][3]
  0.0f,  // body_gyro_y [rad/s]  <- replace with norm_params["mean"][4]
  0.0f,  // body_gyro_z [rad/s]  <- replace with norm_params["mean"][5]
};

static const float NORM_STD[6] = {
  1.0f,  // body_acc_x  <- replace with norm_params["std"][0]
  1.0f,  // body_acc_y  <- replace with norm_params["std"][1]
  1.0f,  // body_acc_z  <- replace with norm_params["std"][2]
  1.0f,  // body_gyro_x <- replace with norm_params["std"][3]
  1.0f,  // body_gyro_y <- replace with norm_params["std"][4]
  1.0f,  // body_gyro_z <- replace with norm_params["std"][5]
};

// ─── Constants ────────────────────────────────────────────────────────────────
static const float DEG_TO_RAD  = 0.017453292519943295f;  // PI/180
static const float GRAVITY_MS2 = 9.80665f;               // not used (acc in g)

// IIR low-pass filter coefficient for gravity estimation
// α = exp(-2π * Fc / Fs) with Fc=0.8Hz, Fs=50Hz ≈ 0.904
// Lower α → faster gravity tracking; higher α → smoother estimate
static const float GRAVITY_LP_ALPHA = 0.904f;

// Bias calibration
static const int   BIAS_N_SAMPLES   = 100;   // samples @ 50Hz = 2 s
static const float BIAS_WAIT_MS     = 3000;  // wait before calibration (ms)

// ─── State variables ─────────────────────────────────────────────────────────
static float bias_acc[3]     = {0, 0, 0};
static float bias_gyro[3]    = {0, 0, 0};
static float gravity_est[3]  = {0, 0, 0};   // IIR gravity estimate
static bool  bias_calibrated = false;

// ─── Public functions ─────────────────────────────────────────────────────────

/**
 * imu_begin() — Initialize the IMU hardware.
 * Call once in setup(). Returns true on success.
 */
inline bool imu_begin() {
  if (!IMU.begin()) {
    Serial.println("[IMU] ERROR: IMU.begin() failed!");
    return false;
  }
  Serial.print("[IMU] Accelerometer sample rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("[IMU] Gyroscope sample rate:     ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

#ifdef NANO33_REV2
  if (MAG.begin()) {
    imu_mag_ok = true;
    Serial.println("[IMU] Magnetometer: OK (LIS2MDL)");
  } else {
    imu_mag_ok = false;
    Serial.println("[IMU] Magnetometer: FAILED");
  }
#else
  Serial.println("[IMU] Magnetometer: OK (LSM9DS1 built-in)");
#endif

  return true;
}

/**
 * imu_calibrate_bias() — Measure static bias by averaging BIAS_N_SAMPLES
 * while the sensor is stationary.  Also seeds the gravity IIR filter.
 *
 * Hold the device still during calibration (~2 seconds).
 * Call once in setup(), after imu_begin().
 */
inline void imu_calibrate_bias() {
  Serial.print("[IMU] Bias calibration — hold device still for ");
  Serial.print(BIAS_N_SAMPLES / 50.0f, 1);
  Serial.println(" s ...");

  delay((int)BIAS_WAIT_MS);

  double sum_acc[3]  = {0, 0, 0};
  double sum_gyro[3] = {0, 0, 0};
  int    count       = 0;

  while (count < BIAS_N_SAMPLES) {
    if (!IMU_ACCEL_AVAIL() || !IMU_GYRO_AVAIL()) { delay(1); continue; }

    float ax, ay, az, gx, gy, gz;
    IMU_READ_ACCEL(ax, ay, az);
    IMU_READ_GYRO(gx, gy, gz);

    sum_acc[0]  += ax;  sum_acc[1]  += ay;  sum_acc[2]  += az;
    sum_gyro[0] += gx;  sum_gyro[1] += gy;  sum_gyro[2] += gz;
    count++;
    delay(20);   // ~50 Hz pacing
  }

  for (int i = 0; i < 3; i++) {
    bias_acc[i]  = (float)(sum_acc[i]  / BIAS_N_SAMPLES);
    bias_gyro[i] = (float)(sum_gyro[i] / BIAS_N_SAMPLES);
  }

  // Remove expected gravity from Z-axis bias
  // (the sensor sits with +Z pointing up in typical flat placement)
  // Uncomment if needed: bias_acc[2] -= 1.0f;  // subtract 1g from Z

  // Seed gravity IIR with the average static acceleration
  gravity_est[0] = bias_acc[0];
  gravity_est[1] = bias_acc[1];
  gravity_est[2] = bias_acc[2];

  bias_calibrated = true;

  Serial.println("[IMU] Bias calibration complete:");
  Serial.print("  acc  bias [g]     : ");
  Serial.print(bias_acc[0], 4); Serial.print(", ");
  Serial.print(bias_acc[1], 4); Serial.print(", ");
  Serial.println(bias_acc[2], 4);
  Serial.print("  gyro bias [deg/s] : ");
  Serial.print(bias_gyro[0], 4); Serial.print(", ");
  Serial.print(bias_gyro[1], 4); Serial.print(", ");
  Serial.println(bias_gyro[2], 4);
}

/**
 * imu_read_sample(body_acc[3], body_gyro[3]) — Read one IMU sample.
 *
 * Applies:
 *   1. Bias subtraction (static offset removed during calibration)
 *   2. Gravity estimation via IIR low-pass filter (GRAVITY_LP_ALPHA)
 *   3. Gravity removal from accelerometer (body_acc = total - gravity_est)
 *   4. Gyroscope deg/s -> rad/s conversion
 *   5. z-score normalization using per-channel NORM_MEAN / NORM_STD
 *
 * Output units match UCI HAR training data:
 *   body_acc[i]  in normalized g units
 *   body_gyro[i] in normalized rad/s units
 *
 * Returns true if a new sample was available, false otherwise.
 */
inline bool imu_read_sample(float body_acc_norm[3], float body_gyro_norm[3]) {
  if (!IMU_ACCEL_AVAIL() || !IMU_GYRO_AVAIL()) return false;

  float ax, ay, az, gx, gy, gz;
  IMU_READ_ACCEL(ax, ay, az);
  IMU_READ_GYRO(gx, gy, gz);

  // 1. Subtract static bias
  float acc_raw[3]  = { ax - bias_acc[0],  ay - bias_acc[1],  az - bias_acc[2] };
  float gyro_raw[3] = { gx - bias_gyro[0], gy - bias_gyro[1], gz - bias_gyro[2] };

  // 2. Update gravity IIR estimate  (low-pass → static component)
  for (int i = 0; i < 3; i++) {
    gravity_est[i] = GRAVITY_LP_ALPHA * gravity_est[i]
                   + (1.0f - GRAVITY_LP_ALPHA) * acc_raw[i];
  }

  // 3. Remove gravity  (high-pass → dynamic / body acceleration)
  float body_acc_g[3];
  for (int i = 0; i < 3; i++) {
    body_acc_g[i] = acc_raw[i] - gravity_est[i];
  }

  // 4. Convert gyroscope deg/s -> rad/s
  float body_gyro_rads[3];
  for (int i = 0; i < 3; i++) {
    body_gyro_rads[i] = gyro_raw[i] * DEG_TO_RAD;
  }

  // 5. Z-score normalization  (match training/dataset.py)
  //    body_acc:  channels 0,1,2
  //    body_gyro: channels 3,4,5
  for (int i = 0; i < 3; i++) {
    body_acc_norm[i]  = (body_acc_g[i]    - NORM_MEAN[i])     / NORM_STD[i];
    body_gyro_norm[i] = (body_gyro_rads[i] - NORM_MEAN[3 + i]) / NORM_STD[3 + i];
  }

  return true;
}

/**
 * imu_read_magnetometer(mx, my, mz) — Optional magnetometer read.
 * Returns true if the magnetometer is available and reading succeeded.
 * Units: microtesla (uT).  Not used in inference — logged only.
 */
inline bool imu_read_magnetometer(float& mx, float& my, float& mz) {
  if (!imu_mag_ok) return false;
  if (!MAG_AVAIL()) return false;
  MAG_READ(mx, my, mz);
  return true;
}
