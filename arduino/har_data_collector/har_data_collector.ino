/**
 * har_data_collector.ino — 9-axis IMU data collection sketch
 *
 * Arduino Nano 33 BLE Sense (nRF52840 / LSM9DS1)
 * Streams all 9 IMU axes over USB-Serial for offline HAR dataset building.
 *
 * ─── Serial Protocol (115200 baud, line-terminated with '\n') ───────────────
 *
 *  PC → Arduino commands:
 *    PING              — connection test
 *    INFO              — device info (sample rates, full-scale ranges)
 *    CALIB <N>         — calibrate bias over N still samples (default 100)
 *    START             — begin streaming data lines
 *    STOP              — stop streaming
 *
 *  Arduino → PC responses:
 *    PONG                            — reply to PING
 *    INFO fs=50 accel_fs=4g gyro_fs=2000dps mag_fs=400uT
 *    CALIB_START <N>                 — calibration begins
 *    CALIB_DONE <ax> <ay> <az> <gx> <gy> <gz>  — mean bias values (raw units)
 *    READY                           — after CALIB_DONE, ready to collect
 *    HEADER timestamp_ms total_ax_g total_ay_g total_az_g ...  — column names
 *    D,<ts>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<mx>,<my>,<mz>      — data sample
 *    STOPPED <N>                     — streaming stopped, N = samples sent
 *
 * ─── Data format ─────────────────────────────────────────────────────────────
 *  Each data line: D,timestamp_ms,ax,ay,az,gx,gy,gz,mx,my,mz
 *
 *  Units:
 *    total_ax/ay/az : total acceleration including gravity  [g]
 *    gyro_x/y/z     : angular velocity                      [deg/s]
 *    mag_x/y/z      : magnetic field                        [µT]
 *    timestamp_ms   : millis() at sample time               [ms]
 *
 *  NOTE: values are RAW (bias not subtracted on-device).
 *  Bias subtraction and gravity removal are done in Python preprocessing
 *  (data/preprocess_collected.py) using the CALIB_DONE bias values.
 *
 * ─── LED indicators ──────────────────────────────────────────────────────────
 *  LED_BUILTIN solid ON   → idle / ready
 *  LED_BUILTIN slow blink → calibrating (1 Hz)
 *  LED_BUILTIN fast blink → collecting data (5 Hz)
 *  LEDR (red)             → error / not ready
 *
 * ─── Libraries required ──────────────────────────────────────────────────────
 *  Arduino_LSM9DS1  (Rev1)  — or —  Arduino_LSM6DS3 + Arduino_LIS2MDL (Rev2)
 */

// ─── Board selection ──────────────────────────────────────────────────────────
// #define NANO33_REV2   // uncomment for Rev2 (LSM6DS3 + LIS2MDL)

#ifdef NANO33_REV2
  #include <Arduino_LSM6DS3.h>
  #include <Arduino_LIS2MDL.h>
  #define MAG_BEGIN()       MAG.begin()
  #define MAG_AVAIL()       MAG.magneticFieldAvailable()
  #define MAG_READ(x,y,z)   MAG.readMagneticField(x,y,z)
  static bool has_mag = false;
#else
  #include <Arduino_LSM9DS1.h>
  #define MAG_BEGIN()       true   // built into LSM9DS1
  #define MAG_AVAIL()       IMU.magneticFieldAvailable()
  #define MAG_READ(x,y,z)   IMU.readMagneticField(x,y,z)
  static bool has_mag = true;
#endif

// ─── Configuration ────────────────────────────────────────────────────────────
static const int   BAUD_RATE        = 115200;
static const int   TARGET_HZ        = 50;
static const unsigned long SAMPLE_INTERVAL_US = 1000000UL / TARGET_HZ; // 20000 µs
static const int   DEFAULT_CALIB_N  = 100;   // ~2 s at 50 Hz
static const int   CMD_BUF_SIZE     = 64;

// ─── State machine ────────────────────────────────────────────────────────────
enum State { IDLE, CALIBRATING, COLLECTING };
static State     g_state      = IDLE;
static bool      g_calibrated = false;

// ─── Bias storage (raw sensor units) ─────────────────────────────────────────
static double g_bias_ax = 0, g_bias_ay = 0, g_bias_az = 0;
static double g_bias_gx = 0, g_bias_gy = 0, g_bias_gz = 0;

// ─── Sampling state ───────────────────────────────────────────────────────────
static unsigned long g_last_sample_us = 0;
static unsigned long g_sample_count   = 0;   // total samples sent this session

// ─── Magnetometer cache (updated when available, slower than acc/gyro) ────────
static float g_mx = 0, g_my = 0, g_mz = 0;
static bool  g_mag_valid = false;

// ─── Serial command buffer ────────────────────────────────────────────────────
static char  g_cmd_buf[CMD_BUF_SIZE];
static int   g_cmd_len = 0;

// ─── LED helpers ─────────────────────────────────────────────────────────────
static void led_idle()       { digitalWrite(LED_BUILTIN, HIGH); }
static void led_off()        { digitalWrite(LED_BUILTIN, LOW); }
static void led_blink_tick(unsigned long period_ms) {
  unsigned long t = millis() % period_ms;
  digitalWrite(LED_BUILTIN, t < (period_ms / 2) ? HIGH : LOW);
}

// ─── Forward declarations ─────────────────────────────────────────────────────
static void handle_command(const char* cmd);
static void do_calibrate(int n_samples);
static void do_start_collecting();
static void do_stop_collecting();
static void send_data_sample();
static void poll_magnetometer();

// ═════════════════════════════════════════════════════════════════════════════
// setup()
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial && millis() < 5000);

  pinMode(LED_BUILTIN, OUTPUT);
  led_idle();

#ifdef LEDR
  pinMode(LEDR, OUTPUT);
  digitalWrite(LEDR, HIGH);  // active-low on Nano 33; HIGH = off
#endif

  // ── IMU init
  if (!IMU.begin()) {
    Serial.println("ERROR IMU_INIT_FAILED");
#ifdef LEDR
    digitalWrite(LEDR, LOW);   // turn red LED on
#endif
    while (true) { led_blink_tick(200); delay(10); }
  }

  // ── Magnetometer init (Rev2 only)
#ifdef NANO33_REV2
  has_mag = MAG_BEGIN();
#endif

  // ── Announce
  Serial.println("READY");
  Serial.print("INFO fs=");
  Serial.print(TARGET_HZ);
  Serial.print(" accel_samplerate_hz=");
  Serial.print((int)IMU.accelerationSampleRate());
  Serial.print(" gyro_samplerate_hz=");
  Serial.print((int)IMU.gyroscopeSampleRate());
  Serial.print(" has_mag=");
  Serial.println(has_mag ? "1" : "0");
}

// ═════════════════════════════════════════════════════════════════════════════
// loop()
// ═════════════════════════════════════════════════════════════════════════════
void loop() {
  // ── 1. Read and process any incoming serial command characters
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (g_cmd_len > 0) {
        g_cmd_buf[g_cmd_len] = '\0';
        handle_command(g_cmd_buf);
        g_cmd_len = 0;
      }
    } else if (g_cmd_len < CMD_BUF_SIZE - 1) {
      g_cmd_buf[g_cmd_len++] = c;
    }
  }

  // ── 2. Poll magnetometer whenever new data is ready (it runs slower)
  poll_magnetometer();

  // ── 3. LED indicator
  if      (g_state == CALIBRATING) led_blink_tick(1000);
  else if (g_state == COLLECTING)  led_blink_tick(200);
  else                              led_idle();

  // ── 4. Pace sampling to TARGET_HZ
  if (g_state != COLLECTING) return;

  unsigned long now_us = micros();
  if (now_us - g_last_sample_us < SAMPLE_INTERVAL_US) return;
  g_last_sample_us = now_us;

  // ── 5. Emit one data sample
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    send_data_sample();
    g_sample_count++;
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// handle_command() — parse and dispatch a received command line
// ═════════════════════════════════════════════════════════════════════════════
static void handle_command(const char* cmd) {
  // Trim leading whitespace
  while (*cmd == ' ') cmd++;

  if (strncmp(cmd, "PING", 4) == 0) {
    Serial.println("PONG");

  } else if (strncmp(cmd, "INFO", 4) == 0) {
    Serial.print("INFO fs="); Serial.print(TARGET_HZ);
    Serial.print(" accel_hz="); Serial.print((int)IMU.accelerationSampleRate());
    Serial.print(" gyro_hz=");  Serial.print((int)IMU.gyroscopeSampleRate());
    Serial.print(" has_mag=");  Serial.println(has_mag ? "1" : "0");

  } else if (strncmp(cmd, "CALIB", 5) == 0) {
    // Optional argument: number of calibration samples
    int n = DEFAULT_CALIB_N;
    const char* arg = cmd + 5;
    while (*arg == ' ') arg++;
    if (*arg != '\0') n = atoi(arg);
    if (n < 10)  n = 10;
    if (n > 500) n = 500;
    do_calibrate(n);

  } else if (strncmp(cmd, "START", 5) == 0) {
    if (g_state == COLLECTING) {
      Serial.println("WARN already_collecting");
    } else {
      do_start_collecting();
    }

  } else if (strncmp(cmd, "STOP", 4) == 0) {
    if (g_state == COLLECTING) {
      do_stop_collecting();
    } else {
      Serial.println("WARN not_collecting");
    }

  } else {
    Serial.print("WARN unknown_cmd: ");
    Serial.println(cmd);
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// do_calibrate() — measure static bias over n_samples
// ═════════════════════════════════════════════════════════════════════════════
static void do_calibrate(int n_samples) {
  if (g_state == COLLECTING) {
    do_stop_collecting();
  }
  g_state = CALIBRATING;

  Serial.print("CALIB_START ");
  Serial.println(n_samples);

  double sum_ax = 0, sum_ay = 0, sum_az = 0;
  double sum_gx = 0, sum_gy = 0, sum_gz = 0;
  int count = 0;

  unsigned long t_last = micros();

  while (count < n_samples) {
    // Pace at TARGET_HZ
    unsigned long now = micros();
    if (now - t_last < SAMPLE_INTERVAL_US) {
      delay(1);
      continue;
    }
    t_last = now;

    if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) continue;

    float ax, ay, az, gx, gy, gz;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    sum_ax += ax; sum_ay += ay; sum_az += az;
    sum_gx += gx; sum_gy += gy; sum_gz += gz;
    count++;

    led_blink_tick(1000);   // slow blink during calibration
  }

  g_bias_ax = sum_ax / n_samples;
  g_bias_ay = sum_ay / n_samples;
  g_bias_az = sum_az / n_samples;
  g_bias_gx = sum_gx / n_samples;
  g_bias_gy = sum_gy / n_samples;
  g_bias_gz = sum_gz / n_samples;

  g_calibrated = true;
  g_state = IDLE;

  // Send bias values (raw sensor units: g for acc, deg/s for gyro)
  Serial.print("CALIB_DONE ");
  Serial.print(g_bias_ax, 6); Serial.print(" ");
  Serial.print(g_bias_ay, 6); Serial.print(" ");
  Serial.print(g_bias_az, 6); Serial.print(" ");
  Serial.print(g_bias_gx, 6); Serial.print(" ");
  Serial.print(g_bias_gy, 6); Serial.print(" ");
  Serial.println(g_bias_gz, 6);

  Serial.println("READY");
  led_idle();
}

// ═════════════════════════════════════════════════════════════════════════════
// do_start_collecting() — begin data stream
// ═════════════════════════════════════════════════════════════════════════════
static void do_start_collecting() {
  g_sample_count   = 0;
  g_last_sample_us = micros();
  g_state          = COLLECTING;

  // Print column header once so Python can parse it
  Serial.println(
    "HEADER "
    "timestamp_ms,"
    "total_ax_g,total_ay_g,total_az_g,"
    "gyro_x_degs,gyro_y_degs,gyro_z_degs,"
    "mag_x_uT,mag_y_uT,mag_z_uT"
  );
  Serial.println("STREAMING");
}

// ═════════════════════════════════════════════════════════════════════════════
// do_stop_collecting() — end data stream
// ═════════════════════════════════════════════════════════════════════════════
static void do_stop_collecting() {
  g_state = IDLE;
  Serial.print("STOPPED ");
  Serial.println(g_sample_count);
  led_idle();
}

// ═════════════════════════════════════════════════════════════════════════════
// send_data_sample() — emit one CSV data line
// Format: D,timestamp_ms,ax,ay,az,gx,gy,gz,mx,my,mz
// ═════════════════════════════════════════════════════════════════════════════
static void send_data_sample() {
  float ax, ay, az, gx, gy, gz;
  IMU.readAcceleration(ax, ay, az);   // g
  IMU.readGyroscope(gx, gy, gz);     // deg/s

  // Build output in a char buffer to minimise Serial.print() overhead
  char buf[128];
  snprintf(
    buf, sizeof(buf),
    "D,%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f",
    millis(),
    (double)ax, (double)ay, (double)az,
    (double)gx, (double)gy, (double)gz,
    (double)(g_mag_valid ? g_mx : 0.0f),
    (double)(g_mag_valid ? g_my : 0.0f),
    (double)(g_mag_valid ? g_mz : 0.0f)
  );
  Serial.println(buf);
}

// ═════════════════════════════════════════════════════════════════════════════
// poll_magnetometer() — update cached mag reading whenever new data is ready
// The LSM9DS1 magnetometer runs at 20 Hz; acc/gyro run at 104 Hz.
// We cache the latest mag value and attach it to acc/gyro data lines.
// ═════════════════════════════════════════════════════════════════════════════
static void poll_magnetometer() {
  if (!has_mag) return;
  if (!MAG_AVAIL()) return;
  MAG_READ(g_mx, g_my, g_mz);
  g_mag_valid = true;
}
