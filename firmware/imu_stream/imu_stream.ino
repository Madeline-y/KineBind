#include <Arduino.h>
#include <LSM6DS3.h>
#include <Wire.h>

// XIAO nRF52840 Sense, Seeed nRF52 mbed-enabled Boards, LSM6DS3 library.
LSM6DS3 imu(I2C_MODE, 0x6A);
constexpr uint32_t SAMPLE_INTERVAL_US = 20000;  // 50 Hz USB output.
uint32_t nextSampleUs = 0;
uint32_t sequence = 0;
bool imuReady = false;
bool streaming = false;

void setup() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  // The three user LEDs are active LOW; keep them off during acquisition.
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  Serial.begin(115200);
  // The sensor refreshes at 104 Hz; the host receives approximately 50 Hz.
  imu.settings.accelSampleRate = 104;
  imu.settings.gyroSampleRate = 104;
  imu.settings.accelRange = 4;    // +/-4 g, including gravity.
  imu.settings.gyroRange = 500;   // +/-500 degrees/second.
  imuReady = (imu.begin() == 0);
}

void loop() {
  if (!Serial) {
    streaming = false;
    delay(1);
    return;
  }
  if (!imuReady) {
    Serial.println("# ERROR: IMU initialization failed; check Sense board selection");
    delay(1000);
    return;
  }
  if (!streaming) {
    Serial.println("# KineBind IMU v1; output_hz=50; accel_range_g=4; gyro_range_dps=500");
    Serial.println("timestamp_ms,sequence,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
    nextSampleUs = micros();
    streaming = true;
  }
  const uint32_t nowUs = micros();
  if (static_cast<int32_t>(nowUs - nextSampleUs) < 0) {
    delay(1);
    return;
  }
  // Skip missed slots rather than emitting a burst of stale catch-up samples.
  const uint32_t skipped = (nowUs - nextSampleUs) / SAMPLE_INTERVAL_US;
  nextSampleUs += (skipped + 1) * SAMPLE_INTERVAL_US;
  const uint32_t timestampMs = millis();
  const float values[6] = {
    imu.readFloatAccelX(), imu.readFloatAccelY(), imu.readFloatAccelZ(),
    imu.readFloatGyroX(), imu.readFloatGyroY(), imu.readFloatGyroZ()
  };
  Serial.print(timestampMs);
  Serial.print(',');
  Serial.print(sequence++);
  for (const float value : values) {
    Serial.print(',');
    Serial.print(value, 4);
  }
  Serial.println();
}
