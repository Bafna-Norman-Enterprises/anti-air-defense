/*
 * ============================================================================
 * BAFNA NORMAN ENTERPRISES (BNE) — ANTI-AIR DEFENSE (MARK 1)
 * Gimbal Actuator Controller Firmware
 * Hardware: Arduino Nano (ATmega328P) + PCA9685 16-Channel 12-Bit PWM Driver
 * Actuators: 2x TowerPro MG90S 9G Micro Servos (Pan & Tilt)
 *
 * SERIAL PROTOCOL:
 * Baud Rate: 115200 bps
 * Command Packet: <P,xxx.xx,T,yyy.yy>\n
 *   Example:  <P,090.0,T,090.0>\n
 * Response:   <ACK,xxx.xx,yyy.yy>\n
 * ============================================================================
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ---------------------------------------------------------------------------
// HARDWARE CHANNELS & PIN DEFINITIONS
// ---------------------------------------------------------------------------
#define PAN_SERVO_CHANNEL   0   // PCA9685 Channel 0 = Pan (Azimuth)
#define TILT_SERVO_CHANNEL  1   // PCA9685 Channel 1 = Tilt (Elevation)
#define STATUS_LED_PIN      13  // Onboard Arduino LED (blinks on packet receive)

// ---------------------------------------------------------------------------
// SERVO CALIBRATION & PHYSICAL SAFETY LIMITS (MG90S)
// ---------------------------------------------------------------------------
// Standard MG90S pulse widths (microseconds)
#define SERVO_US_MIN        544   // Approx pulse width for 0 degrees (us)
#define SERVO_US_MAX        2400  // Approx pulse width for 180 degrees (us)
#define SERVO_PWM_FREQ      50    // 50 Hz standard analog/digital servo update rate

// Software safety stops to protect 3D-printed linkages and prevent cable twisting
const float PAN_MIN_LIMIT   = 15.0;
const float PAN_MAX_LIMIT   = 165.0;
const float TILT_MIN_LIMIT  = 30.0;
const float TILT_MAX_LIMIT  = 150.0;

const float BORESIGHT_PAN   = 90.0;
const float BORESIGHT_TILT  = 90.0;

// Watchdog timeout: park/hold if no packet received within 2500ms
const unsigned long WATCHDOG_TIMEOUT_MS = 2500;

// ---------------------------------------------------------------------------
// GLOBAL STATE & OBJECTS
// ---------------------------------------------------------------------------
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

float currentPan  = BORESIGHT_PAN;
float currentTilt = BORESIGHT_TILT;

unsigned long lastPacketTime = 0;
bool targetLocked = false;

// Serial parsing buffer
const byte BUFFER_SIZE = 32;
char serialBuffer[BUFFER_SIZE];
byte bufferIndex = 0;
bool receivingPacket = false;

// ---------------------------------------------------------------------------
// HELPER: CONVERT DEGREES (0-180) TO MICROSECONDS FOR PCA9685
// ---------------------------------------------------------------------------
uint16_t angleToMicroseconds(float angle) {
  angle = constrain(angle, 0.0, 180.0);
  float us = SERVO_US_MIN + (angle / 180.0) * (SERVO_US_MAX - SERVO_US_MIN);
  return (uint16_t)round(us);
}

void commandServos(float panAngle, float tiltAngle) {
  // Enforce mechanical travel limits
  currentPan  = constrain(panAngle, PAN_MIN_LIMIT, PAN_MAX_LIMIT);
  currentTilt = constrain(tiltAngle, TILT_MIN_LIMIT, TILT_MAX_LIMIT);

  uint16_t panUs  = angleToMicroseconds(currentPan);
  uint16_t tiltUs = angleToMicroseconds(currentTilt);

  pwm.writeMicroseconds(PAN_SERVO_CHANNEL, panUs);
  pwm.writeMicroseconds(TILT_SERVO_CHANNEL, tiltUs);
}

// ---------------------------------------------------------------------------
// PARSE INCOMING FRAMED SERIAL PACKET: <P,090.0,T,090.0>
// ---------------------------------------------------------------------------
void processPacket(char* packet) {
  // Expected format: P,<float>,T,<float>
  char* pToken = strstr(packet, "P,");
  char* tToken = strstr(packet, "T,");

  if (pToken != NULL && tToken != NULL) {
    float newPan  = atof(pToken + 2);
    float newTilt = atof(tToken + 2);

    commandServos(newPan, newTilt);
    lastPacketTime = millis();
    digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN)); // Toggle LED

    // Send acknowledgment back to Python
    Serial.print(F("<ACK,"));
    Serial.print(currentPan, 2);
    Serial.print(F(","));
    Serial.print(currentTilt, 2);
    Serial.println(F(">"));
  }
}

// ---------------------------------------------------------------------------
// SETUP
// ---------------------------------------------------------------------------
void setup() {
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  Serial.begin(115200);
  while (!Serial && millis() < 3000); // Wait for USB connection (up to 3s)

  Serial.println(F("[BNE MARK 1] Gimbal Firmware Booting..."));

  Wire.begin();
  pwm.begin();
  pwm.setOscillatorFrequency(27000000); // Internal 27MHz calibrated oscillator
  pwm.setPWMFreq(SERVO_PWM_FREQ);

  // Initialize servos to neutral boresight (90 deg, 90 deg)
  commandServos(BORESIGHT_PAN, BORESIGHT_TILT);
  lastPacketTime = millis();

  Serial.println(F("[BNE MARK 1] PCA9685 Initialized. Gimbal Centered at (90, 90)."));
  Serial.println(F("[BNE MARK 1] Ready for Serial Control Packets."));
}

// ---------------------------------------------------------------------------
// MAIN LOOP
// ---------------------------------------------------------------------------
void loop() {
  // 1. Read non-blocking incoming serial bytes
  while (Serial.available() > 0) {
    char inChar = Serial.read();

    if (inChar == '<') {
      // Packet start delimiter
      receivingPacket = true;
      bufferIndex = 0;
      serialBuffer[0] = '\0';
    } 
    else if (inChar == '>') {
      // Packet end delimiter
      if (receivingPacket) {
        serialBuffer[bufferIndex] = '\0';
        processPacket(serialBuffer);
        receivingPacket = false;
      }
    } 
    else if (receivingPacket) {
      if (bufferIndex < BUFFER_SIZE - 1) {
        serialBuffer[bufferIndex++] = inChar;
      } else {
        // Buffer overflow: discard packet
        receivingPacket = false;
        bufferIndex = 0;
      }
    }
  }

  // 2. Safety Watchdog Check
  if (millis() - lastPacketTime > WATCHDOG_TIMEOUT_MS) {
    // If no signal received from tracking computer for > 2.5s, slow pulse LED
    if ((millis() / 500) % 2 == 0) {
      digitalWrite(STATUS_LED_PIN, HIGH);
    } else {
      digitalWrite(STATUS_LED_PIN, LOW);
    }
  }
}
