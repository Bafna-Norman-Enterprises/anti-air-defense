# Bafna Norman Enterprises (BNE) — Mark 1 Hardware & Firmware Guide

This document contains the complete wiring schematic, hardware pinouts, power distribution architecture, and firmware flashing instructions for the **Mark 1 2-Axis Optical Tracking Gimbal**.

---

## 1. Hardware Architecture Overview

```
 [ PC Tracking Computer (Python) ]
              │  (USB Serial @ 115200 baud)
              ▼
    [ Arduino Nano (MCU) ]
              │  (I2C: A4=SDA, A5=SCL)
              ▼
   [ PCA9685 16-Ch PWM Driver ] ◄─── [ External 5V 3A Regulated PSU ]
     │                      │
     ▼ (Ch 0: PWM)          ▼ (Ch 1: PWM)
 [ MG90S Pan Servo ]   [ MG90S Tilt Servo ]
 (Azimuth 15-165°)     (Elevation 30-150°)
```

### Why This Dual-Controller Design?
1. **Zero CPU Overhead on Servo Jitter:** The Arduino Nano communicates with the PCA9685 over a fast 2-wire I2C bus ($400\text{ kHz}$). The PCA9685 has an onboard $27\text{ MHz}$ oscillator and dedicated 12-bit PWM hardware channels. This guarantees **zero servo twitching or jitter** even during heavy USB serial traffic.
2. **Dedicated Power Rail (Brownout Protection):** Servos can draw peak stall currents of up to $0.8\text{A}$ each. Attempting to power servos directly from the Arduino Nano's 5V pin causes severe voltage dips that reset the microcontroller. The PCA9685 has a dedicated high-current screw terminal for an external 5V 3A power supply with isolated power and shared ground.

---

## 2. Complete Wiring Schematic

### A. Arduino Nano to PCA9685 (Logic & I2C Control)

| Arduino Nano Pin | PCA9685 Pin | Description |
| :--- | :--- | :--- |
| **5V** | **VCC** | Logic Power (powers PCA9685 chip, NOT the servos) |
| **GND** | **GND** | Logic Common Ground |
| **A4** | **SDA** | I2C Data line |
| **A5** | **SCL** | I2C Clock line |

> [!IMPORTANT]
> The PCA9685 has two separate power pins:
> - **VCC:** Logic power for the chip itself ($3.3\text{V} - 5\text{V}$). Powered by the Arduino Nano 5V pin.
> - **V+ (Screw Terminal):** Motor/servo power ($5\text{V} - 6\text{V}$). **MUST be powered by the external 5V 3A power supply**, NEVER the Arduino!

---

### B. External Power Supply to PCA9685 (Servo Motor Power)

| External 5V 3A Supply | PCA9685 Screw Terminal | Description |
| :--- | :--- | :--- |
| **+5V (Positive)** | **V+** | Dedicated high-current servo supply rail |
| **GND (Negative)** | **GND** | Power Ground (**Must share common ground with Arduino GND**) |

---

### C. MG90S Micro-Servos to PCA9685 Header Pins

The PCA9685 has 16 sets of 3-pin headers. Each column has:
- **Top Row (Black/Brown):** Ground (`GND`)
- **Middle Row (Red):** Servo Power (`V+`)
- **Bottom Row (Yellow/Orange):** PWM Signal (`PWM`)

| Servo | PCA9685 Header | Wire Colors | Axis Function |
| :--- | :--- | :--- | :--- |
| **Pan Servo** | **Channel 0** | Orange (PWM), Red (V+), Brown (GND) | Azimuth ($15^\circ \text{–} 165^\circ$) |
| **Tilt Servo** | **Channel 1** | Orange (PWM), Red (V+), Brown (GND) | Elevation ($30^\circ \text{–} 150^\circ$) |

---

## 3. Firmware Flashing Guide (Arduino IDE)

### Step 1: Install Arduino IDE & Libraries
1. Download and install [Arduino IDE](https://www.arduino.cc/en/software) (version 2.0+ recommended).
2. Open the IDE and navigate to **Tools > Manage Libraries...** (or press `Ctrl + Shift + I`).
3. Search for: `Adafruit PWM Servo Driver`
4. Click **Install** (select *Install all dependencies* when prompted).

### Step 2: Open and Configure Firmware
1. Open the sketch in:
   `mark-1-camera-tracker/arduino/gimbal_firmware/gimbal_firmware.ino`
2. Connect your Arduino Nano to your computer via USB.
3. Under **Tools**, configure:
   - **Board:** `Arduino Nano`
   - **Processor:** `ATmega328P` *(Note: If upload fails or hangs, switch Processor to `ATmega328P (Old Bootloader)`)*
   - **Port:** Select your Arduino's COM port (e.g., `COM3`, `COM4`).

### Step 3: Compile and Upload
1. Click the **Upload** button (the arrow `→` icon in the top left).
2. Once the upload finishes, open the **Serial Monitor** (`Ctrl + Shift + M`).
3. Set the baud rate to **115200 baud**.
4. You should see the startup banner:
   ```text
   [BNE MARK 1] Gimbal Firmware Booting...
   [BNE MARK 1] PCA9685 Initialized. Gimbal Centered at (90, 90).
   [BNE MARK 1] Ready for Serial Control Packets.
   ```
5. Test a manual command packet by typing:
   `<P,110.0,T,075.0>`
   The Arduino will respond with:
   `<ACK,110.00,75.00>`
   and the servos will immediately slew to position!

---

## 4. Software Serial Integration

The Python tracker integrates automatically with the physical hardware:
- File: [src/serial_interface.py](file:///c:/Users/daksh/OneDrive/Documents/anti-air-defense/mark-1-camera-tracker/src/serial_interface.py)
- **Zero-Config Auto Discovery:** When plugged in, `serial_interface.py` scans your USB ports, finds the Arduino Nano, connects at 115,200 baud, and streams PID angles at 50 Hz.
- **Virtual Mock Fallback:** If no Arduino is connected, it gracefully prints a mock message and lets you continue testing in simulation without errors.
