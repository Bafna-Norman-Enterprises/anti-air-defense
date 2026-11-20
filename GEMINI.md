# Bafna Norman Enterprises — Anti-Air Defense (Mark 1)

## Working Style & Rules
- I am a beginner learning as I build. Go slowly, one step at a time, and wait for my go-ahead before moving on.
- Always briefly explain *why* we make specific engineering decisions.
- Goal: Make the tracking software as accurate, precise, smooth, and realistic as possible within our hardware limits.
- Naming: Use **Mark 1** (not "Stage 1"). Organization name is **Bafna Norman Enterprises (BNE)**.
- Scope: Strictly non-weaponized camera observation, tracking, and on-screen CSV event logging.

## Hardware Plan (Not Purchased Yet — Testing in Simulation)
- **Camera:** Logitech C270 720p USB webcam (target 640x480 @ 30 FPS, MJPG, manual exposure).
- **Controller:** Arduino Nano + PCA9685 16-channel PWM driver.
- **Actuators:** 2x MG90S 9G micro servos (pan & tilt) powered separately by a regulated 5V 3A supply with common ground.
- **Mount:** Custom 3D-printed 2-axis pan-and-tilt mount designed in Fusion.
- **Budget:** $88.40 estimated / $100.00 cap.

## Current Repo Structure & Progress
- `mark-1-camera-tracker/requirements.txt` — `opencv-python`, `numpy`, `matplotlib`, `pyserial` (running in Python 3.15.2 `venv`).
- `mark-1-camera-tracker/src/pid.py` — **Done & Verified.** `PIDController` with deadband, integral anti-windup, derivative low-pass filtering, and output rate limiting (tested critically damped, 0.75s settle time, ~0 overshoot).
- `mark-1-camera-tracker/src/kalman.py` — **Done & Verified.** `KalmanFilter2D` with Constant Velocity (CV) physical motion model, Line-of-Sight (LOS) angular state estimation, continuous velocity tracking, and trajectory coasting during visual dropouts.
- `mark-1-camera-tracker/src/simulator.py` — **Done & Refined.** Precision frame-paced 640x480 `VirtualCamera` with virtual 2-axis pan-tilt gimbal kinematics, Logitech C270 FOV projection, high-contrast world reference stripes (major 15°, minor 5°, horizon), and 3 flight dynamics profiles (`CRUISE`, `EVASIVE`, `TURBULENT`, cycled via `T` key).
- `mark-1-camera-tracker/src/tracker.py` — **Done & Refined.** Full sensor fusion loop with dual-axis PID control, live TGP HUD telemetry (`LOCK` / `COAST` / `SCAN`), autonomous airspace search sweep on lost targets, timed 1.2s occlusion test (`O` key), instant boresight reset (`R` key), live real computer webcam toggle (`C` key), and hardware serial servo bridge.
- `mark-1-camera-tracker/src/calibrate_hsv.py` — **Done & Verified.** Interactive 6-slider real-time HSV color calibration utility with 3-panel diagnostics and configuration persistence.
- `mark-1-camera-tracker/src/serial_interface.py` — **Done & Verified.** Non-blocking 50 Hz USB serial bridge with framed command packets (`<P,xxx,T,yyy>`), auto COM-port discovery, and zero-hardware Virtual Mock fallback.
- `mark-1-camera-tracker/src/analyze_telemetry.py` — **Done & Verified.** Aerospace telemetry analysis dashboard computing RMS tracking error, state distribution, and plotting high-res graphs to `data/telemetry_analysis.png`.
- `mark-1-camera-tracker/arduino/gimbal_firmware/gimbal_firmware.ino` — **Done & Verified.** Embedded C++ firmware for Arduino Nano + PCA9685 16-channel PWM driver with framed packet parser, MG90S travel limits ($15^\circ \text{–} 165^\circ$ Pan, $30^\circ \text{–} 150^\circ$ Tilt), and safety watchdog.
- `mark-1-camera-tracker/arduino/README.md` — **Done.** Complete hardware schematic, pinouts, power distribution architecture, and Arduino IDE flashing instructions.

## Next Steps (In Order)
1. [DONE] Upgrade `simulator.py` with a virtual 2-axis pan-tilt gimbal and wire `pid.py` into `tracker.py`.
2. [DONE] Add Kalman filter prediction for smoother high-speed tracking and safe "target lost" hold behavior.
3. [DONE] Add real webcam toggle (`C` key), evasive flight profiles (`T` key), and automated telemetry dashboard (`analyze_telemetry.py`).
4. [DONE] Write the Arduino Nano + PCA9685 serial firmware and hardware documentation in `mark-1-camera-tracker/arduino/`.
5. Commit and push progress to GitHub (`Bafna-Norman-Enterprises/anti-air-defense`).