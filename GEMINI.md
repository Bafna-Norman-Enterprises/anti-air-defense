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
- `mark-1-camera-tracker/src/pid.py` — **Done & Verified.** `PIDController` with deadband, integral anti-windup, derivative low-pass filtering, output rate limiting, and **Kinematic Velocity Feedforward** ($K_{ff} \cdot \vec{v}_{tgt}$) to cancel physical servo tracking lag on high-speed flybys.
- `mark-1-camera-tracker/src/appearance.py` — **Done & Verified.** `AppearanceModel` with 2D normalized Hue-Saturation histogram fingerprinting, Bhattacharyya similarity distance, and dynamic scale/aspect ratio tracking to defeat real-world false color clutter (shirts, reflections, orange/purple impostors).
- `mark-1-camera-tracker/src/kalman.py` — **Done & Verified.** `KalmanFilter2D` upgraded to a 6-state **Constant Acceleration (CA) Ballistic Model** ($[p_x, p_y, v_x, v_y, a_x, a_y]^T$) with continuous white-noise jerk covariance, Singer acceleration damping, two-tier Mahalanobis gating, single-candidate throw capture, and adaptive covariance inflation.
- `mark-1-camera-tracker/src/track_manager.py` — **Done & Verified.** Aerospace **Multi-Target Tracking (MTT) & Track-While-Scan (TWS)** engine maintaining up to 6 simultaneous tracks (`TRK-01`, `TRK-02`, ...), M/N confirmation logic (`3` hits to confirm, pruning 1-frame glare), **Track Quality Index (`TQI`)**, and **Kinematic Trajectory Classification** (`HOVER`, `CRUISE`, `HIGH-G`, `BALLISTIC`).
- `mark-1-camera-tracker/src/hybrid_tracker.py` — **Done & Verified.** `HybridFallbackTracker` combining sparse Lucas-Kanade Optical Flow (`FLOW`) and localized Normalized Cross-Correlation (`NCC`) template matching to hold lock when HSV color drops out in shadow/glare.
- `mark-1-camera-tracker/src/simulator.py` — **Done & Refined.** Precision frame-paced 640x480 `VirtualCamera` with virtual 2-axis pan-tilt gimbal kinematics, Logitech C270 FOV projection, high-contrast world reference stripes, 3 flight dynamics profiles (`CRUISE`, `EVASIVE`, `TURBULENT`), 3D distance/scale breathing (14px–34px), rolling-shutter directional motion blur, continuous aerodynamic momentum, and 4 heterogeneous clutter decoys.
- `mark-1-camera-tracker/src/tracker.py` — **Done & Refined.** Full sensor fusion loop with **Manual Fast-Shutter Exposure & White-Balance Lock** (`E`, `[` / `]`), **Hybrid Optical Flow / NCC Fallback** (`HYB` state), **2nd-order Ballistic Lead Pipper**, **Continuous Flight Recording** (`V` key), **Target Cycling (`N`) & No-Track Zones (`X`)**, **Anti-Glare Filter**, **Static Background Zeroing (`B`)**, **Fullscreen (`F`)**, **Click-to-Lock**, and hardware serial servo bridge.
- `mark-1-camera-tracker/src/calibrate_hsv.py` — **Done & Verified.** Interactive 6-slider real-time HSV color calibration utility with 3-panel diagnostics and configuration persistence.
- `mark-1-camera-tracker/src/serial_interface.py` — **Done & Verified.** Non-blocking 50 Hz USB serial bridge with framed command packets (`<P,xxx,T,yyy>`), auto COM-port discovery, and zero-hardware Virtual Mock fallback.
- `mark-1-camera-tracker/src/analyze_telemetry.py` — **Done & Verified.** Aerospace telemetry analysis dashboard computing RMS tracking error, state distribution, and plotting high-res graphs to `data/telemetry_analysis.png`.
- `mark-1-camera-tracker/src/replay_telemetry.py` — **Done & Verified.** Interactive visual telemetry replay engine with timeline scrubbing, frame-by-frame stepping (`.` / `,`), variable playback speed, lead pipper projection, and demo flight synthesis.
- `mark-1-camera-tracker/arduino/gimbal_firmware/gimbal_firmware.ino` — **Done & Verified.** Embedded C++ firmware for Arduino Nano + PCA9685 16-channel PWM driver with framed packet parser, MG90S travel limits ($15^\circ \text{–} 165^\circ$ Pan, $30^\circ \text{–} 150^\circ$ Tilt), and safety watchdog.
- `mark-1-camera-tracker/arduino/README.md` — **Done.** Complete hardware schematic, pinouts, power distribution architecture, and Arduino IDE flashing instructions.

## Advanced Aerospace Features Roadmap (Phase 2 & Phase 3)
1. [DONE] **Step 2.1: Target Association & Clutter Rejection:** Nearest-Neighbor Data Association (NNDA) with Mahalanobis Validation Gating ($\chi^2 \le 16.0$) to reject false pink objects/decoys (`D` key in simulator).
2. [DONE] **Step 2.2: Predictive Lead Indicator (Future Intercept Pipper):** Compute forward-projected intercept point ($t_{\text{lead}} = 0.5\text{s}$) using Kalman velocity vector and draw a military-style lead pipper reticle on the TGP HUD (`P` key to toggle).
3. [DONE] **Step 2.3: Tactical Flight Telemetry Replay:** Interactive visual replay tool (`replay_telemetry.py`) with timeline scrubbing to analyze logged flights frame-by-frame.
4. [DONE] **Step 3.1: Interactive Click-to-Designate:** Click *any* physical object on screen to auto-sample HSV, re-fingerprint appearance model, and lock on it live with saturation guard ($S \ge 45$) and red wrap-around.
5. Commit and push progress to GitHub (`Bafna-Norman-Enterprises/anti-air-defense`).