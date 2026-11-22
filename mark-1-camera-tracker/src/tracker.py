import cv2
import numpy as np
import math
import time
import csv
import json
import os
from datetime import datetime
from collections import deque
from simulator import VirtualCamera
from pid import PIDController
from kalman import KalmanFilter2D
from serial_interface import GimbalSerialController
from appearance import AppearanceModel

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
DEFAULT_USE_SIMULATOR = True  # Can be toggled live using the 'C' key
CAMERA_INDEX = 0              # 0 = default computer webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Optical scaling (Logitech C270 / standard webcam FOV: ~55 deg H, ~42 deg V)
FOV_H = 55.0
FOV_V = 42.0
PX_PER_DEG_X = FRAME_WIDTH / FOV_H
PX_PER_DEG_Y = FRAME_HEIGHT / FOV_V

# Gimbal Travel Limits (degrees) - prevents cable twist and mechanical binding
PAN_MIN = 15.0
PAN_MAX = 165.0
TILT_MIN = 30.0
TILT_MAX = 150.0
PAN_NEUTRAL = 90.0
TILT_NEUTRAL = 90.0

# Telemetry and Calibration paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "tracking_telemetry.csv")
CALIBRATION_PATH = os.path.join(DATA_DIR, "hsv_calibration.json")

# Default HSV Color Range for Light Pink target
LOWER_HSV = np.array([140, 35, 120])
UPPER_HSV = np.array([179, 200, 255])

# Load custom user calibration if available
if os.path.exists(CALIBRATION_PATH):
    try:
        with open(CALIBRATION_PATH, "r") as f:
            _cfg = json.load(f)
            LOWER_HSV = np.array(_cfg["lower"])
            UPPER_HSV = np.array(_cfg["upper"])
            print(f"[CALIBRATION] Loaded custom HSV bounds: Lower={LOWER_HSV}, Upper={UPPER_HSV}")
    except Exception as e:
        print(f"[WARNING] Could not parse calibration file: {e}")

MIN_TARGET_RADIUS = 12    # Ignore tiny specks of noise smaller than this (pixels)

# HUD Colors (BGR format)
HUD_WHITE = (235, 235, 235)
HUD_DIM = (165, 165, 165)
HUD_DARK = (20, 20, 20)
RETICLE_GREEN = (120, 255, 120)  # Light green for optical lock
RETICLE_AMBER = (60, 200, 255)   # Amber/yellow for Kalman coasting


def open_webcam(index=CAMERA_INDEX):
    """Attempts to open real computer webcam with MJPG fast streaming."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    return None


def init_csv():
    """Creates or updates the telemetry CSV file with complete headers."""
    headers = [
        "Timestamp", "Status", "Source", "Obj_X", "Obj_Y",
        "Frame_X", "Frame_Y", "Delta_X", "Delta_Y", "Error_Px",
        "Pan_Deg", "Tilt_Deg", "Vel_Pan", "Vel_Tilt"
    ]
    need_header = True
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="r", newline="") as f:
            first_line = f.readline()
            if "Source" in first_line and "Vel_Pan" in first_line:
                need_header = False

    if need_header:
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def draw_inverted_badge(img, text, x, y, bg_color=HUD_WHITE, text_color=HUD_DARK):
    """Draws a TGP-style solid highlight box with dark text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 5, 4
    cv2.rectangle(img, (x - pad_x, y - th - pad_y), (x + tw + pad_x, y + pad_y), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_tgp_frame(
    img, mode_label, tracking_state, elapsed_sec, fps, lock_pct,
    dx, dy, error_px, pan, tilt, vel_total, servo_status="MOCK",
    app_sim=1.0, range_m=None, target_label="OPT-PINK"
):
    """Draws a clean military Targeting Pod (TGP) perimeter HUD."""
    h, w = img.shape[:2]
    m = 18  # Outer margin
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Perimeter Rails
    cv2.line(img, (m, m), (110, m), HUD_WHITE, 1)
    cv2.line(img, (215, m), (285, m), HUD_WHITE, 1)
    cv2.line(img, (375, m), (435, m), HUD_WHITE, 1)
    cv2.line(img, (515, m), (w - m, m), HUD_WHITE, 1)

    cv2.line(img, (m, h - m), (120, h - m), HUD_WHITE, 1)
    cv2.line(img, (205, h - m), (285, h - m), HUD_WHITE, 1)
    cv2.line(img, (355, h - m), (445, h - m), HUD_WHITE, 1)
    cv2.line(img, (535, h - m), (w - m, h - m), HUD_WHITE, 1)

    cv2.line(img, (m, m), (m, 145), HUD_WHITE, 1)
    cv2.line(img, (m, 235), (m, h - m), HUD_WHITE, 1)
    cv2.line(img, (w - m, m), (w - m, 145), HUD_WHITE, 1)
    cv2.line(img, (w - m, 335), (w - m, h - m), HUD_WHITE, 1)

    # 2. Top Rail Labels
    cv2.putText(img, "BNE-MK1", (120, m + 5), font, 0.45, HUD_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, mode_label, (295, m + 5), font, 0.45, HUD_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"SRV:{servo_status}", (445, m + 5), font, 0.42, HUD_WHITE, 1, cv2.LINE_AA)

    # 3. Left Rail Status Badge
    if tracking_state == "LOCK":
        draw_inverted_badge(img, "LOCK", m + 6, 175, bg_color=RETICLE_GREEN, text_color=HUD_DARK)
        cv2.putText(img, "TRK", (m + 6, 212), font, 0.45, HUD_WHITE, 1, cv2.LINE_AA)
    elif tracking_state == "CST":
        draw_inverted_badge(img, "COAST", m + 4, 175, bg_color=RETICLE_AMBER, text_color=HUD_DARK)
        cv2.putText(img, "KF", (m + 6, 212), font, 0.45, RETICLE_AMBER, 1, cv2.LINE_AA)
    else:
        draw_inverted_badge(img, "SCAN", m + 6, 175, bg_color=HUD_WHITE, text_color=HUD_DARK)
        cv2.putText(img, "SRCH", (m + 6, 212), font, 0.45, HUD_DIM, 1, cv2.LINE_AA)

    # 4. Right Rail Telemetry
    cv2.putText(img, f"{fps:04.1f} FPS", (w - m - 78, 45), font, 0.42, HUD_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"IV {lock_pct:05.1f}%", (w - m - 78, 65), font, 0.42, HUD_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"PAN {pan:05.1f}", (w - m - 78, 85), font, 0.42, HUD_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"TLT {tilt:05.1f}", (w - m - 78, 105), font, 0.42, HUD_WHITE, 1, cv2.LINE_AA)
    if tracking_state in ("LOCK", "CST"):
        cv2.putText(img, f"V {vel_total:04.1f}D/S", (w - m - 82, 125), font, 0.38, HUD_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, f"APP {int(app_sim * 100):02d}%", (w - m - 82, 145), font, 0.38,
                    RETICLE_GREEN if app_sim > 0.70 else RETICLE_AMBER, 1, cv2.LINE_AA)
        if range_m is not None and range_m > 0:
            cv2.putText(img, f"RNG {range_m:04.2f}M", (w - m - 82, 165), font, 0.38, RETICLE_GREEN, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "RNG ----M", (w - m - 82, 165), font, 0.38, HUD_DIM, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "RNG ----M", (w - m - 82, 165), font, 0.38, HUD_DIM, 1, cv2.LINE_AA)

    # 5. Bottom Rail Indicators & Timer
    cv2.putText(img, target_label, (130, h - m + 5), font, 0.43, HUD_WHITE, 1, cv2.LINE_AA)
    draw_inverted_badge(img, "TGP", 302, h - m + 5, bg_color=HUD_WHITE, text_color=HUD_DARK)
    cv2.putText(img, "LOG:SPC", (455, h - m + 5), font, 0.43, HUD_DIM, 1, cv2.LINE_AA)

    mins = int(elapsed_sec // 60)
    secs = int(elapsed_sec % 60)
    cv2.putText(img, f"{mins:03d}:{secs:02d}", (w - m - 68, h - m - 14),
                font, 0.48, HUD_WHITE, 1, cv2.LINE_AA)

    # 6. Tracking Error Readout
    if tracking_state in ("LOCK", "CST"):
        err_color = RETICLE_GREEN if tracking_state == "LOCK" else RETICLE_AMBER
        cv2.putText(img, f"ERR {error_px:05.1f} PX", (m + 10, h - m - 32),
                    font, 0.43, err_color, 1, cv2.LINE_AA)
        cv2.putText(img, f"dX {dx:+04d}  dY {dy:+04d}", (m + 10, h - m - 14),
                    font, 0.42, HUD_WHITE, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "SCANNING AIRSPACE", (m + 10, h - m - 14),
                    font, 0.41, HUD_DIM, 1, cv2.LINE_AA)


def draw_tgp_boresight(img, cx, cy):
    """Draws the crisp open-center TGP crosshair."""
    gap = 10
    arm = 28
    cv2.line(img, (cx - arm, cy), (cx - gap, cy), HUD_WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (cx + arm, cy), HUD_WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - arm), (cx, cy - gap), HUD_WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, cy + arm), HUD_WHITE, 2, cv2.LINE_AA)


def draw_target_reticle(img, x, y, r, color=RETICLE_GREEN, thickness=2, is_prediction=False):
    """Draws clean targeting brackets around the locked or predicted object."""
    d = int(r * 1.35)
    l = max(int(r * 0.45), 8)

    # Corner brackets
    cv2.line(img, (x - d, y - d), (x - d + l, y - d), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x - d, y - d), (x - d, y - d + l), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + d, y - d), (x + d - l, y - d), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + d, y - d), (x + d, y - d + l), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x - d, y + d), (x - d + l, y + d), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x - d, y + d), (x - d, y + d - l), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + d, y + d), (x + d - l, y + d), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + d, y + d), (x + d, y + d - l), color, thickness, cv2.LINE_AA)

    if is_prediction:
        cv2.circle(img, (x, y), 3, color, 1, cv2.LINE_AA)
    else:
        cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)


def draw_lead_pipper(
    img, tgt_x: int, tgt_y: int, vel_pan: float, vel_tilt: float,
    px_per_deg_x: float, px_per_deg_y: float, lead_time: float = 0.50
) -> tuple[int, int]:
    """
    Renders an aerospace predictive lead pipper (future intercept reticle) on the TGP HUD.
    Projects target position forward along the Kalman velocity vector: p_lead = p_tgt + v * t_lead.
    """
    speed_deg = math.hypot(vel_pan, vel_tilt)
    if speed_deg < 2.0:
        return tgt_x, tgt_y  # Collapse lead indicator when target is nearly stationary

    # Forward-projected offset (angular -> pixel)
    dx_lead = (vel_pan * lead_time) * px_per_deg_x
    dy_lead = -(vel_tilt * lead_time) * px_per_deg_y  # Screen Y is inverted relative to elevation

    lead_x = int(round(tgt_x + dx_lead))
    lead_y = int(round(tgt_y + dy_lead))

    pipper_col = (80, 220, 240)  # Military tactical cyan reticle

    # 1. Segmented dashed lead trajectory vector
    segments = 6
    for i in range(0, segments, 2):
        t1 = i / segments
        t2 = (i + 1) / segments
        p1 = (int(round(tgt_x + dx_lead * t1)), int(round(tgt_y + dy_lead * t1)))
        p2 = (int(round(tgt_x + dx_lead * t2)), int(round(tgt_y + dy_lead * t2)))
        cv2.line(img, p1, p2, pipper_col, 1, cv2.LINE_AA)

    # 2. Predictive Lead Pipper Reticle (open center dot with crosshair tick marks)
    pipper_r = 8
    cv2.circle(img, (lead_x, lead_y), pipper_r, pipper_col, 1, cv2.LINE_AA)
    cv2.circle(img, (lead_x, lead_y), 2, pipper_col, -1, cv2.LINE_AA)

    tick = 4
    cv2.line(img, (lead_x, lead_y - pipper_r - tick), (lead_x, lead_y - pipper_r), pipper_col, 1, cv2.LINE_AA)
    cv2.line(img, (lead_x, lead_y + pipper_r), (lead_x, lead_y + pipper_r + tick), pipper_col, 1, cv2.LINE_AA)
    cv2.line(img, (lead_x - pipper_r - tick, lead_y), (lead_x - pipper_r, lead_y), pipper_col, 1, cv2.LINE_AA)
    cv2.line(img, (lead_x + pipper_r, lead_y), (lead_x + pipper_r + tick, lead_y), pipper_col, 1, cv2.LINE_AA)

    # 3. Lead Time Telemetry Tag
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        f"+{lead_time:.2f}S",
        (lead_x + pipper_r + 4, lead_y + 4),
        font,
        0.33,
        pipper_col,
        1,
        cv2.LINE_AA,
    )

    return lead_x, lead_y


def main():
    init_csv()

    use_simulator = DEFAULT_USE_SIMULATOR
    sim_cam = VirtualCamera(width=FRAME_WIDTH, height=FRAME_HEIGHT, target_fps=60)
    webcam = None

    def get_frame():
        nonlocal webcam, use_simulator
        if use_simulator:
            return sim_cam.read()
        else:
            if webcam is None or not webcam.isOpened():
                webcam = open_webcam()
                if webcam is None:
                    print("[WARNING] Could not open computer webcam. Reverting to Simulator.")
                    use_simulator = True
                    return sim_cam.read()
            ret, f = webcam.read()
            if not ret or f is None:
                print("[WARNING] Frame capture dropped from webcam. Reverting to Simulator.")
                use_simulator = True
                return sim_cam.read()
            # Resize if needed
            if f.shape[1] != FRAME_WIDTH or f.shape[0] != FRAME_HEIGHT:
                f = cv2.resize(f, (FRAME_WIDTH, FRAME_HEIGHT))
            return True, f

    # Closed-loop PID Controllers - Tuned for fast evasion pickup
    pan_pid = PIDController(
        kp=0.025, ki=0.006, kd=0.0012,
        deadband=1.0, output_limit=4.2,
        integral_limit=15.0, d_filter_alpha=0.20,
    )

    tilt_pid = PIDController(
        kp=0.025, ki=0.006, kd=0.0012,
        deadband=1.0, output_limit=4.2,
        integral_limit=15.0, d_filter_alpha=0.20,
    )

    # 2D Kalman Filter with Adaptive Maneuver Gating for high-g turns and boundary bounces
    kf = KalmanFilter2D(
        process_noise_accel=120.0,
        measurement_noise=0.30,
        max_coast_frames=75,
        gate_threshold=16.0,
        maneuver_gate_threshold=180.0,
    )

    # Hardware Serial Bridge to Arduino Nano + PCA9685 (auto-detects or runs in Virtual Mock)
    # Electro-Optical Appearance Fingerprinting & Scale Estimation Engine
    appearance = AppearanceModel()

    serial_gimbal = GimbalSerialController()

    current_pan = PAN_NEUTRAL
    current_tilt = TILT_NEUTRAL
    scan_sweep_dir = 1.0

    frame_times = deque(maxlen=30)
    start_t = time.perf_counter()
    prev_t = start_t

    total_frames = 0
    locked_frames = 0
    show_mask = False
    flir_mode = False
    occlusion_until = 0.0
    log_flash_until = 0.0
    target_radius = 22
    current_app_sim = 1.0
    show_pipper = True
    lead_time = 0.50

    # Interactive Click-to-Designate & Fullscreen state
    designated_click = None
    designation_flash_until = 0.0
    designated_color_name = "OPT-PINK"
    is_fullscreen = False

    # Stadiametric Rangefinder parameters (focal length for 55 deg FOV @ 640px = ~614.7px)
    focal_px = (FRAME_WIDTH / 2.0) / math.tan(math.radians(FOV_H / 2.0))
    TARGET_REAL_DIAMETER_M = 0.065  # Standard target reference diameter (6.5 cm ball)
    smoothed_range_m = 1.0

    lower_hsv = LOWER_HSV.copy()
    upper_hsv = UPPER_HSV.copy()

    WINDOW_NAME = "BNE Mark 1 | TGP Optical HUD"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    def on_mouse_click(event, x, y, flags, param):
        nonlocal designated_click
        if event == cv2.EVENT_LBUTTONDOWN:
            designated_click = (int(np.clip(x, 0, FRAME_WIDTH - 1)), int(np.clip(y, 0, FRAME_HEIGHT - 1)))

    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

    print("[BNE MARK 1] Optical Tracker Online.")
    print("[CONTROLS] Click = Lock Any Object | F = Fullscreen | C = Webcam/Sim | P = Pipper | T = Dynamics | D = Decoys | I = FLIR | R = Reset | Q = Quit")

    while True:
        ret, frame = get_frame()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        total_frames += 1
        height, width = frame.shape[:2]
        frame_cx, frame_cy = width // 2, height // 2

        # Process Operator Click-to-Lock Designation
        if designated_click is not None:
            cx, cy = designated_click
            designated_click = None

            px1 = max(0, cx - 10)
            py1 = max(0, cy - 10)
            px2 = min(width, cx + 10)
            py2 = min(height, cy + 10)
            patch = frame[py1:py2, px1:px2]

            if patch.size > 0:
                patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                med_h = int(np.median(patch_hsv[:, :, 0]))
                med_s = int(np.median(patch_hsv[:, :, 1]))
                med_v = int(np.median(patch_hsv[:, :, 2]))

                # Adaptive color margins
                lower_h = max(0, med_h - 18)
                upper_h = min(179, med_h + 18)
                lower_s = max(35, med_s - 65)
                upper_s = min(255, med_s + 75)
                lower_v = max(40, med_v - 75)
                upper_v = min(255, med_v + 75)

                lower_hsv = np.array([lower_h, lower_s, lower_v])
                upper_hsv = np.array([upper_h, upper_s, upper_v])

                if med_s < 45 and med_v > 180:
                    designated_color_name = "OPT-WHITE"
                elif med_v < 45:
                    designated_color_name = "OPT-DARK"
                elif med_h < 12 or med_h > 168:
                    designated_color_name = "OPT-RED"
                elif 12 <= med_h < 26:
                    designated_color_name = "OPT-ORANGE"
                elif 26 <= med_h < 35:
                    designated_color_name = "OPT-YELLOW"
                elif 35 <= med_h < 85:
                    designated_color_name = "OPT-GREEN"
                elif 85 <= med_h < 130:
                    designated_color_name = "OPT-BLUE"
                elif 130 <= med_h < 155:
                    designated_color_name = "OPT-PURPLE"
                else:
                    designated_color_name = "OPT-PINK"

                appearance.reset()
                appearance.update(frame, cx, cy, radius=18.0)

                meas_pan = current_pan + (cx - frame_cx) / PX_PER_DEG_X
                meas_tilt = current_tilt - (cy - frame_cy) / PX_PER_DEG_Y
                kf.reset()
                kf.update(meas_pan, meas_tilt)

                designation_flash_until = curr_t + 1.5
                print(f"[OPERATOR DESIGNATION] Locked on HSV({med_h}, {med_s}, {med_v}) -> Label: {designated_color_name}")

        curr_t = time.perf_counter()
        dt = max(min(curr_t - prev_t, 0.05), 1e-4)
        frame_times.append(curr_t - prev_t)
        prev_t = curr_t

        is_occluded = curr_t < occlusion_until

        # 1. Color Segmentation
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 2. Multi-Candidate Detection & Validation Gating (Aerospace NNDA)
        candidate_blobs = []
        rejected_clutter = []
        best_target = None
        optical_detected = False

        if contours and not is_occluded:
            for c in contours:
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                if radius >= MIN_TARGET_RADIUS:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        blob_x = int(M["m10"] / M["m00"])
                        blob_y = int(M["m01"] / M["m00"])
                        meas_pan = current_pan + (blob_x - frame_cx) / PX_PER_DEG_X
                        meas_tilt = current_tilt - (blob_y - frame_cy) / PX_PER_DEG_Y

                        # Extract 2D Hue-Saturation color distribution signature
                        cand_hist = appearance.extract_signature(frame, blob_x, blob_y, radius)
                        app_sim = appearance.compute_similarity(cand_hist) if appearance.is_initialized else 1.0

                        candidate_blobs.append({
                            "screen_x": blob_x,
                            "screen_y": blob_y,
                            "pos_x": meas_pan,
                            "pos_y": meas_tilt,
                            "radius": radius,
                            "appearance_similarity": app_sim,
                            "has_appearance": appearance.is_initialized,
                        })

        # 3. Kalman State Prediction & Multi-Feature Association Gating
        pred_pan, pred_tilt = kf.predict(dt)

        if candidate_blobs:
            best_target, rejected_clutter = kf.associate_and_gate(
                candidate_blobs, target_radius=target_radius
            )

        if best_target is not None:
            is_man = best_target.get("is_maneuvering", False)
            est_pan, est_tilt = kf.update(
                best_target["pos_x"], best_target["pos_y"], is_maneuvering=is_man
            )
            target_radius = int(best_target["radius"])

            # Stadiametric Optical Range Calculation
            raw_range_m = (focal_px * TARGET_REAL_DIAMETER_M) / max(2.0 * target_radius, 2.0)
            smoothed_range_m = float(np.clip(0.88 * smoothed_range_m + 0.12 * raw_range_m, 0.20, 25.0))

            # Online adaptive appearance learning on confirmed target
            appearance.update(frame, best_target["screen_x"], best_target["screen_y"], target_radius)
            current_app_sim = best_target.get("appearance_similarity", 1.0)
            tracking_state = "LOCK"
            locked_frames += 1
            optical_detected = True
        elif kf.is_coasting:
            est_pan, est_tilt = pred_pan, pred_tilt
            tracking_state = "CST"
        else:
            tracking_state = "SRCH"
            pan_pid.reset()
            tilt_pid.reset()
            kf.reset()
            appearance.reset()
            current_app_sim = 1.0

        # 4. Closed-Loop Tracking with Kinematic Velocity Feedforward
        dx, dy, error_px = 0, 0, -1.0
        tgt_x, tgt_y = -1, -1

        if tracking_state in ("LOCK", "CST"):
            tgt_x = int(round(frame_cx + (est_pan - current_pan) * PX_PER_DEG_X))
            tgt_y = int(round(frame_cy - (est_tilt - current_tilt) * PX_PER_DEG_Y))

            dx = tgt_x - frame_cx
            dy = tgt_y - frame_cy
            error_px = math.hypot(dx, dy)

            # Velocity Feedforward: target motion in this time step (degrees)
            # Injects 85% forward rate to bypass mechanical servo lag
            vel_x, vel_y = kf.velocity
            ff_pan = 0.85 * (vel_x * dt)
            ff_tilt = 0.85 * (vel_y * dt)

            d_pan = pan_pid.update(dx, dt, feedforward=ff_pan)
            d_tilt = tilt_pid.update(-dy, dt, feedforward=ff_tilt)

            current_pan = float(np.clip(current_pan + d_pan, PAN_MIN, PAN_MAX))
            current_tilt = float(np.clip(current_tilt + d_tilt, TILT_MIN, TILT_MAX))

            if use_simulator:
                sim_cam.set_gimbal_angles(current_pan, current_tilt)
            serial_gimbal.send_angles(current_pan, current_tilt)

        elif tracking_state == "SRCH":
            scan_speed = 35.0  # deg/s
            current_pan += scan_sweep_dir * (scan_speed * dt)
            if current_pan >= 145.0:
                current_pan = 145.0
                scan_sweep_dir = -1.0
            elif current_pan <= 35.0:
                current_pan = 35.0
                scan_sweep_dir = 1.0

            current_tilt += (TILT_NEUTRAL - current_tilt) * 0.1
            if use_simulator:
                sim_cam.set_gimbal_angles(current_pan, current_tilt)
            serial_gimbal.send_angles(current_pan, current_tilt)

        # 5. Display Frame Preparation & Velocity Extraction
        vel_pan, vel_tilt = kf.velocity
        vel_total = math.hypot(vel_pan, vel_tilt)

        if flir_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            display = frame

        # Draw rejected clutter/decoys with dim red/gray warning boxes
        font = cv2.FONT_HERSHEY_SIMPLEX
        for clut in rejected_clutter:
            csx = clut["screen_x"]
            csy = clut["screen_y"]
            cr = int(clut["radius"])
            cv2.rectangle(display, (csx - cr, csy - cr), (csx + cr, csy + cr), (75, 75, 75), 1, cv2.LINE_AA)
            cv2.putText(display, "REJ", (csx - cr, csy - cr - 4), font, 0.35, (80, 80, 220), 1, cv2.LINE_AA)

        # Draw Target Reticle & Tracking Vector
        if tracking_state in ("LOCK", "CST"):
            is_pred = (tracking_state == "CST")
            reticle_col = RETICLE_AMBER if is_pred else RETICLE_GREEN
            cv2.line(display, (frame_cx, frame_cy), (tgt_x, tgt_y), (80, 140, 80), 1, cv2.LINE_AA)
            draw_target_reticle(display, tgt_x, tgt_y, target_radius, color=reticle_col, is_prediction=is_pred)

            # Predictive Lead Indicator (Future Intercept Pipper)
            if show_pipper:
                draw_lead_pipper(
                    display, tgt_x, tgt_y, vel_pan, vel_tilt,
                    PX_PER_DEG_X, PX_PER_DEG_Y, lead_time=lead_time
                )

        # 6. Telemetry & FPS
        avg_fps = len(frame_times) / max(sum(frame_times), 1e-6)
        lock_pct = (locked_frames / total_frames) * 100.0
        elapsed_sec = curr_t - start_t

        # Mode label (e.g. SIM:CRU, SIM:EVA, SIM:TRB or CAM)
        if use_simulator:
            mode_label = f"SIM:{sim_cam.flight_mode[:3]}"
        else:
            mode_label = "CAM:REAL"

        servo_tag = "LINK" if serial_gimbal.is_connected else "MOCK"

        # 7. Draw HUD
        draw_tgp_boresight(display, frame_cx, frame_cy)
        draw_tgp_frame(
            display, mode_label, tracking_state, elapsed_sec, avg_fps, lock_pct,
            dx, dy, error_px, current_pan, current_tilt, vel_total,
            servo_status=servo_tag, app_sim=current_app_sim,
            range_m=smoothed_range_m, target_label=designated_color_name
        )

        if curr_t < designation_flash_until:
            draw_inverted_badge(display, f"DESIGNATED: {designated_color_name}", frame_cx - 100, 45,
                                bg_color=(240, 220, 80), text_color=HUD_DARK)

        if is_occluded:
            rem = occlusion_until - curr_t
            draw_inverted_badge(display, f"OCCLUSION PULSE ({rem:.1f}S)", frame_cx - 95, 66,
                                bg_color=RETICLE_AMBER, text_color=HUD_DARK)

        if use_simulator and sim_cam.show_decoys:
            draw_inverted_badge(display, "CLUTTER / DECOYS ACTIVE", frame_cx - 90, 87,
                                bg_color=(60, 60, 180), text_color=HUD_WHITE)

        if curr_t < log_flash_until:
            draw_inverted_badge(display, "EVENT LOGGED", frame_cx - 52, height - 42,
                                bg_color=RETICLE_GREEN, text_color=HUD_DARK)

        cv2.imshow(WINDOW_NAME, display)
        if show_mask:
            cv2.imshow("BNE Mark 1 | HSV Mask Debug", mask)

        # 8. Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            source_tag = "SIM" if use_simulator else "CAM"
            with open(CSV_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts, tracking_state, source_tag,
                    tgt_x, tgt_y, frame_cx, frame_cy, dx, dy, round(error_px, 2),
                    round(current_pan, 2), round(current_tilt, 2),
                    round(vel_pan, 2), round(vel_tilt, 2)
                ])
            log_flash_until = curr_t + 0.8
            print(f"[LOGGED] {ts} | {source_tag} | {tracking_state} | ERR: {error_px:.2f} px | VEL: {vel_total:.1f} d/s")
        elif key == ord("c"):
            # Toggle between Simulator and Real Webcam
            use_simulator = not use_simulator
            if not use_simulator and webcam is None:
                webcam = open_webcam()
            print(f"[FEED] Switched Video Source: {'SIMULATOR' if use_simulator else 'REAL WEBCAM'}")
        elif key == ord("t"):
            # Cycle simulator flight dynamics profile
            if use_simulator:
                new_mode = sim_cam.cycle_flight_mode()
                print(f"[TARGET] Flight Dynamics Profile: {new_mode}")
        elif key == ord("d"):
            # Toggle simulated clutter / decoys in simulator
            if use_simulator:
                d_state = sim_cam.toggle_decoys()
                print(f"[CLUTTER] Simulated Decoys: {'ACTIVE' if d_state else 'DISABLED'}")
        elif key == ord("p"):
            # Toggle Predictive Lead Pipper
            show_pipper = not show_pipper
            print(f"[HUD] Predictive Lead Pipper (+{lead_time:.2f}s): {'ENABLED' if show_pipper else 'DISABLED'}")
        elif key in (ord("f"), ord("F")):
            # Fullscreen toggle
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("[DISPLAY] Fullscreen: ON (Press 'F' again to exit)")
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
                print("[DISPLAY] Fullscreen: OFF (Standard 640x480)")
        elif key in (ord("i"), ord("I")):
            flir_mode = not flir_mode
            print(f"[OPTICS] FLIR / Infrared Mode: {'ON' if flir_mode else 'OFF'}")
        elif key == ord("o"):
            occlusion_until = curr_t + 1.2
            print("[TEST] 1.2s Occlusion Pulse triggered! Watch Kalman coasting.")
        elif key == ord("r"):
            current_pan = PAN_NEUTRAL
            current_tilt = TILT_NEUTRAL
            pan_pid.reset()
            tilt_pid.reset()
            kf.reset()
            appearance.reset()
            lower_hsv = LOWER_HSV.copy()
            upper_hsv = UPPER_HSV.copy()
            designated_color_name = "OPT-PINK"
            occlusion_until = 0.0
            serial_gimbal.send_angles(PAN_NEUTRAL, TILT_NEUTRAL)
            if use_simulator:
                sim_cam.reset_target()
            print("[RESET] Gimbal, target, and color thresholds reset to defaults.")
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("BNE Mark 1 | HSV Mask Debug")
        elif key == ord("q"):
            break

    sim_cam.release()
    if webcam is not None:
        webcam.release()
    serial_gimbal.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
