import cv2
import numpy as np
import csv
import math
import os
import sys
import time
from typing import List, Dict, Any

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "tracking_telemetry.csv")

VIEW_WIDTH = 640
VIEW_HEIGHT = 480
TIMELINE_HEIGHT = 80
TOTAL_HEIGHT = VIEW_HEIGHT + TIMELINE_HEIGHT

FRAME_CX = VIEW_WIDTH // 2
FRAME_CY = VIEW_HEIGHT // 2

FOV_H = 55.0
FOV_V = 42.0
PX_PER_DEG_X = VIEW_WIDTH / FOV_H
PX_PER_DEG_Y = VIEW_HEIGHT / FOV_V

# Military TGP HUD Color Palette (BGR)
HUD_DARK = (18, 18, 18)
HUD_PANEL = (28, 28, 30)
HUD_WHITE = (220, 220, 220)
HUD_DIM = (110, 110, 110)
RETICLE_GREEN = (0, 230, 115)
RETICLE_AMBER = (0, 190, 245)
PIPPER_CYAN = (240, 220, 80)
WARNING_RED = (80, 80, 230)


def load_telemetry_csv(filepath: str) -> List[Dict[str, Any]]:
    """Loads and validates telemetry records from the CSV file."""
    if not os.path.exists(filepath):
        return []

    records = []
    with open(filepath, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rec = {
                    "timestamp": row.get("Timestamp", "N/A"),
                    "status": row.get("Status", "LOCK"),
                    "source": row.get("Source", "SIM"),
                    "obj_x": int(float(row.get("Obj_X", FRAME_CX))),
                    "obj_y": int(float(row.get("Obj_Y", FRAME_CY))),
                    "dx": int(float(row.get("Delta_X", 0))),
                    "dy": int(float(row.get("Delta_Y", 0))),
                    "error_px": float(row.get("Error_Px", 0.0)),
                    "pan": float(row.get("Pan_Deg", 90.0)),
                    "tilt": float(row.get("Tilt_Deg", 90.0)),
                    "v_pan": float(row.get("Vel_Pan", 0.0)),
                    "v_tilt": float(row.get("Vel_Tilt", 0.0)),
                }
                records.append(rec)
            except (ValueError, KeyError):
                continue

    return records


def generate_demo_flight() -> List[Dict[str, Any]]:
    """Synthesizes a realistic 300-frame tactical flight trajectory for demo replay."""
    records = []
    t_start = 1732185600.0  # Synthetic epoch

    pan = 90.0
    tilt = 90.0
    v_pan = 22.0
    v_tilt = 14.0
    target_pan = 90.0
    target_tilt = 90.0

    for i in range(300):
        dt = 1.0 / 60.0
        sim_t = i * dt

        # Maneuver phases
        if i < 90:
            status = "LOCK"
            # Steady cruise
            cur_vx = v_pan
            cur_vy = v_tilt
        elif i < 180:
            status = "LOCK"
            # Evasive weave
            cur_vx = v_pan + 18.0 * math.sin(sim_t * 3.5)
            cur_vy = v_tilt + 12.0 * math.cos(sim_t * 2.8)
        elif i < 225:
            # Simulated occlusion pulse (target coasting)
            status = "COAST"
            cur_vx = v_pan
            cur_vy = v_tilt
        else:
            status = "LOCK"
            cur_vx = -v_pan
            cur_vy = -v_tilt

        target_pan += cur_vx * dt
        target_tilt += cur_vy * dt

        # Gimbal tracking lag
        pan += (target_pan - pan) * 0.75
        tilt += (target_tilt - tilt) * 0.75

        obj_x = int(round(FRAME_CX + (target_pan - pan) * PX_PER_DEG_X))
        obj_y = int(round(FRAME_CY - (target_tilt - tilt) * PX_PER_DEG_Y))
        dx = obj_x - FRAME_CX
        dy = obj_y - FRAME_CY
        err = math.hypot(dx, dy)

        records.append({
            "timestamp": f"DEMO-{sim_t:06.2f}s",
            "status": status,
            "source": "SIM:DEMO",
            "obj_x": obj_x,
            "obj_y": obj_y,
            "dx": dx,
            "dy": dy,
            "error_px": err,
            "pan": pan,
            "tilt": tilt,
            "v_pan": cur_vx,
            "v_tilt": cur_vy,
        })

    return records


def draw_hud_badge(img, text, x, y, bg_color=HUD_WHITE, text_color=HUD_DARK):
    """Draws a crisp solid highlight badge with inverted text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 5, 4
    cv2.rectangle(img, (x - pad_x, y - th - pad_y), (x + tw + pad_x, y + pad_y), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_tactical_replay_frame(rec: Dict[str, Any], frame_idx: int, total_frames: int, is_paused: bool, speed: float) -> np.ndarray:
    """Renders the complete tactical TGP HUD view and timeline scrub strip."""
    canvas = np.full((TOTAL_HEIGHT, VIEW_WIDTH, 3), 20, dtype=np.uint8)

    # 1. Background Grid & Horizon Line
    for gx in range(40, VIEW_WIDTH, 60):
        cv2.line(canvas, (gx, 0), (gx, VIEW_HEIGHT), (30, 30, 30), 1)
    for gy in range(40, VIEW_HEIGHT, 60):
        cv2.line(canvas, (0, gy), (VIEW_WIDTH, gy), (30, 30, 30), 1)

    # Horizon elevation bar (neutral 90 deg tilt)
    tilt_offset = int((rec["tilt"] - 90.0) * PX_PER_DEG_Y)
    horizon_y = FRAME_CY + tilt_offset
    if 0 <= horizon_y < VIEW_HEIGHT:
        cv2.line(canvas, (40, horizon_y), (VIEW_WIDTH - 40, horizon_y), (45, 45, 45), 1)

    # 2. Open-Center Boresight Crosshair
    gap = 10
    arm = 26
    cv2.line(canvas, (FRAME_CX - arm, FRAME_CY), (FRAME_CX - gap, FRAME_CY), HUD_WHITE, 2, cv2.LINE_AA)
    cv2.line(canvas, (FRAME_CX + gap, FRAME_CY), (FRAME_CX + arm, FRAME_CY), HUD_WHITE, 2, cv2.LINE_AA)
    cv2.line(canvas, (FRAME_CX, FRAME_CY - arm), (FRAME_CX, FRAME_CY - gap), HUD_WHITE, 2, cv2.LINE_AA)
    cv2.line(canvas, (FRAME_CX, FRAME_CY + gap), (FRAME_CX, FRAME_CY + arm), HUD_WHITE, 2, cv2.LINE_AA)

    # 3. Perimeter Rails
    m = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(canvas, (m, m), (VIEW_WIDTH - m, VIEW_HEIGHT - m), (50, 50, 50), 1)

    # Header Readouts
    cv2.putText(canvas, "BNE MARK 1 // TACTICAL REPLAY", (m + 8, m - 4), font, 0.42, HUD_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FRAME {frame_idx + 1:04d}/{total_frames:04d}", (VIEW_WIDTH - m - 130, m - 4), font, 0.40, HUD_DIM, 1, cv2.LINE_AA)

    # Status & Source Badges
    st = rec["status"]
    if st == "LOCK":
        draw_hud_badge(canvas, "LOCK", m + 8, 45, bg_color=RETICLE_GREEN, text_color=HUD_DARK)
    elif st == "COAST":
        draw_hud_badge(canvas, "COAST", m + 8, 45, bg_color=RETICLE_AMBER, text_color=HUD_DARK)
    else:
        draw_hud_badge(canvas, "SCAN", m + 8, 45, bg_color=HUD_WHITE, text_color=HUD_DARK)

    cv2.putText(canvas, rec["source"], (m + 8, 68), font, 0.38, HUD_DIM, 1, cv2.LINE_AA)

    # Gimbal Kinematics (Right Rail)
    cv2.putText(canvas, f"PAN {rec['pan']:05.1f}*", (VIEW_WIDTH - m - 85, 45), font, 0.40, HUD_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"TLT {rec['tilt']:05.1f}*", (VIEW_WIDTH - m - 85, 65), font, 0.40, HUD_WHITE, 1, cv2.LINE_AA)
    v_total = math.hypot(rec["v_pan"], rec["v_tilt"])
    cv2.putText(canvas, f"V {v_total:04.1f} D/S", (VIEW_WIDTH - m - 85, 85), font, 0.38, HUD_DIM, 1, cv2.LINE_AA)

    # 4. Target Reticle, Tracking Vector, and Predictive Lead Pipper
    tx = rec["obj_x"]
    ty = rec["obj_y"]
    if 0 <= tx < VIEW_WIDTH and 0 <= ty < VIEW_HEIGHT and st in ("LOCK", "COAST"):
        reticle_col = RETICLE_GREEN if st == "LOCK" else RETICLE_AMBER
        # Tracking Line from Boresight to Target
        cv2.line(canvas, (FRAME_CX, FRAME_CY), (tx, ty), (60, 110, 60), 1, cv2.LINE_AA)

        # Target Corner Brackets
        r = 16
        d = int(r * 1.3)
        l = 8
        cv2.line(canvas, (tx - d, ty - d), (tx - d + l, ty - d), reticle_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (tx - d, ty - d), (tx - d, ty - d + l), reticle_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (tx + d, ty - d), (tx + d - l, ty - d), reticle_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (tx + d, ty - d), (tx + d, ty - d + l), reticle_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (tx - d, ty + d), (tx - d + l, ty + d), reticle_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (tx - d, ty + d), (tx - d, ty + d - l), reticle_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (tx + d, ty + d), (tx + d - l, ty + d), reticle_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (tx + d, ty + d), (tx + d, ty + d - l), reticle_col, 2, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 2, reticle_col, -1, cv2.LINE_AA)

        # Predictive Lead Intercept Pipper (0.50s Forward Projection)
        if v_total > 2.0:
            lead_time = 0.50
            dx_lead = (rec["v_pan"] * lead_time) * PX_PER_DEG_X
            dy_lead = -(rec["v_tilt"] * lead_time) * PX_PER_DEG_Y
            pip_x = int(round(tx + dx_lead))
            pip_y = int(round(ty + dy_lead))

            # Segmented vector
            for seg in range(0, 6, 2):
                p1 = (int(round(tx + dx_lead * (seg / 6))), int(round(ty + dy_lead * (seg / 6))))
                p2 = (int(round(tx + dx_lead * ((seg + 1) / 6))), int(round(ty + dy_lead * ((seg + 1) / 6))))
                cv2.line(canvas, p1, p2, PIPPER_CYAN, 1, cv2.LINE_AA)

            # Pipper circle & tick marks
            cv2.circle(canvas, (pip_x, pip_y), 8, PIPPER_CYAN, 1, cv2.LINE_AA)
            cv2.circle(canvas, (pip_x, pip_y), 2, PIPPER_CYAN, -1, cv2.LINE_AA)
            cv2.line(canvas, (pip_x, pip_y - 12), (pip_x, pip_y - 8), PIPPER_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (pip_x, pip_y + 8), (pip_x, pip_y + 12), PIPPER_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (pip_x - 12, pip_y), (pip_x - 8, pip_y), PIPPER_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (pip_x + 8, pip_y), (pip_x + 12, pip_y), PIPPER_CYAN, 1, cv2.LINE_AA)
            cv2.putText(canvas, "+0.50S", (pip_x + 11, pip_y + 4), font, 0.32, PIPPER_CYAN, 1, cv2.LINE_AA)

    # Bottom Error Readout
    cv2.putText(canvas, f"ERR {rec['error_px']:05.1f} PX", (m + 8, VIEW_HEIGHT - m - 12), font, 0.42,
                RETICLE_GREEN if rec["error_px"] < 15.0 else RETICLE_AMBER, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"dX {rec['dx']:+04d}  dY {rec['dy']:+04d}", (m + 120, VIEW_HEIGHT - m - 12), font, 0.40, HUD_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, rec["timestamp"], (VIEW_WIDTH - m - 170, VIEW_HEIGHT - m - 12), font, 0.36, HUD_DIM, 1, cv2.LINE_AA)

    # 5. Timeline Scrubbing Strip (Bottom 80px)
    cv2.rectangle(canvas, (0, VIEW_HEIGHT), (VIEW_WIDTH, TOTAL_HEIGHT), HUD_PANEL, -1)
    cv2.line(canvas, (0, VIEW_HEIGHT), (VIEW_WIDTH, VIEW_HEIGHT), (60, 60, 65), 1)

    # Scrub bar track
    bar_x = 24
    bar_y = VIEW_HEIGHT + 28
    bar_w = VIEW_WIDTH - 48
    bar_h = 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 48), -1)

    # Progress fill
    prog_pct = (frame_idx / max(total_frames - 1, 1))
    fill_w = int(bar_w * prog_pct)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), RETICLE_GREEN, -1)

    # Scrubber thumb needle
    needle_x = bar_x + fill_w
    cv2.rectangle(canvas, (needle_x - 3, bar_y - 4), (needle_x + 3, bar_y + bar_h + 4), HUD_WHITE, -1)

    # Timeline Controls Legend
    play_tag = "PAUSED [STEP MODE]" if is_paused else f"PLAYING [{speed:.1f}x]"
    tag_col = RETICLE_AMBER if is_paused else RETICLE_GREEN
    cv2.putText(canvas, play_tag, (bar_x, VIEW_HEIGHT + 18), font, 0.38, tag_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[SPACE] Play/Pause | [,] / [.] Step | [-] / [+] Speed | [R] Rewind | [Q] Quit",
                (bar_x, TOTAL_HEIGHT - 12), font, 0.34, HUD_DIM, 1, cv2.LINE_AA)

    return canvas


def main():
    print("[BNE MARK 1] Tactical Flight Telemetry Replay Engine")
    print("-" * 60)

    records = load_telemetry_csv(CSV_PATH)
    if len(records) < 5 or "--demo" in sys.argv:
        print(f"[REPLAY] Loading tactical flight demonstration ({'--demo flag' if '--demo' in sys.argv else 'insufficient CSV records'})...")
        records = generate_demo_flight()
    else:
        print(f"[REPLAY] Loaded {len(records)} telemetry records from: {CSV_PATH}")

    window_name = "BNE Mark 1 | Tactical Flight Telemetry Replay"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    total_frames = len(records)
    current_frame = 0
    is_paused = False
    speed = 1.0

    def on_trackbar(val):
        nonlocal current_frame
        current_frame = int(np.clip(val, 0, total_frames - 1))

    cv2.createTrackbar("Timeline", window_name, 0, total_frames - 1, on_trackbar)

    last_step_t = time.perf_counter()

    while True:
        now = time.perf_counter()
        dt_frame = 1.0 / (60.0 * speed)

        if not is_paused and (now - last_step_t) >= dt_frame:
            current_frame = (current_frame + 1) % total_frames
            cv2.setTrackbarPos("Timeline", window_name, current_frame)
            last_step_t = now

        # Render current frame
        rec = records[current_frame]
        display = draw_tactical_replay_frame(rec, current_frame, total_frames, is_paused, speed)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(15) & 0xFF
        if key == ord(" "):
            is_paused = not is_paused
        elif key in (ord("."), ord("d"), 83):  # Right arrow or period
            current_frame = min(current_frame + 1, total_frames - 1)
            cv2.setTrackbarPos("Timeline", window_name, current_frame)
            is_paused = True
        elif key in (ord(","), ord("a"), 81):  # Left arrow or comma
            current_frame = max(current_frame - 1, 0)
            cv2.setTrackbarPos("Timeline", window_name, current_frame)
            is_paused = True
        elif key in (ord("+"), ord("=")):
            speed = min(speed * 2.0, 8.0)
            print(f"[REPLAY] Speed: {speed:.1f}x")
        elif key in (ord("-"), ord("_")):
            speed = max(speed / 2.0, 0.25)
            print(f"[REPLAY] Speed: {speed:.1f}x")
        elif key == ord("r"):
            current_frame = 0
            cv2.setTrackbarPos("Timeline", window_name, 0)
            print("[REPLAY] Rewound to beginning.")
        elif key in (ord("q"), 27):  # 'q' or ESC
            break

    cv2.destroyAllWindows()
    print("[REPLAY] Tactical Replay Terminated.")


if __name__ == "__main__":
    main()
