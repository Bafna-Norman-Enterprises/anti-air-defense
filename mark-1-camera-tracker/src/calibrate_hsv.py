import cv2
import numpy as np
import json
import os
from simulator import VirtualCamera

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(DATA_DIR, "hsv_calibration.json")

# Default starting values for Light Pink
DEFAULT_H_MIN = 140
DEFAULT_H_MAX = 179
DEFAULT_S_MIN = 35
DEFAULT_S_MAX = 200
DEFAULT_V_MIN = 120
DEFAULT_V_MAX = 255


def nothing(x):
    pass


def open_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap if cap.isOpened() else None


def main():
    print("=" * 60)
    print("   BAFNA NORMAN ENTERPRISES (BNE) — HSV CALIBRATION TOOL")
    print("=" * 60)
    print("Adjust the sliders until ONLY your target object appears white.")
    print("Press 'S' to SAVE to config | 'C' to toggle Webcam/Sim | 'Q' to QUIT.")
    print("-" * 60)

    # Always initialize both virtual camera and real webcam cleanly
    sim_cam = VirtualCamera(width=640, height=480, target_fps=60)
    webcam = None
    use_sim = True  # Default to Simulator (never turns on webcam unless 'C' is explicitly pressed)

    # Load existing config if available
    h_min, h_max = DEFAULT_H_MIN, DEFAULT_H_MAX
    s_min, s_max = DEFAULT_S_MIN, DEFAULT_S_MAX
    v_min, v_max = DEFAULT_V_MIN, DEFAULT_V_MAX

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                h_min, h_max = cfg["lower"][0], cfg["upper"][0]
                s_min, s_max = cfg["lower"][1], cfg["upper"][1]
                v_min, v_max = cfg["lower"][2], cfg["upper"][2]
                print(f"[LOADED] Found saved calibration: Lower={cfg['lower']}, Upper={cfg['upper']}")
        except Exception:
            pass

    window_name = "BNE Mark 1 | HSV Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar("H Min", window_name, h_min, 179, nothing)
    cv2.createTrackbar("H Max", window_name, h_max, 179, nothing)
    cv2.createTrackbar("S Min", window_name, s_min, 255, nothing)
    cv2.createTrackbar("S Max", window_name, s_max, 255, nothing)
    cv2.createTrackbar("V Min", window_name, v_min, 255, nothing)
    cv2.createTrackbar("V Max", window_name, v_max, 255, nothing)

    while True:
        if use_sim:
            ret, frame = sim_cam.read()
            source_label = "SIMULATOR (VIRTUAL PINK BALL)"
        else:
            if webcam is None or not webcam.isOpened():
                webcam = open_cam()
                if webcam is None:
                    print("[WARNING] Webcam unavailable. Reverting to simulator.")
                    use_sim = True
                    ret, frame = sim_cam.read()
                    source_label = "SIMULATOR"
                else:
                    ret, frame = webcam.read()
                    source_label = "WEBCAM (REAL ROOM OBJECT)"
            else:
                ret, frame = webcam.read()
                source_label = "WEBCAM (REAL ROOM OBJECT)"

            if not ret or frame is None:
                use_sim = True
                ret, frame = sim_cam.read()
                source_label = "SIMULATOR"
            else:
                frame = cv2.resize(frame, (640, 480))

        # Read trackbar values
        h_min = cv2.getTrackbarPos("H Min", window_name)
        h_max = cv2.getTrackbarPos("H Max", window_name)
        s_min = cv2.getTrackbarPos("S Min", window_name)
        s_max = cv2.getTrackbarPos("S Max", window_name)
        v_min = cv2.getTrackbarPos("V Min", window_name)
        v_max = cv2.getTrackbarPos("V Max", window_name)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Filter
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Cutout preview
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Stack mask side-by-side with preview (3 panels: Raw, Mask, Cutout)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([frame, mask_bgr, result])
        combined_small = cv2.resize(combined, (960, 240))

        cv2.putText(
            combined_small,
            f"SOURCE: {source_label} | Press 'C' to Toggle | Press 'S' to Save",
            (15, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined_small,
            f"LOWER: [{h_min}, {s_min}, {v_min}]  UPPER: [{h_max}, {s_max}, {v_max}]",
            (15, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, combined_small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            config = {
                "color_name": "Light Pink (Custom Calibrated)",
                "lower": [int(h_min), int(s_min), int(v_min)],
                "upper": [int(h_max), int(s_max), int(v_max)],
            }
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
            print(f"[SAVED] Calibration saved to: {CONFIG_PATH}")
            print(f"[VALUES] Lower: {config['lower']} | Upper: {config['upper']}")
        elif key == ord("c"):
            use_sim = not use_sim
            print(f"[SWITCH] Source switched to: {'SIMULATOR' if use_sim else 'WEBCAM'}")
        elif key == ord("q"):
            break

    sim_cam.release()
    if webcam is not None:
        webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
