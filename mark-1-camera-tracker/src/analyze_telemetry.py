import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "tracking_telemetry.csv")
OUTPUT_PLOT_PATH = os.path.join(DATA_DIR, "telemetry_analysis.png")


def load_telemetry():
    """Loads and parses tracking telemetry from CSV."""
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Telemetry file not found at: {CSV_PATH}")
        return None

    rows = []
    with open(CSV_PATH, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("[WARNING] Telemetry CSV is empty! Press SPACE in tracker.py to log telemetry events.")
        return None

    return rows


def analyze():
    data = load_telemetry()
    if not data:
        return

    # Extract columns
    timestamps = []
    errors = []
    pan_angles = []
    tilt_angles = []
    vel_totals = []
    states = []
    sources = []

    for r in data:
        try:
            ts = datetime.strptime(r["Timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            timestamps.append(ts)
            errors.append(float(r["Error_Px"]))
            pan_angles.append(float(r["Pan_Deg"]))
            tilt_angles.append(float(r["Tilt_Deg"]))
            vp = float(r.get("Vel_Pan", 0.0))
            vt = float(r.get("Vel_Tilt", 0.0))
            vel_totals.append(np.hypot(vp, vt))
            states.append(r.get("Status", "LOCK"))
            sources.append(r.get("Source", "SIM"))
        except Exception:
            continue

    if not timestamps:
        print("[WARNING] No valid telemetry records could be parsed.")
        return

    # Normalize time axis to seconds from start
    t0 = timestamps[0]
    time_sec = [(t - t0).total_seconds() for t in timestamps]

    # Calculate Key Performance Indicators (KPIs)
    mean_err = np.mean(errors)
    rms_err = np.sqrt(np.mean(np.array(errors) ** 2))
    max_err = np.max(errors)
    max_vel = np.max(vel_totals) if vel_totals else 0.0

    print("=" * 60)
    print("  BAFNA NORMAN ENTERPRISES (BNE) — MARK 1 TELEMETRY REPORT")
    print("=" * 60)
    print(f"Total Logged Events:   {len(timestamps)}")
    print(f"Elapsed Time Window:   {time_sec[-1]:.1f} seconds")
    print(f"Mean Tracking Error:   {mean_err:.2f} pixels")
    print(f"RMS Tracking Error:    {rms_err:.2f} pixels")
    print(f"Peak Tracking Error:   {max_err:.2f} pixels")
    print(f"Max Target Velocity:   {max_vel:.1f} deg/sec")
    print(f"Lock Rate:             {(states.count('LOCK') / len(states)) * 100:.1f}%")
    print("-" * 60)

    # Apply sleek dark aerospace tactical theme
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#111315")
    for ax in axes.flat:
        ax.set_facecolor("#181b1e")
        ax.grid(True, linestyle="--", alpha=0.25, color="#888888")

    # Panel 1: Pixel Tracking Error Over Time
    ax1 = axes[0, 0]
    ax1.plot(time_sec, errors, color="#58d68d", marker="o", markersize=4, linewidth=1.8, label="Tracking Error (px)")
    ax1.axhline(1.0, color="#f39c12", linestyle=":", label="PID Deadband (1 px)")
    ax1.set_title("Optical Tracking Error Over Time", fontsize=12, fontweight="bold", color="#ecf0f1", pad=10)
    ax1.set_xlabel("Elapsed Time (seconds)", color="#bdc3c7")
    ax1.set_ylabel("Pixel Distance from Boresight", color="#bdc3c7")
    ax1.legend(loc="upper right", facecolor="#222629", edgecolor="#444b50")

    # Panel 2: Gimbal Pan & Tilt Kinematics
    ax2 = axes[0, 1]
    ax2.plot(time_sec, pan_angles, color="#3498db", linewidth=1.8, marker="s", markersize=3, label="Pan Axis (deg)")
    ax2.plot(time_sec, tilt_angles, color="#9b59b6", linewidth=1.8, marker="^", markersize=3, label="Tilt Axis (deg)")
    ax2.axhline(90.0, color="#7f8c8d", linestyle="--", alpha=0.5, label="Neutral Boresight (90°)")
    ax2.set_title("Gimbal Servo Kinematics Trajectory", fontsize=12, fontweight="bold", color="#ecf0f1", pad=10)
    ax2.set_xlabel("Elapsed Time (seconds)", color="#bdc3c7")
    ax2.set_ylabel("Angle (degrees)", color="#bdc3c7")
    ax2.legend(loc="upper right", facecolor="#222629", edgecolor="#444b50")

    # Panel 3: Kalman Estimated Target Velocity Vector
    ax3 = axes[1, 0]
    ax3.plot(time_sec, vel_totals, color="#e67e22", linewidth=2.0, label="Estimated Velocity |V| (deg/s)")
    ax3.set_title("Kalman Target Velocity Estimation", fontsize=12, fontweight="bold", color="#ecf0f1", pad=10)
    ax3.set_xlabel("Elapsed Time (seconds)", color="#bdc3c7")
    ax3.set_ylabel("Target Speed (deg/sec)", color="#bdc3c7")
    ax3.legend(loc="upper right", facecolor="#222629", edgecolor="#444b50")

    # Panel 4: System State Distribution
    ax4 = axes[1, 1]
    unique_states = ["LOCK", "COAST", "SCAN"]
    counts = [states.count(s) for s in unique_states]
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    # Only keep non-zero states
    labels = [s for s, c in zip(unique_states, counts) if c > 0]
    clean_colors = [col for col, c in zip(colors, counts) if c > 0]
    clean_counts = [c for c in counts if c > 0]

    if clean_counts:
        wedges, texts, autotexts = ax4.pie(
            clean_counts, labels=labels, autopct="%1.1f%%",
            startangle=140, colors=clean_colors,
            textprops=dict(color="#ecf0f1")
        )
        for at in autotexts:
            at.set_color("#181b1e")
            at.set_fontweight("bold")
    ax4.set_title("Engagement State Distribution", fontsize=12, fontweight="bold", color="#ecf0f1", pad=10)

    plt.suptitle("BNE MARK 1 — OPTICAL TRACKING TELEMETRY ANALYSIS", fontsize=14, fontweight="bold", color="#ffffff", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(OUTPUT_PLOT_PATH, dpi=180, bbox_inches="tight")
    print(f"[SUCCESS] Telemetry report chart saved to: {OUTPUT_PLOT_PATH}")
    plt.close()


if __name__ == "__main__":
    analyze()
