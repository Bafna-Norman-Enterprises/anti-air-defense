import cv2
import numpy as np
import time
import math
from collections import deque


class VirtualCamera:
    """
    Precision frame-paced simulated webcam feed with a virtual 2-axis Pan-Tilt Gimbal.
    Accurately models the Logitech C270 Field of View (FOV) and MG90S servo travel limits (0-180 deg).
    Features high-visibility world reference stripes (Major 15 deg, Minor 5 deg, Horizon)
    and 3 flight dynamics profiles: CRUISE, EVASIVE, and TURBULENT.
    """

    FLIGHT_MODES = ["CRUISE", "EVASIVE", "TURBULENT"]

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        ball_radius: int = 22,
        target_fps: int = 60,
        fov_h: float = 55.0,
        fov_v: float = 42.0,
    ):
        self.width = width
        self.height = height
        self.ball_radius = ball_radius
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps

        # Optical specifications (Logitech C270 standard FOV)
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.px_per_deg_x = width / fov_h
        self.px_per_deg_y = height / fov_v

        # Gimbal orientation (degrees) - starts centered at 90.0 boresight
        self.cam_pan = 90.0
        self.cam_tilt = 90.0

        # Servo physical/software limits (degrees)
        self.pan_min = 15.0
        self.pan_max = 165.0
        self.tilt_min = 30.0
        self.tilt_max = 150.0

        # Target world position in angular space (degrees)
        self.target_pan = 90.0
        self.target_tilt = 90.0

        # World bounds for target motion (degrees)
        self.world_pan_min = 25.0
        self.world_pan_max = 155.0
        self.world_tilt_min = 35.0
        self.world_tilt_max = 145.0

        # Base target velocity (degrees/sec)
        self.base_v_pan = 24.0
        self.base_v_tilt = 18.0
        self.v_pan = self.base_v_pan
        self.v_tilt = self.base_v_tilt

        # Flight dynamics profile
        self.flight_mode_idx = 0
        self.sim_time = 0.0

        # Clocks for smooth physics and exact frame pacing
        self.last_time = time.perf_counter()
        self.next_frame_time = self.last_time

        # Pre-generate 8 noisy dark-gray backgrounds ONCE at startup for max performance
        self.noise_level = 6
        self.bg_pool = []
        base = np.full((self.height, self.width, 3), 32, dtype=np.int16)
        for _ in range(8):
            noise = np.random.randint(
                -self.noise_level, self.noise_level + 1, base.shape, dtype=np.int16
            )
            bg = np.clip(base + noise, 0, 255).astype(np.uint8)
            self.bg_pool.append(bg)
        self.bg_index = 0

    @property
    def flight_mode(self) -> str:
        return self.FLIGHT_MODES[self.flight_mode_idx]

    def cycle_flight_mode(self) -> str:
        """Cycles through target flight dynamics profiles: CRUISE -> EVASIVE -> TURBULENT."""
        self.flight_mode_idx = (self.flight_mode_idx + 1) % len(self.FLIGHT_MODES)
        return self.flight_mode

    def set_gimbal_angles(self, pan: float, tilt: float):
        """Commands the virtual gimbal to a given Pan and Tilt angle (degrees)."""
        self.cam_pan = float(np.clip(pan, self.pan_min, self.pan_max))
        self.cam_tilt = float(np.clip(tilt, self.tilt_min, self.tilt_max))

    def get_gimbal_angles(self) -> tuple[float, float]:
        """Returns the current (pan, tilt) angles of the gimbal mount in degrees."""
        return self.cam_pan, self.cam_tilt

    def reset_target(self):
        """Resets target position and gimbal orientation to 90.0 boresight."""
        self.target_pan = 90.0
        self.target_tilt = 90.0
        self.cam_pan = 90.0
        self.cam_tilt = 90.0
        self.v_pan = self.base_v_pan
        self.v_tilt = self.base_v_tilt
        self.sim_time = 0.0

    def _pace_frame(self):
        """Locks the simulator to target_fps using hybrid sleep + spin-wait."""
        self.next_frame_time += self.frame_duration
        now = time.perf_counter()
        remaining = self.next_frame_time - now

        if remaining > 0:
            if remaining > 0.002:
                time.sleep(remaining - 0.002)
            while time.perf_counter() < self.next_frame_time:
                pass
        else:
            self.next_frame_time = now

    def read(self):
        """
        Generates the next simulated video frame viewed from the gimbal's current orientation.
        Returns (True, frame) just like cv2.VideoCapture.read().
        """
        self._pace_frame()

        now = time.perf_counter()
        dt = min(now - self.last_time, 0.05)
        self.last_time = now
        self.sim_time += dt

        # Apply flight dynamics profile
        current_vx = self.v_pan
        current_vy = self.v_tilt

        if self.flight_mode == "EVASIVE":
            # High-G sinusoidal serpentine weave across both axes
            weave_x = 22.0 * math.sin(self.sim_time * 2.8)
            weave_y = 16.0 * math.cos(self.sim_time * 3.4)
            current_vx += weave_x
            current_vy += weave_y
        elif self.flight_mode == "TURBULENT":
            # Random atmospheric gust impulses
            gust_x = np.random.normal(0.0, 9.0)
            gust_y = np.random.normal(0.0, 7.0)
            current_vx += gust_x
            current_vy += gust_y

        # Update target position in world angular space
        self.target_pan += current_vx * dt
        self.target_tilt += current_vy * dt

        # Bounce target off world boundaries
        if self.target_pan <= self.world_pan_min:
            self.target_pan = self.world_pan_min
            self.v_pan = abs(self.base_v_pan)
        elif self.target_pan >= self.world_pan_max:
            self.target_pan = self.world_pan_max
            self.v_pan = -abs(self.base_v_pan)

        if self.target_tilt <= self.world_tilt_min:
            self.target_tilt = self.world_tilt_min
            self.v_tilt = abs(self.base_v_tilt)
        elif self.target_tilt >= self.world_tilt_max:
            self.target_tilt = self.world_tilt_max
            self.v_tilt = -abs(self.base_v_tilt)

        # Grab noisy background
        frame = self.bg_pool[self.bg_index].copy()
        self.bg_index = (self.bg_index + 1) % len(self.bg_pool)

        # ==========================================================
        # HIGH-VISIBILITY WORLD REFERENCE STRIPES (PAN & TILT GRID)
        # ==========================================================
        # 1. Minor Vertical Stripes (every 5 degrees)
        minor_color = (46, 46, 46)
        for w_pan in range(10, 175, 5):
            if w_pan % 15 != 0:  # Skip majors
                sx = int(round((self.width / 2.0) + (w_pan - self.cam_pan) * self.px_per_deg_x))
                if 0 <= sx < self.width:
                    # Subtle dashed line
                    for y_seg in range(0, self.height, 12):
                        cv2.line(frame, (sx, y_seg), (sx, min(y_seg + 6, self.height)), minor_color, 1)

        # 2. Major Vertical Stripes (every 15 degrees) - High Contrast
        major_color = (78, 78, 78)
        text_color = (135, 135, 135)
        for w_pan in range(15, 175, 15):
            sx = int(round((self.width / 2.0) + (w_pan - self.cam_pan) * self.px_per_deg_x))
            if 0 <= sx < self.width:
                cv2.line(frame, (sx, 0), (sx, self.height), major_color, 1)
                # Angle label near top
                cv2.putText(
                    frame,
                    f"{w_pan:03d}*",
                    (sx + 4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
                # Angle label near bottom
                cv2.putText(
                    frame,
                    f"{w_pan:03d}*",
                    (sx + 4, self.height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        # 3. Horizontal Elevation Rails & Horizon Line
        # Neutral Horizon (90 degrees Tilt) - Bold line
        sy_horizon = int(round((self.height / 2.0) - (90.0 - self.cam_tilt) * self.px_per_deg_y))
        if 0 <= sy_horizon < self.height:
            cv2.line(frame, (0, sy_horizon), (self.width, sy_horizon), (85, 85, 85), 1)
            cv2.putText(
                frame,
                "HORIZON 090*",
                (self.width - 110, sy_horizon - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

        # Other elevation rails every 15 degrees
        for w_tilt in range(30, 160, 15):
            if w_tilt != 90:
                sy = int(round((self.height / 2.0) - (w_tilt - self.cam_tilt) * self.px_per_deg_y))
                if 0 <= sy < self.height:
                    cv2.line(frame, (0, sy), (self.width, sy), (52, 52, 52), 1)
                    cv2.putText(
                        frame,
                        f"{w_tilt:03d}T",
                        (25, sy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (100, 100, 100),
                        1,
                        cv2.LINE_AA,
                    )

        # Project world target onto 2D camera viewport relative to gimbal orientation
        screen_x = (self.width / 2.0) + (self.target_pan - self.cam_pan) * self.px_per_deg_x
        screen_y = (self.height / 2.0) - (self.target_tilt - self.cam_tilt) * self.px_per_deg_y

        # Render target if within optical viewport (Light Pink)
        r = self.ball_radius
        if -r <= screen_x <= self.width + r and -r <= screen_y <= self.height + r:
            center = (int(round(screen_x)), int(round(screen_y)))
            # Anti-aliased light pink target ball (BGR: 193, 182, 255)
            cv2.circle(frame, center, r, (193, 182, 255), -1, lineType=cv2.LINE_AA)

            # Specular highlight
            highlight = (
                int(round(screen_x - r * 0.3)),
                int(round(screen_y - r * 0.3)),
            )
            cv2.circle(
                frame,
                highlight,
                int(r * 0.25),
                (235, 225, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

        return True, frame

    def release(self):
        pass


if __name__ == "__main__":
    cam = VirtualCamera(target_fps=60)
    frame_times = deque(maxlen=30)
    prev_t = time.perf_counter()

    print("[SIMULATOR] Running 60 FPS Virtual Pan-Tilt Camera test.")
    print("[CONTROLS] T = Cycle flight mode | Q = Quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        curr_t = time.perf_counter()
        frame_times.append(curr_t - prev_t)
        prev_t = curr_t
        avg_fps = len(frame_times) / max(sum(frame_times), 1e-6)

        cv2.putText(
            frame,
            f"SIM FPS: {avg_fps:.1f} | MODE: {cam.flight_mode} | PAN: {cam.cam_pan:.1f} TLT: {cam.cam_tilt:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Mark 1 | Virtual Pan-Tilt Camera (Raw)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("t"):
            mode = cam.cycle_flight_mode()
            print(f"[SIMULATOR] Switched Flight Dynamics: {mode}")
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
