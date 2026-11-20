import numpy as np


class KalmanFilter2D:
    """
    2D Linear Kalman Filter for target tracking and velocity estimation.
    Uses a Constant Velocity (CV) physical motion model:
        State vector:        x = [pos_x, pos_y, vel_x, vel_y]^T
        Measurement vector:  z = [pos_x, pos_y]^T

    Provides smooth state estimation, velocity tracking, and trajectory
    coasting through temporary visual dropouts/occlusions.
    """

    def __init__(
        self,
        process_noise_accel: float = 25.0,  # Expected target acceleration variance (units/s^2)
        measurement_noise: float = 2.0,      # Sensor/centroid noise standard deviation (units)
        max_coast_frames: int = 45,          # Coast for ~0.75s @ 60 FPS before declaring lost
    ):
        self.process_noise_accel = process_noise_accel
        self.measurement_noise = measurement_noise
        self.max_coast_frames = max_coast_frames

        # State vector: [px, py, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float64)

        # State covariance matrix P: initial uncertainty
        self.P = np.diag([50.0, 50.0, 100.0, 100.0]).astype(np.float64)

        # Measurement matrix H: maps state to measurement (extracts px, py)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * (self.measurement_noise ** 2)

        # State transition identity template
        self.I = np.eye(4, dtype=np.float64)

        self.initialized = False
        self.frames_since_update = 0

    def reset(self):
        """Resets the filter state and clears memory."""
        self.x.fill(0.0)
        self.P = np.diag([50.0, 50.0, 100.0, 100.0]).astype(np.float64)
        self.initialized = False
        self.frames_since_update = 0

    def predict(self, dt: float) -> tuple[float, float]:
        """
        Predicts target state at the next time step based on the physical motion model.
        Returns predicted (pos_x, pos_y).
        """
        if not self.initialized:
            return 0.0, 0.0

        # 1. State Transition Matrix F:
        # px_new = px + vx * dt
        # py_new = py + vy * dt
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 2. Continuous White Noise Acceleration Process Covariance Q:
        # Models target maneuvers and physical disturbances
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q_var = self.process_noise_accel

        Q = np.array([
            [dt4 / 4.0, 0.0,       dt3 / 2.0, 0.0      ],
            [0.0,       dt4 / 4.0, 0.0,       dt3 / 2.0],
            [dt3 / 2.0, 0.0,       dt2,       0.0      ],
            [0.0,       dt3 / 2.0, 0.0,       dt2      ],
        ], dtype=np.float64) * q_var

        # 3. Project state forward: x_pred = F * x
        self.x = F @ self.x

        # 4. Project error covariance: P_pred = F * P * F^T + Q
        self.P = F @ self.P @ F.T + Q

        self.frames_since_update += 1

        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x: float, z_y: float) -> tuple[float, float]:
        """
        Corrects predicted state using a new noisy optical measurement (z_x, z_y).
        Returns filtered (pos_x, pos_y).
        """
        z = np.array([[z_x], [z_y]], dtype=np.float64)

        # If first measurement, initialize state directly
        if not self.initialized:
            self.x[0, 0] = z_x
            self.x[1, 0] = z_y
            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0
            self.P = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float64)
            self.initialized = True
            self.frames_since_update = 0
            return z_x, z_y

        # 1. Innovation (measurement residual): y = z - H * x_pred
        y = z - (self.H @ self.x)

        # 2. Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # 3. Optimal Kalman Gain: K = P * H^T * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 4. Updated state estimate: x = x_pred + K * y
        self.x = self.x + (K @ y)

        # 5. Updated error covariance: P = (I - K * H) * P
        self.P = (self.I - (K @ self.H)) @ self.P

        self.frames_since_update = 0

        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def is_coasting(self) -> bool:
        """True if filter is predicting through visual dropouts within the max limit."""
        return self.initialized and (0 < self.frames_since_update <= self.max_coast_frames)

    @property
    def is_lost(self) -> bool:
        """True if target has been missing longer than max coast duration."""
        return not self.initialized or (self.frames_since_update > self.max_coast_frames)

    @property
    def velocity(self) -> tuple[float, float]:
        """Returns estimated target velocity (vx, vy) in units/sec."""
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def position(self) -> tuple[float, float]:
        """Returns current best position estimate (px, py)."""
        return float(self.x[0, 0]), float(self.x[1, 0])


# ==========================================
# STANDALONE VERIFICATION BENCHMARK
# ==========================================
if __name__ == "__main__":
    print("[KALMAN TEST] Simulating noisy target path with a 10-frame visual dropout...")
    print("-" * 65)

    kf = KalmanFilter2D(process_noise_accel=15.0, measurement_noise=3.0, max_coast_frames=20)
    dt = 1.0 / 60.0

    true_pos = 100.0
    true_vel = 50.0  # px/s

    for frame in range(1, 31):
        true_pos += true_vel * dt
        # Dropout between frame 15 and 24
        is_occluded = 15 <= frame <= 24

        pred_x, _ = kf.predict(dt)

        if not is_occluded:
            noise = np.random.normal(0, 3.0)
            meas_x = true_pos + noise
            filt_x, _ = kf.update(meas_x, 0.0)
            status = "MEAS"
        else:
            filt_x = pred_x
            status = f"COAST ({kf.frames_since_update:02d})"

        vx, _ = kf.velocity
        if frame in (1, 5, 10, 14, 15, 18, 22, 24, 25, 30):
            print(
                f"Frame {frame:02d} | True: {true_pos:6.2f} | "
                f"Est: {filt_x:6.2f} | Vel: {vx:+5.1f} px/s | State: {status}"
            )
