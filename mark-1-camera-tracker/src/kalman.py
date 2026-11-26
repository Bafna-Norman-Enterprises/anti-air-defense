import numpy as np
from typing import Optional


class KalmanFilter2D:
    """
    2D Linear Kalman Filter with Mahalanobis Validation Gating and Nearest-Neighbor Data Association.
    Uses a Constant Velocity (CV) physical motion model:
        State vector:        x = [pos_x, pos_y, vel_x, vel_y]^T
        Measurement vector:  z = [pos_x, pos_y]^T

    Provides smooth state estimation, velocity tracking, trajectory coasting,
    and statistical clutter/decoy rejection using Chi-Squared validation gates.
    """

    def __init__(
        self,
        process_noise_accel: float = 120.0,  # Target acceleration variance (units/s^2)
        measurement_noise: float = 0.30,     # Sensor/centroid noise standard deviation (degrees)
        max_coast_frames: int = 75,          # Coast for ~1.25s @ 60 FPS before declaring lost
        gate_threshold: float = 16.0,        # Nominal Chi-squared gate (~4-sigma statistical distance)
        maneuver_gate_threshold: float = 180.0, # Expanded validation gate for high-g maneuvers/bounces
    ):
        self.process_noise_accel = process_noise_accel
        self.measurement_noise = measurement_noise
        self.max_coast_frames = max_coast_frames
        self.gate_threshold = gate_threshold
        self.maneuver_gate_threshold = maneuver_gate_threshold

        # State vector: [px, py, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float64)

        # State covariance matrix P: initial uncertainty
        self.P = np.diag([50.0, 50.0, 100.0, 100.0]).astype(np.float64)

        # Measurement matrix H: extracts px, py
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * (self.measurement_noise ** 2)

        # State transition identity template
        self.I = np.eye(4, dtype=np.float64)

        # Innovation covariance S and its inverse (computed during predict)
        self.S = np.eye(2, dtype=np.float64)
        self.S_inv = np.eye(2, dtype=np.float64)

        self.initialized = False
        self.frames_since_update = 0

    def reset(self):
        """Resets filter state and clears track memory."""
        self.x.fill(0.0)
        self.P = np.diag([50.0, 50.0, 100.0, 100.0]).astype(np.float64)
        self.initialized = False
        self.frames_since_update = 0
        self.S = np.eye(2, dtype=np.float64)
        self.S_inv = np.eye(2, dtype=np.float64)

    def predict(self, dt: float) -> tuple[float, float]:
        """
        Projects target state forward in time using the physical Constant Velocity motion model.
        Also calculates the innovation covariance S for validation gating.
        Returns predicted (pos_x, pos_y).
        """
        if not self.initialized:
            return 0.0, 0.0

        # 1. State Transition Matrix F
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 2. Continuous White Noise Acceleration Covariance Q
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

        # 5. Precompute Innovation Covariance S and S_inv for Mahalanobis Gating
        self.S = self.H @ self.P @ self.H.T + self.R
        try:
            self.S_inv = np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            self.S_inv = np.linalg.pinv(self.S)

        self.frames_since_update += 1

        return float(self.x[0, 0]), float(self.x[1, 0])

    def associate_and_gate(
        self,
        candidates: list[dict],
        target_radius: float = 22.0,
    ) -> tuple[Optional[dict], list[dict]]:
        """
        Aerospace Nearest-Neighbor Data Association with Adaptive Maneuver Validation Gating.
        Evaluates candidate detections against the predicted target state.

        Candidate format: {'pos_x': float, 'pos_y': float, 'radius': float, ...}
        Returns:
            (associated_target_candidate or None, list_of_rejected_clutter_candidates)
        """
        if not candidates:
            return None, []

        # If track is uninitialized, prefer high-motion/high-saturation target over static clutter,
        # falling back to candidate closest to boresight center if no acquisition score is provided.
        if not self.initialized:
            if 'acq_score' in candidates[0]:
                best = max(candidates, key=lambda c: c['acq_score'])
            else:
                best = min(candidates, key=lambda c: (c.get('screen_x', 320) - 320)**2 + (c.get('screen_y', 240) - 240)**2)
            clutter = [c for c in candidates if c is not best]
            return best, clutter

        pred_pos = np.array([[self.x[0, 0]], [self.x[1, 0]]], dtype=np.float64)

        # Two-Tier Gating:
        # Tier 1: Nominal Gate (chi^2 <= gate_threshold)
        # Tier 2: Maneuver Gate (expanded gate for high-g turns/bounces when no nominal match exists)
        nominal_matches = []
        maneuver_matches = []
        clutter_candidates = []

        for cand in candidates:
            z = np.array([[cand['pos_x']], [cand['pos_y']]], dtype=np.float64)
            y = z - pred_pos  # Innovation vector

            # Mahalanobis Distance squared: d^2 = y^T * S^-1 * y
            d2 = float((y.T @ self.S_inv @ y)[0, 0])
            cand['mahalanobis_d2'] = d2
            r_diff = abs(cand.get('radius', target_radius) - target_radius) / max(target_radius, 1.0)
            app_sim = cand.get('appearance_similarity', 1.0)

            # Appearance Gate: Reject obvious color impostors if appearance signature is established
            # During fast motion / throws, motion blur can lower similarity; only reject single candidate if totally disjoint
            min_app_sim = 0.08 if len(candidates) == 1 else 0.25
            if app_sim < min_app_sim and cand.get('has_appearance', False):
                cand['rejection_reason'] = f"APPEARANCE_MISMATCH (sim={app_sim:.2f} < {min_app_sim:.2f})"
                clutter_candidates.append(cand)
                continue

            app_penalty = (1.0 - app_sim) * 18.0

            if d2 <= self.gate_threshold:
                cost = d2 + (r_diff * 6.0) + app_penalty
                nominal_matches.append((cost, cand))
            elif d2 <= self.maneuver_gate_threshold or len(candidates) == 1:
                cost = d2 + (r_diff * 20.0) + app_penalty
                maneuver_matches.append((cost, cand))
            else:
                cand['rejection_reason'] = f"OUT_OF_GATE (d2={d2:.1f} > {self.maneuver_gate_threshold:.1f})"
                clutter_candidates.append(cand)

        # Decision Logic:
        if nominal_matches:
            # We have a valid nominal track! Any maneuver-gate candidates were actually clutter
            nominal_matches.sort(key=lambda item: item[0])
            best_match = nominal_matches[0][1]
            best_match['is_maneuvering'] = False

            # Secondary nominal matches and any maneuver candidates are clutter
            for _, other in nominal_matches[1:]:
                other['rejection_reason'] = "SECONDARY_VALIDATED"
                clutter_candidates.append(other)
            for _, m_cand in maneuver_matches:
                m_cand['rejection_reason'] = f"OUT_OF_GATE (d2={m_cand.get('mahalanobis_d2', 0):.1f} > {self.gate_threshold:.1f})"
                clutter_candidates.append(m_cand)

            return best_match, clutter_candidates

        elif maneuver_matches:
            # High-G maneuver, sudden throw, or gravitational drop envelope
            maneuver_matches.sort(key=lambda item: item[0])
            best_match = maneuver_matches[0][1]
            best_match['is_maneuvering'] = True

            for _, other in maneuver_matches[1:]:
                other['rejection_reason'] = "SECONDARY_MANEUVER"
                clutter_candidates.append(other)

            return best_match, clutter_candidates

        elif len(candidates) == 1:
            # Single unambiguous candidate in camera field: fast throw / drop capture
            single_cand = candidates[0]
            single_cand['is_maneuvering'] = True
            return single_cand, []

        else:
            # All candidates outside both nominal and maneuver gates
            return None, clutter_candidates

    def update(self, z_x: float, z_y: float, is_maneuvering: bool = False) -> tuple[float, float]:
        """
        Corrects predicted state using the validated measurement (z_x, z_y).
        If is_maneuvering is True, adaptively inflates position and velocity covariance so the filter
        instantly snaps to the new high-speed trajectory instead of lagging or dropping lock.
        Returns filtered (pos_x, pos_y).
        """
        z = np.array([[z_x], [z_y]], dtype=np.float64)

        if not self.initialized:
            self.x[0, 0] = z_x
            self.x[1, 0] = z_y
            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0
            self.P = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float64)
            self.initialized = True
            self.frames_since_update = 0
            return z_x, z_y

        if is_maneuvering:
            # High-G acceleration / throw / drop: adaptively inflate covariance for instantaneous catch-up
            self.P[0, 0] += 60.0
            self.P[1, 1] += 60.0
            self.P[2, 2] += 800.0
            self.P[3, 3] += 800.0
            self.S = self.H @ self.P @ self.H.T + self.R
            try:
                self.S_inv = np.linalg.inv(self.S)
            except np.linalg.LinAlgError:
                self.S_inv = np.linalg.pinv(self.S)

        # Innovation: y = z - H * x_pred
        y = z - (self.H @ self.x)

        # Optimal Kalman Gain: K = P * H^T * S_inv
        K = self.P @ self.H.T @ self.S_inv

        # Updated state estimate: x = x_pred + K * y
        self.x = self.x + (K @ y)

        # Updated error covariance: P = (I - K * H) * P
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
# STANDALONE VALIDATION BENCHMARK
# ==========================================
if __name__ == "__main__":
    print("[KALMAN TEST] Testing Mahalanobis Gating & Clutter Rejection...")
    print("-" * 65)

    kf = KalmanFilter2D(process_noise_accel=65.0, measurement_noise=0.25, gate_threshold=16.0)
    kf.predict(1/60.0)
    kf.update(90.0, 90.0)  # Locked on target at (90, 90)

    # Project to next frame
    pred_x, pred_y = kf.predict(1/60.0)
    print(f"Predicted target position: ({pred_x:.2f}, {pred_y:.2f})")

    # Present 3 candidate detections:
    # 1. True target (near predicted position with slight noise)
    # 2. Nearby decoy / clutter (15 degrees away)
    # 3. Distant decoy / clutter (40 degrees away)
    candidates = [
        {'id': 'True Target', 'pos_x': pred_x + 0.15, 'pos_y': pred_y - 0.10, 'radius': 22.0},
        {'id': 'Decoy 1 (Nearby)', 'pos_x': pred_x + 8.5,  'pos_y': pred_y + 6.0,  'radius': 18.0},
        {'id': 'Decoy 2 (Distant)', 'pos_x': pred_x - 25.0, 'pos_y': pred_y + 20.0, 'radius': 24.0},
    ]

    best, clutter = kf.associate_and_gate(candidates, target_radius=22.0)

    print(f"\n[GATING RESULT] Associated Track: {best['id']} (d2={best.get('mahalanobis_d2', 0):.2f})")
    print("[REJECTED CLUTTER]:")
    for c in clutter:
        print(f"  - {c['id']}: {c.get('rejection_reason')} (d2={c.get('mahalanobis_d2', 0):.2f})")

    assert best['id'] == 'True Target', "Data Association failed!"
    print("\n[TEST PASSED] True target associated and all decoys successfully gated out!")
