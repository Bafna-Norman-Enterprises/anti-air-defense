import math
import numpy as np
from typing import Optional
from kalman import KalmanFilter2D
from appearance import AppearanceModel


class Track:
    """
    Represents a single tracked airborne or room object in the Mark 1 Track-While-Scan table.
    Maintains its own 6-state Constant Acceleration Kalman Filter, appearance signature,
    Track Quality Index (TQI), and kinematic trajectory classification.
    """

    def __init__(
        self,
        track_id: int,
        pos_x: float,
        pos_y: float,
        screen_x: int,
        screen_y: int,
        radius: float,
        frame_bgr: Optional[np.ndarray] = None,
        acq_score: float = 0.0,
        confirmed: bool = False,
    ):
        self.track_id = track_id
        self.label = f"TRK-{track_id:02d}"

        self.kf = KalmanFilter2D(
            process_noise_accel=650.0,
            measurement_noise=0.20,
            max_coast_frames=50,
            gate_threshold=25.0,
            maneuver_gate_threshold=900.0,
        )
        self.kf.update(pos_x, pos_y)

        self.appearance = AppearanceModel()
        if frame_bgr is not None:
            self.appearance.update(frame_bgr, screen_x, screen_y, radius)

        self.status = "CONFIRMED" if confirmed else "TENTATIVE"
        self.hits = 3 if confirmed else 1
        self.age = 1
        self.misses = 0

        self.last_screen_x = screen_x
        self.last_screen_y = screen_y
        self.last_radius = float(radius)
        self.last_app_sim = 1.0
        self.acq_score = acq_score

        # Track Quality Index (0..100%) and kinematic classification
        self.tqi = 75.0 if confirmed else 35.0
        self.flight_class = "HOVER"

    def classify_kinematics(self):
        """Classifies target trajectory dynamics from Kalman velocity & acceleration."""
        vx, vy = self.kf.velocity
        ax, ay = self.kf.acceleration
        speed = math.hypot(vx, vy)
        accel = math.hypot(ax, ay)

        if speed < 3.0 and accel < 25.0:
            self.flight_class = "HOVER"
        elif ay < -220.0 and speed > 10.0:
            self.flight_class = "BALLISTIC"
        elif accel > 180.0:
            self.flight_class = "HIGH-G"
        else:
            self.flight_class = "CRUISE"

    def update_quality(self, matched: bool, d2: float = 0.0):
        """Updates Track Quality Index (0..100%) based on measurement residual and continuity."""
        if matched:
            residual_score = max(0.0, 100.0 - d2 * 1.5)
            app_score = self.last_app_sim * 100.0
            sample_q = 0.65 * residual_score + 0.35 * app_score
            self.tqi = float(np.clip(0.75 * self.tqi + 0.25 * sample_q, 5.0, 99.0))
        else:
            self.tqi = float(np.clip(self.tqi * 0.92, 0.0, 99.0))


class TrackManager:
    """
    Aerospace Multi-Target Tracking (MTT) & Track-While-Scan (TWS) Engine.

    Maintains a live track table of up to `max_tracks` simultaneous objects using M/N
    track confirmation logic (filters out 1-frame specular glare before promotion),
    global nearest-neighbor data association, and operator-selectable Primary Track designation.
    """

    def __init__(
        self,
        max_tracks: int = 6,
        confirm_hits: int = 3,
        max_tentative_misses: int = 3,
        max_confirmed_misses: int = 45,
    ):
        self.max_tracks = max_tracks
        self.confirm_hits = confirm_hits
        self.max_tentative_misses = max_tentative_misses
        self.max_confirmed_misses = max_confirmed_misses

        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1
        self.primary_id: Optional[int] = None

    def reset(self):
        """Clears the entire track table and resets track ID counter."""
        self.tracks.clear()
        self.next_track_id = 1
        self.primary_id = None

    def _spawn_track(
        self,
        cand: dict,
        frame_bgr: Optional[np.ndarray],
        confirmed: bool = False,
    ) -> Optional[Track]:
        if len(self.tracks) >= self.max_tracks:
            return None
        tid = self.next_track_id
        self.next_track_id += 1
        trk = Track(
            track_id=tid,
            pos_x=cand["pos_x"],
            pos_y=cand["pos_y"],
            screen_x=cand["screen_x"],
            screen_y=cand["screen_y"],
            radius=cand["radius"],
            frame_bgr=frame_bgr,
            acq_score=cand.get("acq_score", 0.0),
            confirmed=confirmed,
        )
        self.tracks[tid] = trk
        return trk

    def designate_at(
        self,
        pos_x: float,
        pos_y: float,
        screen_x: int,
        screen_y: int,
        frame_bgr: np.ndarray,
        radius: float = 18.0,
    ) -> Track:
        """
        Operator Click-to-Designate: promotes an existing nearby track or spawns a new
        CONFIRMED Primary Track at the clicked coordinates.
        """
        best_tid = None
        best_dist = 45.0
        for tid, trk in self.tracks.items():
            d = math.hypot(trk.last_screen_x - screen_x, trk.last_screen_y - screen_y)
            if d < best_dist:
                best_dist = d
                best_tid = tid

        if best_tid is not None:
            trk = self.tracks[best_tid]
            trk.kf.reset()
            trk.kf.update(pos_x, pos_y)
            trk.appearance.reset()
            trk.appearance.update(frame_bgr, screen_x, screen_y, radius)
            trk.status = "CONFIRMED"
            trk.hits = max(trk.hits, self.confirm_hits)
            trk.misses = 0
            trk.tqi = 90.0
            self.primary_id = best_tid
            return trk

        # Reset table if full so operator click always succeeds
        if len(self.tracks) >= self.max_tracks:
            victim = min(self.tracks.keys(), key=lambda k: self.tracks[k].tqi)
            del self.tracks[victim]

        trk = self._spawn_track(
            {
                "pos_x": pos_x,
                "pos_y": pos_y,
                "screen_x": screen_x,
                "screen_y": screen_y,
                "radius": radius,
                "acq_score": 5.0,
            },
            frame_bgr=frame_bgr,
            confirmed=True,
        )
        self.primary_id = trk.track_id
        return trk

    def cycle_primary(self) -> Optional[Track]:
        """Cycles the Primary Target of Interest to the next active track in the table."""
        if not self.tracks:
            return None
        confirmed_ids = sorted(tid for tid, t in self.tracks.items() if t.status == "CONFIRMED")
        pool = confirmed_ids if confirmed_ids else sorted(self.tracks.keys())
        if self.primary_id in pool and len(pool) > 1:
            idx = pool.index(self.primary_id)
            self.primary_id = pool[(idx + 1) % len(pool)]
        else:
            self.primary_id = pool[0]
        return self.tracks[self.primary_id]

    def step(
        self,
        candidates: list[dict],
        dt: float,
        frame_bgr: Optional[np.ndarray],
        current_pan: float,
        current_tilt: float,
        frame_cx: int,
        frame_cy: int,
        px_per_deg_x: float,
        px_per_deg_y: float,
        is_occluded: bool = False,
    ):
        """
        Executes one Track-While-Scan (TWS) cycle across all active tracks:
        Predict -> Global Association -> Update / Coast / Prune -> Spawn -> Primary Selection.
        """
        # 1. Predict all active tracks forward
        for trk in self.tracks.values():
            trk.kf.predict(dt)
            trk.age += 1

        matched_tracks = set()
        matched_cands = set()

        # 2. Global Track-to-Candidate Cost Association
        if candidates and self.tracks and not is_occluded:
            pairs = []
            for tid, trk in self.tracks.items():
                pred_pos = np.array([[trk.kf.x[0, 0]], [trk.kf.x[1, 0]]], dtype=np.float64)
                for c_idx, cand in enumerate(candidates):
                    z = np.array([[cand["pos_x"]], [cand["pos_y"]]], dtype=np.float64)
                    y = z - pred_pos
                    d2 = float((y.T @ trk.kf.S_inv @ y)[0, 0])

                    # Compute appearance similarity against this specific track's fingerprint
                    app_sim = 1.0
                    if frame_bgr is not None and trk.appearance.is_initialized:
                        cand_hist = trk.appearance.extract_signature(
                            frame_bgr, cand["screen_x"], cand["screen_y"], cand["radius"]
                        )
                        app_sim = trk.appearance.compute_similarity(cand_hist)

                    # Allow unambiguous 1-to-1 high-G throw capture
                    single_pair = (len(self.tracks) == 1 and len(candidates) == 1)
                    if d2 <= trk.kf.maneuver_gate_threshold or single_pair:
                        if app_sim < 0.15 and trk.appearance.is_initialized and not single_pair:
                            continue
                        r_diff = abs(cand["radius"] - trk.last_radius) / max(trk.last_radius, 1.0)
                        cost = d2 + (1.0 - app_sim) * 18.0 + r_diff * 8.0
                        # Slight preference for confirmed tracks over tentative clutter
                        if trk.status != "CONFIRMED":
                            cost += 12.0
                        pairs.append((cost, d2, app_sim, tid, c_idx))

            pairs.sort(key=lambda item: item[0])
            for cost, d2, app_sim, tid, c_idx in pairs:
                if tid in matched_tracks or c_idx in matched_cands:
                    continue
                trk = self.tracks[tid]
                cand = candidates[c_idx]
                is_man = d2 > trk.kf.gate_threshold
                trk.kf.update(cand["pos_x"], cand["pos_y"], is_maneuvering=is_man)

                trk.last_screen_x = cand["screen_x"]
                trk.last_screen_y = cand["screen_y"]
                trk.last_radius = float(cand["radius"])
                trk.last_app_sim = app_sim
                if "acq_score" in cand:
                    trk.acq_score = 0.7 * trk.acq_score + 0.3 * cand["acq_score"]

                if frame_bgr is not None:
                    trk.appearance.update(frame_bgr, cand["screen_x"], cand["screen_y"], cand["radius"])

                trk.hits += 1
                trk.misses = 0
                if trk.status == "TENTATIVE" and trk.hits >= self.confirm_hits:
                    trk.status = "CONFIRMED"
                elif trk.status == "COAST":
                    trk.status = "CONFIRMED"

                trk.classify_kinematics()
                trk.update_quality(matched=True, d2=d2)
                matched_tracks.add(tid)
                matched_cands.add(c_idx)

        # 3. Coast unmatched tracks & prune expired tracks
        dead_ids = []
        for tid, trk in self.tracks.items():
            if tid not in matched_tracks:
                trk.misses += 1
                if trk.status == "CONFIRMED":
                    trk.status = "COAST"
                trk.update_quality(matched=False)
                trk.classify_kinematics()

                # Update projected screen position while coasting
                est_pan, est_tilt = trk.kf.position
                trk.last_screen_x = int(round(frame_cx + (est_pan - current_pan) * px_per_deg_x))
                trk.last_screen_y = int(round(frame_cy - (est_tilt - current_tilt) * px_per_deg_y))

                max_m = self.max_tentative_misses if trk.status == "TENTATIVE" else self.max_confirmed_misses
                if trk.misses > max_m:
                    dead_ids.append(tid)

        for tid in dead_ids:
            del self.tracks[tid]
            if self.primary_id == tid:
                self.primary_id = None

        # 4. Spawn new TENTATIVE tracks for unmatched detections
        if not is_occluded:
            for c_idx, cand in enumerate(candidates):
                if c_idx not in matched_cands:
                    self._spawn_track(cand, frame_bgr=frame_bgr, confirmed=False)

        # 5. Select / maintain Primary Target of Interest (POI)
        if self.primary_id not in self.tracks:
            self.primary_id = None
            confirmed = [t for t in self.tracks.values() if t.status in ("CONFIRMED", "COAST")]
            if confirmed:
                best_t = max(confirmed, key=lambda t: (t.acq_score, t.tqi))
                self.primary_id = best_t.track_id
            elif self.tracks:
                best_t = max(self.tracks.values(), key=lambda t: (t.hits, t.acq_score))
                self.primary_id = best_t.track_id

    def get_primary(self) -> Optional[Track]:
        """Returns the currently designated Primary Track, or None."""
        if self.primary_id is not None:
            return self.tracks.get(self.primary_id)
        return None

    def get_secondary_tracks(self) -> list[Track]:
        """Returns all other CONFIRMED/COAST tracks in the Track-While-Scan table."""
        return [
            t for tid, t in self.tracks.items()
            if tid != self.primary_id and t.status in ("CONFIRMED", "COAST")
        ]
