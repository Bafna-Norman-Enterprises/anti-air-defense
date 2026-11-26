import cv2
import math
import numpy as np
from typing import Optional


class HybridFallbackTracker:
    """
    Cascaded Hybrid Texture/Motion Fallback Tracker:
      - Tier A: Sparse Lucas-Kanade Optical Flow (fast frame-to-frame feature tracking)
      - Tier B: Localized Normalized Cross-Correlation (NCC) Template Matching

    Maintains target lock when HSV color segmentation momentarily drops out due to
    sudden shadows, harsh backlighting, or exposure shifts.
    """

    def __init__(self, ncc_threshold: float = 0.72, search_margin: int = 55):
        self.ncc_threshold = ncc_threshold
        self.search_margin = search_margin

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.template_gray: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.p0: Optional[np.ndarray] = None
        self.last_cx: float = 0.0
        self.last_cy: float = 0.0
        self.is_ready: bool = False

    def reset(self):
        """Clears stored template and optical-flow keypoints."""
        self.template_gray = None
        self.prev_gray = None
        self.p0 = None
        self.last_cx = 0.0
        self.last_cy = 0.0
        self.is_ready = False

    def _seed_keypoints(self, frame_gray: np.ndarray, cx: int, cy: int, half_box: int) -> Optional[np.ndarray]:
        """Extracts good Shi-Tomasi corners inside the target ROI for Lucas-Kanade flow."""
        h, w = frame_gray.shape[:2]
        x1 = max(0, cx - half_box)
        y1 = max(0, cy - half_box)
        x2 = min(w, cx + half_box)
        y2 = min(h, cy + half_box)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        pts = cv2.goodFeaturesToTrack(
            frame_gray[y1:y2, x1:x2],
            mask=None,
            maxCorners=25,
            qualityLevel=0.15,
            minDistance=4,
            blockSize=5,
        )
        if pts is not None and len(pts) > 0:
            pts[:, 0, 0] += x1
            pts[:, 0, 1] += y1
        return pts

    def update_reference(self, frame_gray: np.ndarray, cx: int, cy: int, radius: float):
        """
        Captures a fresh grayscale template patch & optical-flow keypoints while HSV lock is strong.
        """
        h, w = frame_gray.shape[:2]
        half_box = int(np.clip(round(radius * 0.9), 10, 28))
        x1 = cx - half_box
        y1 = cy - half_box
        x2 = cx + half_box
        y2 = cy + half_box

        # Only capture template when fully inside frame boundaries
        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h and (x2 - x1) >= 16:
            self.template_gray = frame_gray[y1:y2, x1:x2].copy()
            self.p0 = self._seed_keypoints(frame_gray, cx, cy, half_box)
            self.prev_gray = frame_gray.copy()
            self.last_cx = float(cx)
            self.last_cy = float(cy)
            self.is_ready = True

    def track_fallback(
        self, frame_gray: np.ndarray, pred_cx: int, pred_cy: int, radius: float
    ) -> Optional[tuple[int, int, float, str]]:
        """
        Attempts to track the target by texture/structure when color segmentation misses.
        Returns (new_cx, new_cy, confidence, method_tag) or None if fallback fails.
        """
        if not self.is_ready or self.prev_gray is None:
            return None

        h_f, w_f = frame_gray.shape[:2]

        # Tier A: Sparse Lucas-Kanade Optical Flow
        if self.p0 is not None and len(self.p0) >= 5:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.p0, None, **self.lk_params
            )
            if p1 is not None and st is not None:
                valid = (st.flatten() == 1)
                if err is not None:
                    valid = valid & (err.flatten() < 18.0)
                good_new = p1[valid].reshape(-1, 2)
                good_old = self.p0[valid].reshape(-1, 2)

                if len(good_new) >= 4:
                    dx = float(np.median(good_new[:, 0] - good_old[:, 0]))
                    dy = float(np.median(good_new[:, 1] - good_old[:, 1]))
                    if math.hypot(dx, dy) <= 95.0:
                        new_cx = float(np.clip(self.last_cx + dx, 0, w_f - 1))
                        new_cy = float(np.clip(self.last_cy + dy, 0, h_f - 1))
                        self.last_cx = new_cx
                        self.last_cy = new_cy
                        self.p0 = good_new.reshape(-1, 1, 2)
                        self.prev_gray = frame_gray.copy()
                        conf = min(len(good_new) / 15.0, 1.0)
                        return int(round(new_cx)), int(round(new_cy)), conf, "FLOW"

        # Tier B: Localized NCC Template Matching around Kalman predicted position
        if self.template_gray is not None:
            th, tw = self.template_gray.shape[:2]
            half_w, half_h = tw // 2, th // 2
            sx1 = max(0, pred_cx - half_w - self.search_margin)
            sy1 = max(0, pred_cy - half_h - self.search_margin)
            sx2 = min(w_f, pred_cx + half_w + self.search_margin)
            sy2 = min(h_f, pred_cy + half_h + self.search_margin)

            search_roi = frame_gray[sy1:sy2, sx1:sx2]
            if search_roi.shape[0] >= th and search_roi.shape[1] >= tw:
                res = cv2.matchTemplate(search_roi, self.template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= self.ncc_threshold:
                    new_cx = sx1 + max_loc[0] + half_w
                    new_cy = sy1 + max_loc[1] + half_h
                    self.last_cx = float(new_cx)
                    self.last_cy = float(new_cy)
                    half_box = max(half_w, 10)
                    self.p0 = self._seed_keypoints(frame_gray, new_cx, new_cy, half_box)
                    self.prev_gray = frame_gray.copy()
                    return int(new_cx), int(new_cy), float(max_val), "NCC"

        self.prev_gray = frame_gray.copy()
        return None
