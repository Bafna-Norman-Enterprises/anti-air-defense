import cv2
import numpy as np
from typing import Optional, Tuple


class AppearanceModel:
    """
    Electro-Optical Appearance Fingerprinting & Scale Estimation Engine.
    
    Maintains an adaptive 2D Hue-Saturation color distribution signature and geometric
    aspect-ratio/scale model of the physical target using Exponential Moving Average (EMA).
    
    Protects against:
      - Real-world false positives (e.g. pink shirts, orange reflections, red objects)
      - Lighting shifts and soft shadows
      - Distance/scale changes as the physical object approaches or recedes
    """

    def __init__(
        self,
        h_bins: int = 16,
        s_bins: int = 16,
        learning_rate: float = 0.04,
        similarity_threshold: float = 0.35,
    ):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.learning_rate = learning_rate
        self.similarity_threshold = similarity_threshold

        # Target Reference Signature (2D Normalized Hue-Saturation Histogram)
        self.reference_hist: Optional[np.ndarray] = None
        
        # Adaptive geometric state
        self.target_radius_ema: float = 22.0
        self.target_aspect_ema: float = 1.0  # width / height
        
        self.is_initialized: bool = False
        self.lock_count: int = 0

    def reset(self):
        """Clears reference signature and resets model."""
        self.reference_hist = None
        self.is_initialized = False
        self.lock_count = 0
        self.target_radius_ema = 22.0
        self.target_aspect_ema = 1.0

    def extract_signature(
        self,
        frame_bgr: np.ndarray,
        screen_x: int,
        screen_y: int,
        radius: float,
    ) -> Optional[np.ndarray]:
        """
        Extracts a normalized 2D Hue-Saturation histogram from a circular/elliptical ROI.
        """
        h, w = frame_bgr.shape[:2]
        r = max(int(round(radius)), 4)
        
        # Bounding box clamp
        x1 = max(0, screen_x - r)
        y1 = max(0, screen_y - r)
        x2 = min(w, screen_x + r)
        y2 = min(h, screen_y + r)

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return None

        roi_bgr = frame_bgr[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Circular mask to exclude background corners of the bounding box
        roi_h, roi_w = roi_bgr.shape[:2]
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.circle(mask, (roi_w // 2, roi_h // 2), min(roi_w, roi_h) // 2, 255, -1)

        # Compute 2D Hue-Sat Histogram: Hue [0..180], Sat [0..256]
        hist = cv2.calcHist(
            [roi_hsv],
            channels=[0, 1],
            mask=mask,
            histSize=[self.h_bins, self.s_bins],
            ranges=[0, 180, 0, 256],
        )

        # Normalize to unit probability distribution
        hist = cv2.normalize(hist, hist, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        return hist

    def compute_similarity(self, candidate_hist: Optional[np.ndarray]) -> float:
        """
        Calculates the appearance similarity score using the Bhattacharyya distance.
        Returns a similarity metric in [0.0, 1.0], where 1.0 is an identical match.
        """
        if self.reference_hist is None or candidate_hist is None:
            return 1.0  # Neutral prior if uninitialized

        # Bhattacharyya distance: 0.0 = identical, 1.0 = completely disjoint
        d_bhattacharyya = cv2.compareHist(
            self.reference_hist, candidate_hist, cv2.HISTCMP_BHATTACHARYYA
        )
        # Convert distance to similarity score
        similarity = float(np.clip(1.0 - d_bhattacharyya, 0.0, 1.0))
        return similarity

    def update(
        self,
        frame_bgr: np.ndarray,
        screen_x: int,
        screen_y: int,
        radius: float,
        aspect_ratio: float = 1.0,
    ):
        """
        Online adaptive learning: updates reference appearance model with new locked frame.
        """
        new_hist = self.extract_signature(frame_bgr, screen_x, screen_y, radius)
        if new_hist is None:
            return

        if not self.is_initialized:
            self.reference_hist = new_hist.copy()
            self.target_radius_ema = float(radius)
            self.target_aspect_ema = float(aspect_ratio)
            self.is_initialized = True
            self.lock_count = 1
        else:
            # Exponential Moving Average update: adapts smoothly to room lighting shifts
            alpha = self.learning_rate
            self.reference_hist = (1.0 - alpha) * self.reference_hist + alpha * new_hist
            self.reference_hist = cv2.normalize(
                self.reference_hist, self.reference_hist, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX
            )

            # Smooth geometric tracking
            self.target_radius_ema = (1.0 - alpha) * self.target_radius_ema + alpha * radius
            self.target_aspect_ema = (1.0 - alpha) * self.target_aspect_ema + alpha * aspect_ratio
            self.lock_count += 1

    @property
    def radius(self) -> float:
        return self.target_radius_ema

    @property
    def aspect_ratio(self) -> float:
        return self.target_aspect_ema
