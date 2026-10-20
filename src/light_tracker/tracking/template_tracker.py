"""AutoTracker implementation based on Normalized Cross-Correlation (NCC).

Faithfully replicates the OSP Tracker autotracker behavior with template matching,
search window constraints, velocity lookahead, and subpixel quadratic peak interpolation.
"""

from __future__ import annotations

import cv2
import numpy as np

from light_tracker.core.models import Point2D


class TemplateTracker:
    """Template matching tracker using Normalized Cross Correlation (NCC)."""

    def __init__(
        self,
        template_size: tuple[int, int] = (24, 24),
        search_size: tuple[int, int] = (64, 64),
        min_score: float = 0.60,
        evolve_rate: float = 0.0,
    ) -> None:
        """Initializes the TemplateTracker.

        Args:
            template_size: Width and height of template patch (pixels).
            search_size: Width and height of search area window (pixels).
            min_score: Minimum correlation threshold [0.0, 1.0] for match acceptance.
            evolve_rate: Learning rate for template adaptation (0.0 keeps keyframe template).
        """
        self.template_size = template_size
        self.search_size = search_size
        self.min_score = min_score
        self.evolve_rate = max(0.0, min(1.0, evolve_rate))

        self.template: np.ndarray | None = None
        self.last_pos: Point2D | None = None
        self.is_initialized: bool = False

    def initialize(self, frame: np.ndarray, target_pos: Point2D) -> None:
        """Extracts and stores the reference template around target_pos."""
        gray = self._to_gray(frame)
        h, w = gray.shape[:2]
        tw, th = self.template_size

        x1 = round(target_pos.x - tw / 2.0)
        y1 = round(target_pos.y - th / 2.0)
        x2 = x1 + tw
        y2 = y1 + th

        # Clamp safely within frame
        x1_c = max(0, min(w, x1))
        y1_c = max(0, min(h, y1))
        x2_c = max(0, min(w, x2))
        y2_c = max(0, min(h, y2))

        crop = gray[y1_c:y2_c, x1_c:x2_c]
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            raise ValueError("Target position is too close to image boundary to extract template.")

        # Ensure consistent size if boundary-clamped
        if crop.shape != (th, tw):
            crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)

        self.template = crop.astype(np.float32)
        self.last_pos = target_pos
        self.is_initialized = True

    def track_frame(
        self,
        frame: np.ndarray,
        prev_pos: Point2D | None = None,
        velocity_estimate: Point2D | None = None,
    ) -> tuple[Point2D | None, float]:
        """Tracks the template within the search window of the given frame.

        Args:
            frame: Video image.
            prev_pos: Optional previous position (defaults to self.last_pos).
            velocity_estimate: Optional velocity estimate (dx, dy) for lookahead center.

        Returns:
            Tuple of (detected_position, match_score). If match_score < min_score,
            returns (detected_position, match_score) so user can see candidate or decide to reject.
        """
        if not self.is_initialized or self.template is None:
            return None, 0.0

        center = prev_pos or self.last_pos
        if center is None:
            return None, 0.0

        # Apply lookahead prediction if velocity is provided
        pred_x = center.x + (velocity_estimate.x if velocity_estimate else 0.0)
        pred_y = center.y + (velocity_estimate.y if velocity_estimate else 0.0)

        gray = self._to_gray(frame)
        fh, fw = gray.shape[:2]
        sw, sh = self.search_size
        tw, th = self.template_size

        # Define search window centered at predicted location
        sx1 = max(0, round(pred_x - sw / 2.0))
        sy1 = max(0, round(pred_y - sh / 2.0))
        sx2 = min(fw, sx1 + sw)
        sy2 = min(fh, sy1 + sh)

        # Ensure search window is larger than template
        if (sy2 - sy1) < th or (sx2 - sx1) < tw:
            return None, 0.0

        search_roi = gray[sy1:sy2, sx1:sx2].astype(np.float32)

        # Run Normalized Cross-Correlation
        res = cv2.matchTemplate(search_roi, self.template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)

        score = float(max_val)
        if np.isnan(score):
            return None, 0.0

        peak_x, peak_y = max_loc
        res_h, res_w = res.shape

        # Subpixel parabolic interpolation
        sub_dx = 0.0
        sub_dy = 0.0

        if 0 < peak_x < res_w - 1:
            denom_x = 2.0 * res[peak_y, peak_x] - res[peak_y, peak_x - 1] - res[peak_y, peak_x + 1]
            if abs(denom_x) > 1e-5:
                sub_dx = (res[peak_y, peak_x + 1] - res[peak_y, peak_x - 1]) / (2.0 * denom_x)
                sub_dx = max(-0.5, min(0.5, sub_dx))

        if 0 < peak_y < res_h - 1:
            denom_y = 2.0 * res[peak_y, peak_x] - res[peak_y - 1, peak_x] - res[peak_y + 1, peak_x]
            if abs(denom_y) > 1e-5:
                sub_dy = (res[peak_y + 1, peak_x] - res[peak_y - 1, peak_x]) / (2.0 * denom_y)
                sub_dy = max(-0.5, min(0.5, sub_dy))

        # Position of template center in full frame
        detected_x = sx1 + (peak_x + sub_dx) + (tw / 2.0)
        detected_y = sy1 + (peak_y + sub_dy) + (th / 2.0)
        detected_pos = Point2D(x=detected_x, y=detected_y)

        # If match is acceptable and evolve_rate > 0, update template
        if score >= self.min_score and self.evolve_rate > 0.0:
            rx1 = max(0, min(fw - tw, round(detected_x - tw / 2.0)))
            ry1 = max(0, min(fh - th, round(detected_y - th / 2.0)))
            new_patch = gray[ry1 : ry1 + th, rx1 : rx1 + tw].astype(np.float32)
            if new_patch.shape == self.template.shape:
                self.template = (1.0 - self.evolve_rate) * self.template + (
                    self.evolve_rate * new_patch
                )

        if score >= self.min_score:
            self.last_pos = detected_pos

        return detected_pos, score

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Converts an image to single-channel uint8 grayscale."""
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image[:, :, 0]
