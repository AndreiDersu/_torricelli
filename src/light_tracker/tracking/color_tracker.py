"""Color and HSV blob centroid tracker for physics lab markers."""

from __future__ import annotations

import cv2
import numpy as np

from light_tracker.core.models import Point2D


class ColorTracker:
    """Tracks colored objects (balls, lasers, colored tape markers) using HSV segmentation."""

    def __init__(
        self,
        lower_hsv: np.ndarray | None = None,
        upper_hsv: np.ndarray | None = None,
        search_radius: int = 80,
        min_area: float = 8.0,
    ) -> None:
        """Initializes the ColorTracker.

        Args:
            lower_hsv: Lower HSV bounds (array of [H, S, V] with H in 0-179, S,V in 0-255).
            upper_hsv: Upper HSV bounds.
            search_radius: Radius in pixels around previous position to search for object.
            min_area: Minimum contour pixel area to accept.
        """
        self.lower_hsv = (
            lower_hsv if lower_hsv is not None else np.array([0, 100, 100], dtype=np.uint8)
        )
        self.upper_hsv = (
            upper_hsv if upper_hsv is not None else np.array([10, 255, 255], dtype=np.uint8)
        )
        self.search_radius = search_radius
        self.min_area = min_area
        self.last_pos: Point2D | None = None
        self.target_area: float = 50.0

    def initialize(self, frame: np.ndarray, target_pos: Point2D) -> None:
        """Auto-calibrates HSV threshold by sampling color around target_pos."""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv_frame.shape[:2]

        r = 6
        x1 = max(0, round(target_pos.x - r))
        y1 = max(0, round(target_pos.y - r))
        x2 = min(w, round(target_pos.x + r))
        y2 = min(h, round(target_pos.y + r))

        patch = hsv_frame[y1:y2, x1:x2]
        if patch.size > 0:
            median_hsv = np.median(patch.reshape(-1, 3), axis=0)
            # Define sensible tolerance around median
            h_tol, s_tol, v_tol = 15, 60, 60
            self.lower_hsv = np.array(
                [
                    max(0, int(median_hsv[0] - h_tol)),
                    max(40, int(median_hsv[1] - s_tol)),
                    max(40, int(median_hsv[2] - v_tol)),
                ],
                dtype=np.uint8,
            )
            self.upper_hsv = np.array(
                [
                    min(179, int(median_hsv[0] + h_tol)),
                    255,
                    255,
                ],
                dtype=np.uint8,
            )

        self.last_pos = target_pos

    def track_frame(
        self,
        frame: np.ndarray,
        prev_pos: Point2D | None = None,
        velocity_estimate: Point2D | None = None,
    ) -> tuple[Point2D | None, float]:
        """Detects color blob centroid in frame."""
        center = prev_pos or self.last_pos
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv_frame.shape[:2]

        # Region of interest crop
        if center is not None and self.search_radius > 0:
            pred_x = center.x + (velocity_estimate.x if velocity_estimate else 0.0)
            pred_y = center.y + (velocity_estimate.y if velocity_estimate else 0.0)
            rx1 = max(0, round(pred_x - self.search_radius))
            ry1 = max(0, round(pred_y - self.search_radius))
            rx2 = min(w, round(pred_x + self.search_radius))
            ry2 = min(h, round(pred_y + self.search_radius))
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w, h

        roi = hsv_frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None, 0.0

        mask = cv2.inRange(roi, self.lower_hsv, self.upper_hsv)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        best_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_cnt)
        if area < self.min_area:
            return None, 0.0

        moments = cv2.moments(best_cnt)
        if moments["m00"] <= 0:
            return None, 0.0

        cx = (moments["m10"] / moments["m00"]) + rx1
        cy = (moments["m01"] / moments["m00"]) + ry1

        detected_pos = Point2D(x=float(cx), y=float(cy))
        self.last_pos = detected_pos
        confidence = min(1.0, area / max(self.target_area, 1.0))

        return detected_pos, float(confidence)
