"""Base interfaces and protocols for tracking algorithms."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from light_tracker.core.models import Point2D


class TrackerProtocol(Protocol):
    """Protocol implemented by all visual object tracking engines."""

    def initialize(self, frame: np.ndarray, target_pos: Point2D) -> None:
        """Initializes tracking target on a keyframe."""
        ...

    def track_frame(
        self,
        frame: np.ndarray,
        prev_pos: Point2D | None = None,
        velocity_estimate: Point2D | None = None,
    ) -> tuple[Point2D | None, float]:
        """Tracks the target in the given frame.

        Args:
            frame: Video image (RGB, BGR, or Grayscale).
            prev_pos: Last known position in pixels.
            velocity_estimate: Estimated velocity vector (dx, dy) per frame for lookahead.

        Returns:
            Tuple of (detected_position_or_none, confidence_score_between_0_and_1).
        """
        ...
