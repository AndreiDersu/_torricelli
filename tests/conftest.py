"""Pytest fixtures and synthetic physics video generator."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
import pytest

from light_tracker.core.models import Point2D


@pytest.fixture
def synthetic_projectile_video() -> Generator[tuple[Path, list[Point2D]], None, None]:
    """Generates a temporary synthetic video of a particle in projectile motion.

    Returns:
        Tuple of (video_file_path, ground_truth_pixel_positions).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="light_tracker_test_"))
    video_path = tmp_dir / "projectile.mp4"

    width, height = 320, 240
    fps = 30.0
    num_frames = 25

    # Parabolic trajectory parameters in screen coordinates
    # x(t) = 40 + 80 * t
    # y(t) = 180 - 150 * t + 100 * t^2  (t in [0, 0.8] s)
    dt = 1.0 / fps

    ground_truth: list[Point2D] = []
    frames: list[np.ndarray] = []

    radius = 6

    for i in range(num_frames):
        t = i * dt
        px = 40.0 + 100.0 * t
        py = 180.0 - 160.0 * t + 120.0 * (t**2)
        ground_truth.append(Point2D(px, py))

        # White background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 245

        # Draw dark circle with soft anti-aliased edge
        center_int = (round(px), round(py))
        cv2.circle(frame, center_int, radius, (30, 30, 30), -1, cv2.LINE_AA)

        frames.append(frame)

    # Write video with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()

    yield video_path, ground_truth

    # Cleanup
    if video_path.exists():
        video_path.unlink()
    if tmp_dir.exists():
        tmp_dir.rmdir()
