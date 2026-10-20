"""Unit tests for VideoReader."""

from pathlib import Path

from light_tracker.core.models import Point2D
from light_tracker.core.video import VideoReader


def test_video_reader_metadata_and_frames(
    synthetic_projectile_video: tuple[Path, list[Point2D]],
) -> None:
    video_path, ground_truth = synthetic_projectile_video

    with VideoReader(video_path) as reader:
        assert reader.width == 320
        assert reader.height == 240
        assert reader.fps == 30.0
        assert reader.total_frames == len(ground_truth)
        assert reader.duration > 0.0

        # Read first frame
        f0 = reader.get_frame(0)
        assert f0 is not None
        assert f0.shape == (240, 320, 3)

        # Read middle frame
        mid_idx = len(ground_truth) // 2
        f_mid = reader.get_frame(mid_idx)
        assert f_mid is not None
        assert f_mid.shape == (240, 320, 3)

        # Out of bounds frames return None
        assert reader.get_frame(-1) is None
        assert reader.get_frame(9999) is None
