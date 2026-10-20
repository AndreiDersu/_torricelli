"""End-to-end integration test: Video -> Tracking -> Calibration -> Kinematics."""

import math
from pathlib import Path

import numpy as np

from light_tracker.core.coordinate import CoordinateSystem
from light_tracker.core.kinematics import KinematicsEngine
from light_tracker.core.models import Point2D, Track, TrackPoint
from light_tracker.core.video import VideoReader
from light_tracker.tracking.template_tracker import TemplateTracker


def test_end_to_end_tracking_pipeline(
    synthetic_projectile_video: tuple[Path, list[Point2D]],
) -> None:
    video_path, ground_truth = synthetic_projectile_video

    with VideoReader(video_path) as reader:
        # Scale: 100 pixels = 1.0 meter
        coord_sys = CoordinateSystem(
            origin=Point2D(0.0, 240.0),  # Bottom-left origin
            angle_deg=0.0,
            scale_px_per_unit=100.0,
        )

        tracker = TemplateTracker(
            template_size=(20, 20),
            search_size=(50, 50),
            min_score=0.6,
        )

        # Initialize with ground truth on first frame
        f0 = reader.get_frame(0)
        assert f0 is not None
        p0 = ground_truth[0]
        tracker.initialize(f0, p0)

        track = Track(id="track-0", name="Particula Sintetica")
        track.add_point(
            TrackPoint(
                frame_idx=0,
                t=0.0,
                pixel_pos=p0,
                world_pos=coord_sys.pixel_to_world(p0),
                score=1.0,
            )
        )

        curr_pos = p0
        for i in range(1, reader.total_frames):
            frame = reader.get_frame(i)
            assert frame is not None

            # Lookahead velocity vector
            vel_est = None
            pts = track.get_sorted_points()
            if len(pts) >= 2:
                vel_est = Point2D(
                    pts[-1].pixel_pos.x - pts[-2].pixel_pos.x,
                    pts[-1].pixel_pos.y - pts[-2].pixel_pos.y,
                )

            detected, score = tracker.track_frame(
                frame, prev_pos=curr_pos, velocity_estimate=vel_est
            )
            assert detected is not None, f"Failed tracking at frame {i}"
            assert score > 0.8, f"Score too low at frame {i}: {score}"

            # Check precision against ground truth
            gt = ground_truth[i]
            error_px = detected.distance_to(gt)
            assert error_px < 0.8, f"Tracking error {error_px} px exceeds tolerance at frame {i}"

            curr_pos = detected
            world_pos = coord_sys.pixel_to_world(detected)
            track.add_point(
                TrackPoint(
                    frame_idx=i,
                    t=i / reader.fps,
                    pixel_pos=detected,
                    world_pos=world_pos,
                    score=score,
                )
            )

        # Compute Kinematics
        pts = track.get_sorted_points()
        assert len(pts) == reader.total_frames

        computed = KinematicsEngine.compute(pts)

        # Ground truth motion in conftest.py was:
        # px(t) = 40 + 100 * t  -> vx_pixel = 100 px/s -> vx_world = 1.0 m/s
        # py(t) = 180 - 160 * t + 120 * t^2
        # In physical world coordinates (where world Y is inverted relative to screen):
        # y_world = (240 - py) / 100 = 0.60 + 1.60 * t - 1.20 * t^2
        # vy_world = 1.60 - 2.40 * t
        # ay_world = -2.40 m/s^2

        vx_vals = [p.vx for p in computed[2:-2]]
        ay_vals = [p.ay for p in computed[2:-2]]

        # Horizontal velocity should be ~ 1.0 m/s
        assert math.isclose(float(np.mean(vx_vals)), 1.0, abs_tol=0.1)

        # Vertical acceleration should be ~ -2.40 m/s^2
        assert math.isclose(float(np.mean(ay_vals)), -2.40, abs_tol=0.2)
