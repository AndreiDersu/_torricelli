"""Manual click-and-advance tracking manager."""

from __future__ import annotations

from light_tracker.core.coordinate import CoordinateSystem
from light_tracker.core.models import Point2D, Track, TrackPoint


class ManualTracker:
    """Handles manual clicking to mark points and step through frames."""

    def __init__(self, step_size: int = 1) -> None:
        self.step_size = max(1, step_size)

    def mark_point(
        self,
        track: Track,
        frame_idx: int,
        fps: float,
        pixel_pos: Point2D,
        coord_sys: CoordinateSystem,
        is_keyframe: bool = False,
    ) -> TrackPoint:
        """Records a point in the track at the specified frame."""
        t = frame_idx / fps if fps > 0 else 0.0
        world_pos = coord_sys.pixel_to_world(pixel_pos)

        point = TrackPoint(
            frame_idx=frame_idx,
            t=t,
            pixel_pos=pixel_pos,
            world_pos=world_pos,
            score=1.0,
            is_valid=True,
            is_keyframe=is_keyframe,
        )
        track.add_point(point)
        return point
