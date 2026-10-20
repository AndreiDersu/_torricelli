"""Data models and entities for LightTracker."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Point2D:
    """Represents a 2D spatial coordinate."""

    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        """Calculates Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_tuple(self) -> tuple[float, float]:
        """Returns coordinate as an (x, y) tuple."""
        return (self.x, self.y)

    def to_int_tuple(self) -> tuple[int, int]:
        """Returns coordinate as an integer (round(x), round(y)) tuple."""
        return (round(self.x), round(self.y))


@dataclass(slots=True, frozen=True)
class CalibrationStick:
    """A calibration stick defined by two endpoints and a physical length."""

    p1: Point2D
    p2: Point2D
    real_length: float = 1.0
    unit: str = "m"

    @property
    def pixel_length(self) -> float:
        """Returns distance in pixels between the two points."""
        return self.p1.distance_to(self.p2)

    @property
    def scale_px_per_unit(self) -> float:
        """Returns scale factor in pixels per real-world unit."""
        if self.real_length <= 0:
            raise ValueError("Real length must be positive.")
        return self.pixel_length / self.real_length


@dataclass(slots=True, frozen=True)
class CoordinateOrigin:
    """Defines the position of the physical coordinate origin and rotation angle."""

    origin: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))
    angle_deg: float = 0.0  # Angle of X-axis in degrees relative to horizontal (counter-clockwise)


@dataclass(slots=True)
class TrackPoint:
    """Point recorded at a specific video frame."""

    frame_idx: int
    t: float
    pixel_pos: Point2D
    world_pos: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))
    vx: float = 0.0
    vy: float = 0.0
    v: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    a: float = 0.0
    score: float = 1.0
    is_valid: bool = True
    is_keyframe: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Converts point to a dictionary representation."""
        return {
            "frame_idx": self.frame_idx,
            "t": self.t,
            "pixel_x": self.pixel_pos.x,
            "pixel_y": self.pixel_pos.y,
            "world_x": self.world_pos.x,
            "world_y": self.world_pos.y,
            "vx": self.vx,
            "vy": self.vy,
            "v": self.v,
            "ax": self.ax,
            "ay": self.ay,
            "a": self.a,
            "score": self.score,
            "is_valid": self.is_valid,
            "is_keyframe": self.is_keyframe,
        }


@dataclass(slots=True)
class Track:
    """Represents a tracked physical object (Point Mass)."""

    id: str
    name: str
    color: str = "#E63946"  # Professional energetic red default
    mass: float = 1.0  # kg
    points: dict[int, TrackPoint] = field(default_factory=dict)

    def add_point(self, point: TrackPoint) -> None:
        """Adds or updates a track point at its frame_idx."""
        self.points[point.frame_idx] = point

    def remove_point(self, frame_idx: int) -> TrackPoint | None:
        """Removes the point at frame_idx if present."""
        return self.points.pop(frame_idx, None)

    def get_sorted_points(self) -> list[TrackPoint]:
        """Returns all points sorted by frame index."""
        return [self.points[k] for k in sorted(self.points.keys())]

    def clear(self) -> None:
        """Clears all points in the track."""
        self.points.clear()


@dataclass(slots=True)
class ProjectData:
    """Complete project data structure for serialization."""

    video_path: str = ""
    fps: float = 30.0
    total_frames: int = 0
    start_frame: int = 0
    end_frame: int = 0
    step_size: int = 1
    calibration: CalibrationStick | None = None
    coordinate_origin: CoordinateOrigin = field(default_factory=CoordinateOrigin)
    tracks: list[Track] = field(default_factory=list)
