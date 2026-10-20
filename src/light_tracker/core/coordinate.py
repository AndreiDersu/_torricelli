"""Coordinate system transformations and real-world calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from light_tracker.core.models import CalibrationStick, CoordinateOrigin, Point2D


@dataclass(slots=True)
class CoordinateSystem:
    """Manages spatial calibration, origin location, and axis orientation.

    Converts coordinates between screen/pixel space (where Y points down)
    and laboratory Cartesian space (where Y points up, scaled to meters/units).
    """

    origin: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))
    angle_deg: float = 0.0
    scale_px_per_unit: float = 1.0  # Pixels per real unit (e.g. px/m)
    unit_name: str = "m"

    def set_calibration_stick(self, stick: CalibrationStick) -> None:
        """Sets scale factor and unit from a CalibrationStick instance."""
        self.scale_px_per_unit = stick.scale_px_per_unit
        self.unit_name = stick.unit

    def set_origin(self, origin: Point2D, angle_deg: float | None = None) -> None:
        """Updates origin and optionally the rotation angle."""
        self.origin = origin
        if angle_deg is not None:
            self.angle_deg = angle_deg

    def pixel_to_world(self, pixel_point: Point2D) -> Point2D:
        """Converts a pixel point to real-world laboratory coordinates."""
        if self.scale_px_per_unit <= 0.0:
            raise ValueError("Scale must be strictly positive.")

        # Invert screen Y so upward is positive
        du = pixel_point.x - self.origin.x
        dv = self.origin.y - pixel_point.y

        rad = math.radians(self.angle_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)

        # Rotate relative displacement
        x_rot = du * cos_t + dv * sin_t
        y_rot = -du * sin_t + dv * cos_t

        x_world = x_rot / self.scale_px_per_unit
        y_world = y_rot / self.scale_px_per_unit

        return Point2D(x=x_world, y=y_world)

    def world_to_pixel(self, world_point: Point2D) -> Point2D:
        """Converts real-world laboratory coordinates back to pixel space."""
        if self.scale_px_per_unit <= 0.0:
            raise ValueError("Scale must be strictly positive.")

        x_rot = world_point.x * self.scale_px_per_unit
        y_rot = world_point.y * self.scale_px_per_unit

        rad = math.radians(self.angle_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)

        # Inverse rotation
        du = x_rot * cos_t - y_rot * sin_t
        dv = x_rot * sin_t + y_rot * cos_t

        u = self.origin.x + du
        v = self.origin.y - dv

        return Point2D(x=u, y=v)

    def to_origin_model(self) -> CoordinateOrigin:
        """Returns the current state as a CoordinateOrigin model."""
        return CoordinateOrigin(origin=self.origin, angle_deg=self.angle_deg)
