"""Unit tests for coordinate system transformations and calibration."""

import math

from light_tracker.core.coordinate import CoordinateSystem
from light_tracker.core.models import CalibrationStick, Point2D


def test_point2d_distance() -> None:
    p1 = Point2D(0.0, 0.0)
    p2 = Point2D(3.0, 4.0)
    assert math.isclose(p1.distance_to(p2), 5.0)


def test_calibration_stick() -> None:
    p1 = Point2D(100.0, 200.0)
    p2 = Point2D(300.0, 200.0)
    stick = CalibrationStick(p1=p1, p2=p2, real_length=2.0, unit="m")

    assert math.isclose(stick.pixel_length, 200.0)
    assert math.isclose(stick.scale_px_per_unit, 100.0)  # 200 px / 2 m = 100 px/m


def test_coordinate_system_translation_and_scale() -> None:
    # Origin at (100, 200), scale = 50 px/m, angle = 0
    sys = CoordinateSystem(
        origin=Point2D(100.0, 200.0),
        angle_deg=0.0,
        scale_px_per_unit=50.0,
    )

    # Point at origin should be (0, 0) in world
    p_orig = sys.pixel_to_world(Point2D(100.0, 200.0))
    assert math.isclose(p_orig.x, 0.0, abs_tol=1e-7)
    assert math.isclose(p_orig.y, 0.0, abs_tol=1e-7)

    # Point to the right: u = 150 -> x = +1 m
    p_right = sys.pixel_to_world(Point2D(150.0, 200.0))
    assert math.isclose(p_right.x, 1.0, abs_tol=1e-7)
    assert math.isclose(p_right.y, 0.0, abs_tol=1e-7)

    # Point upwards in screen (v = 150 < 200) -> y = +1 m in physical world
    p_up = sys.pixel_to_world(Point2D(100.0, 150.0))
    assert math.isclose(p_up.x, 0.0, abs_tol=1e-7)
    assert math.isclose(p_up.y, 1.0, abs_tol=1e-7)


def test_coordinate_system_rotation() -> None:
    # Rotated by 90 degrees CCW
    sys = CoordinateSystem(
        origin=Point2D(0.0, 0.0),
        angle_deg=90.0,
        scale_px_per_unit=1.0,
    )

    # In screen space, (10, 0) is along original screen u (dx = 10, dy = 0)
    # With 90 deg rotation, this should map to y_world = -10, x_world = 0
    p = sys.pixel_to_world(Point2D(10.0, 0.0))
    assert math.isclose(p.x, 0.0, abs_tol=1e-7)
    assert math.isclose(p.y, -10.0, abs_tol=1e-7)


def test_round_trip_invertibility() -> None:
    sys = CoordinateSystem(
        origin=Point2D(123.4, 567.8),
        angle_deg=37.5,
        scale_px_per_unit=78.9,
    )

    test_points = [
        Point2D(0.0, 0.0),
        Point2D(123.4, 567.8),
        Point2D(350.2, 120.7),
        Point2D(-45.0, 800.0),
    ]

    for pt in test_points:
        world = sys.pixel_to_world(pt)
        restored = sys.world_to_pixel(world)
        assert math.isclose(restored.x, pt.x, abs_tol=1e-5)
        assert math.isclose(restored.y, pt.y, abs_tol=1e-5)
