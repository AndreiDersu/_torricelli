"""Unit tests for the kinematics engine."""

import math

import numpy as np

from light_tracker.core.kinematics import KinematicsEngine
from light_tracker.core.models import Point2D, TrackPoint


def test_empty_and_single_point() -> None:
    assert KinematicsEngine.compute([]) == []

    p1 = TrackPoint(frame_idx=0, t=0.0, pixel_pos=Point2D(0, 0), world_pos=Point2D(1, 2))
    res = KinematicsEngine.compute([p1])
    assert len(res) == 1
    assert res[0].v == 0.0
    assert res[0].a == 0.0


def test_constant_velocity_motion() -> None:
    # x(t) = 3.0 * t, y(t) = 4.0 * t  -> v = 5.0 m/s, a = 0.0 m/s^2
    dt = 0.1
    points: list[TrackPoint] = []
    for i in range(10):
        t = i * dt
        points.append(
            TrackPoint(
                frame_idx=i,
                t=t,
                pixel_pos=Point2D(0, 0),
                world_pos=Point2D(3.0 * t, 4.0 * t),
            )
        )

    res = KinematicsEngine.compute(points)
    assert len(res) == 10

    for p in res:
        assert math.isclose(p.vx, 3.0, abs_tol=1e-5)
        assert math.isclose(p.vy, 4.0, abs_tol=1e-5)
        assert math.isclose(p.v, 5.0, abs_tol=1e-5)
        assert math.isclose(p.ax, 0.0, abs_tol=1e-5)
        assert math.isclose(p.ay, 0.0, abs_tol=1e-5)
        assert math.isclose(p.a, 0.0, abs_tol=1e-5)


def test_constant_acceleration_motion() -> None:
    # y(t) = 0.5 * g * t^2 with g = 9.80 m/s^2
    g = 9.80
    dt = 0.05
    points: list[TrackPoint] = []
    for i in range(15):
        t = i * dt
        y = 0.5 * g * (t**2)
        points.append(
            TrackPoint(
                frame_idx=i,
                t=t,
                pixel_pos=Point2D(0, 0),
                world_pos=Point2D(0.0, y),
            )
        )

    res = KinematicsEngine.compute(points)

    # For internal points, central difference of quadratic is exact!
    for p in res[1:-1]:
        expected_v = g * p.t
        assert math.isclose(p.vy, expected_v, abs_tol=1e-4)
        assert math.isclose(p.ay, g, abs_tol=1e-4)


def test_savitzky_golay_smoothing() -> None:
    # Trajectory with simulated noise
    np.random.seed(42)
    dt = 0.05
    n = 20
    points: list[TrackPoint] = []
    for i in range(n):
        t = i * dt
        noise = np.random.normal(0, 0.02)
        y = 2.0 * t + noise
        points.append(
            TrackPoint(
                frame_idx=i,
                t=t,
                pixel_pos=Point2D(0, 0),
                world_pos=Point2D(0.0, y),
            )
        )

    res_raw = KinematicsEngine.compute(points, smooth=False)
    res_smooth = KinematicsEngine.compute(points, smooth=True, window_length=5, polyorder=2)

    # Variance of smoothed velocities should be lower than raw
    var_raw = np.var([p.vy for p in res_raw])
    var_smooth = np.var([p.vy for p in res_smooth])
    assert var_smooth < var_raw
