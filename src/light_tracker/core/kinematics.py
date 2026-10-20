"""Kinematics computation engine for particle trajectory analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from light_tracker.core.models import TrackPoint


class KinematicsEngine:
    """Calculates velocities and accelerations for a sequence of track points.

    Supports both standard finite central differences and Savitzky-Golay
    filtering for noise reduction in experimental physics data.
    """

    @staticmethod
    def compute(
        points: Sequence[TrackPoint],
        smooth: bool = False,
        window_length: int = 5,
        polyorder: int = 2,
    ) -> list[TrackPoint]:
        """Updates and returns points with computed velocities and accelerations.

        Args:
            points: Sequence of TrackPoint instances sorted chronologically.
            smooth: If True, applies Savitzky-Golay smoothing to positions before differentiation.
            window_length: Window size for Savitzky-Golay filter (must be odd and <= len(points)).
            polyorder: Polynomial order for Savitzky-Golay filter.

        Returns:
            List of TrackPoint with updated vx, vy, v, ax, ay, a fields.
        """
        n = len(points)
        if n == 0:
            return []

        # Make copy of points
        out = [
            TrackPoint(
                frame_idx=p.frame_idx,
                t=p.t,
                pixel_pos=p.pixel_pos,
                world_pos=p.world_pos,
                vx=0.0,
                vy=0.0,
                v=0.0,
                ax=0.0,
                ay=0.0,
                a=0.0,
                score=p.score,
                is_valid=p.is_valid,
                is_keyframe=p.is_keyframe,
            )
            for p in points
        ]

        if n == 1:
            return out

        t_arr = np.array([p.t for p in points], dtype=np.float64)
        x_arr = np.array([p.world_pos.x for p in points], dtype=np.float64)
        y_arr = np.array([p.world_pos.y for p in points], dtype=np.float64)

        # Check for duplicate timestamps to prevent zero division
        dt_min = np.min(np.diff(t_arr))
        if dt_min <= 0.0:
            # Fallback uniform spacing if timestamps are identical
            t_arr = np.arange(n, dtype=np.float64) * (1.0 / 30.0)

        if smooth and n >= 5:
            from scipy.signal import savgol_filter

            w = min(window_length, n if n % 2 != 0 else n - 1)
            if w > polyorder and w >= 3:
                x_arr = savgol_filter(x_arr, window_length=w, polyorder=polyorder)
                y_arr = savgol_filter(y_arr, window_length=w, polyorder=polyorder)

        # Compute velocities: central difference for internal points, 2nd-order for boundaries
        vx = np.zeros(n, dtype=np.float64)
        vy = np.zeros(n, dtype=np.float64)

        if n == 2:
            dt = t_arr[1] - t_arr[0]
            if dt > 0:
                v_x_const = (x_arr[1] - x_arr[0]) / dt
                v_y_const = (y_arr[1] - y_arr[0]) / dt
                vx[0] = vx[1] = v_x_const
                vy[0] = vy[1] = v_y_const
        else:
            # 2nd order forward difference at start: (-3 x0 + 4 x1 - x2) / (t2 - t0)
            dt_start = t_arr[2] - t_arr[0]
            if dt_start > 0:
                vx[0] = (-3.0 * x_arr[0] + 4.0 * x_arr[1] - x_arr[2]) / dt_start
                vy[0] = (-3.0 * y_arr[0] + 4.0 * y_arr[1] - y_arr[2]) / dt_start
            else:
                vx[0] = vy[0] = 0.0

            # Central difference interior
            dt_mid = t_arr[2:] - t_arr[:-2]
            valid_dt = np.where(dt_mid > 0, dt_mid, 1.0)
            vx[1:-1] = (x_arr[2:] - x_arr[:-2]) / valid_dt
            vy[1:-1] = (y_arr[2:] - y_arr[:-2]) / valid_dt

            # 2nd order backward difference at end: (3 x_{n-1} - 4 x_{n-2} + x_{n-3}) / (t_{n-1} - t_{n-3})
            dt_end = t_arr[-1] - t_arr[-3]
            if dt_end > 0:
                vx[-1] = (3.0 * x_arr[-1] - 4.0 * x_arr[-2] + x_arr[-3]) / dt_end
                vy[-1] = (3.0 * y_arr[-1] - 4.0 * y_arr[-2] + y_arr[-3]) / dt_end
            else:
                vx[-1] = vy[-1] = 0.0

        # Compute accelerations directly from position 2nd derivative stencil:
        # a_i = 2 * ((x_{i+1} - x_i)/(t_{i+1} - t_i) - (x_i - x_{i-1})/(t_i - t_{i-1})) / (t_{i+1} - t_{i-1})
        ax = np.zeros(n, dtype=np.float64)
        ay = np.zeros(n, dtype=np.float64)

        if n >= 3:
            dt_fwd = t_arr[2:] - t_arr[1:-1]
            dt_bwd = t_arr[1:-1] - t_arr[:-2]
            dt_span = t_arr[2:] - t_arr[:-2]

            valid_fwd = np.where(dt_fwd > 0, dt_fwd, 1.0)
            valid_bwd = np.where(dt_bwd > 0, dt_bwd, 1.0)
            valid_span = np.where(dt_span > 0, dt_span, 1.0)

            slope_fwd_x = (x_arr[2:] - x_arr[1:-1]) / valid_fwd
            slope_bwd_x = (x_arr[1:-1] - x_arr[:-2]) / valid_bwd
            ax[1:-1] = 2.0 * (slope_fwd_x - slope_bwd_x) / valid_span

            slope_fwd_y = (y_arr[2:] - y_arr[1:-1]) / valid_fwd
            slope_bwd_y = (y_arr[1:-1] - y_arr[:-2]) / valid_bwd
            ay[1:-1] = 2.0 * (slope_fwd_y - slope_bwd_y) / valid_span

            # Boundary accelerations extrapolated from adjacent points
            ax[0] = ax[1]
            ay[0] = ay[1]
            ax[-1] = ax[-2]
            ay[-1] = ay[-2]

        # Populate points
        for i in range(n):
            v_val = math.hypot(float(vx[i]), float(vy[i]))
            a_val = math.hypot(float(ax[i]), float(ay[i]))
            out[i].vx = float(vx[i])
            out[i].vy = float(vy[i])
            out[i].v = v_val
            out[i].ax = float(ax[i])
            out[i].ay = float(ay[i])
            out[i].a = a_val

        return out
