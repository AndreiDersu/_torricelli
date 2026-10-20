"""Generates a realistic physics laboratory demonstration video for LightTracker."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def generate_physics_demo_video(output_path: str | Path = "demo_caida_libre.mp4") -> Path:
    """Generates a demo video of a bouncing ball in a lab setup with a 1-meter calibration scale."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 640, 480
    fps = 30.0
    num_frames = 90  # 3.0 seconds
    dt = 1.0 / fps

    # Physics simulation parameters
    # Real world: 1 meter = 300 pixels
    scale_px_per_m = 300.0
    g = 9.80  # m/s^2

    # Physical coordinates: release from x = 0.5 m, y = 1.3 m
    x_m = 0.4
    y_m = 1.3
    vx_m = 0.35  # m/s horizontal drift
    vy_m = 0.0   # initial vertical velocity
    cor = 0.78   # coefficient of restitution on bounce

    # Floor at y = 0.1 m
    floor_y_m = 0.1

    # Coordinate mapping:
    # origin (0, 0) at pixel (100, 440)
    origin_px = (100, 440)

    def world_to_px(xw: float, yw: float) -> tuple[int, int]:
        px = int(round(origin_px[0] + xw * scale_px_per_m))
        py = int(round(origin_px[1] - yw * scale_px_per_m))
        return px, py

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    ball_radius_px = 12

    for f_idx in range(num_frames):
        # Physics update (Euler-Cromer)
        vy_m -= g * dt
        x_m += vx_m * dt
        y_m += vy_m * dt

        # Bounce detection
        if y_m <= floor_y_m:
            y_m = floor_y_m
            vy_m = -vy_m * cor
            if abs(vy_m) < 0.2:
                vy_m = 0.0

        # Background rendering: Laboratory wall with subtle grid
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240

        # Subtle grid lines
        for gx in range(0, width, 50):
            cv2.line(frame, (gx, 0), (gx, height), (225, 225, 225), 1)
        for gy in range(0, height, 50):
            cv2.line(frame, (0, gy), (width, gy), (225, 225, 225), 1)

        # Floor line
        floor_px = int(round(origin_px[1] - floor_y_m * scale_px_per_m))
        cv2.line(frame, (0, floor_px + ball_radius_px), (width, floor_px + ball_radius_px), (80, 80, 80), 3)

        # Draw a 1.0-meter Calibration Reference Stick on the left wall
        stick_top_px = world_to_px(0.05, 1.1)
        stick_bot_px = world_to_px(0.05, 0.1)  # Exactly 1.0 meter span
        cv2.line(frame, stick_top_px, stick_bot_px, (20, 20, 20), 4)
        cv2.line(frame, (stick_top_px[0] - 8, stick_top_px[1]), (stick_top_px[0] + 8, stick_top_px[1]), (20, 20, 20), 3)
        cv2.line(frame, (stick_bot_px[0] - 8, stick_bot_px[1]), (stick_bot_px[0] + 8, stick_bot_px[1]), (20, 20, 20), 3)
        cv2.putText(frame, "1.00 m", (stick_top_px[0] + 12, (stick_top_px[1] + stick_bot_px[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

        # Draw the target red ball
        bx, by = world_to_px(x_m, y_m)
        # Shadow on floor
        shadow_width = max(4, int(ball_radius_px * (1.0 - (y_m - floor_y_m) * 0.5)))
        cv2.ellipse(frame, (bx, floor_px + ball_radius_px + 2), (shadow_width, 4), 0, 0, 360, (190, 190, 190), -1)

        # Red ball with specular highlight
        cv2.circle(frame, (bx, by), ball_radius_px, (30, 40, 220), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx - 3, by - 3), 3, (120, 130, 255), -1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Video demo generado exitosamente en: {path}")
    return path


if __name__ == "__main__":
    generate_physics_demo_video()
