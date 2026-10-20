"""Project serialization and deserialization in JSON format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from light_tracker.core.models import (
    CalibrationStick,
    CoordinateOrigin,
    Point2D,
    ProjectData,
    Track,
    TrackPoint,
)


def save_project(project: ProjectData, file_path: str | Path) -> None:
    """Saves complete project state to a .trk.json file."""
    path = Path(file_path).resolve()

    # Store relative video path if in same or sub directory
    video_path_str = project.video_path
    try:
        vid_p = Path(project.video_path).resolve()
        proj_dir = path.parent
        rel_vid = vid_p.relative_to(proj_dir)
        video_path_str = str(rel_vid)
    except ValueError:
        video_path_str = project.video_path

    data: dict[str, Any] = {
        "version": "1.0",
        "video_path": video_path_str,
        "fps": project.fps,
        "total_frames": project.total_frames,
        "start_frame": project.start_frame,
        "end_frame": project.end_frame,
        "step_size": project.step_size,
        "coordinate_origin": {
            "x": project.coordinate_origin.origin.x,
            "y": project.coordinate_origin.origin.y,
            "angle_deg": project.coordinate_origin.angle_deg,
        },
        "calibration": (
            {
                "p1": {"x": project.calibration.p1.x, "y": project.calibration.p1.y},
                "p2": {"x": project.calibration.p2.x, "y": project.calibration.p2.y},
                "real_length": project.calibration.real_length,
                "unit": project.calibration.unit,
            }
            if project.calibration is not None
            else None
        ),
        "tracks": [
            {
                "id": t.id,
                "name": t.name,
                "color": t.color,
                "mass": t.mass,
                "points": [p.to_dict() for p in t.get_sorted_points()],
            }
            for t in project.tracks
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_project(file_path: str | Path) -> ProjectData:
    """Loads a project state from a .trk.json file."""
    path = Path(file_path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Resolve video path
    raw_vid = data.get("video_path", "")
    resolved_vid = raw_vid
    if raw_vid:
        candidate = (path.parent / raw_vid).resolve()
        if candidate.exists():
            resolved_vid = str(candidate)

    # Origin
    coord_raw = data.get("coordinate_origin", {})
    origin = CoordinateOrigin(
        origin=Point2D(x=float(coord_raw.get("x", 0.0)), y=float(coord_raw.get("y", 0.0))),
        angle_deg=float(coord_raw.get("angle_deg", 0.0)),
    )

    # Calibration
    calib = None
    cal_raw = data.get("calibration")
    if cal_raw:
        calib = CalibrationStick(
            p1=Point2D(x=float(cal_raw["p1"]["x"]), y=float(cal_raw["p1"]["y"])),
            p2=Point2D(x=float(cal_raw["p2"]["x"]), y=float(cal_raw["p2"]["y"])),
            real_length=float(cal_raw.get("real_length", 1.0)),
            unit=str(cal_raw.get("unit", "m")),
        )

    # Tracks
    tracks: list[Track] = []
    for t_raw in data.get("tracks", []):
        track = Track(
            id=str(t_raw.get("id", "track-0")),
            name=str(t_raw.get("name", "Masa A")),
            color=str(t_raw.get("color", "#E63946")),
            mass=float(t_raw.get("mass", 1.0)),
        )
        for p_raw in t_raw.get("points", []):
            pt = TrackPoint(
                frame_idx=int(p_raw["frame_idx"]),
                t=float(p_raw["t"]),
                pixel_pos=Point2D(x=float(p_raw["pixel_x"]), y=float(p_raw["pixel_y"])),
                world_pos=Point2D(x=float(p_raw["world_x"]), y=float(p_raw["world_y"])),
                vx=float(p_raw.get("vx", 0.0)),
                vy=float(p_raw.get("vy", 0.0)),
                v=float(p_raw.get("v", 0.0)),
                ax=float(p_raw.get("ax", 0.0)),
                ay=float(p_raw.get("ay", 0.0)),
                a=float(p_raw.get("a", 0.0)),
                score=float(p_raw.get("score", 1.0)),
                is_valid=bool(p_raw.get("is_valid", True)),
                is_keyframe=bool(p_raw.get("is_keyframe", False)),
            )
            track.add_point(pt)
        tracks.append(track)

    return ProjectData(
        video_path=resolved_vid,
        fps=float(data.get("fps", 30.0)),
        total_frames=int(data.get("total_frames", 0)),
        start_frame=int(data.get("start_frame", 0)),
        end_frame=int(data.get("end_frame", 0)),
        step_size=int(data.get("step_size", 1)),
        calibration=calib,
        coordinate_origin=origin,
        tracks=tracks,
    )
