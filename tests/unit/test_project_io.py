"""Unit tests for project file serialization and export routines."""

import csv
import tempfile
from pathlib import Path

from light_tracker.core.models import (
    CalibrationStick,
    CoordinateOrigin,
    Point2D,
    ProjectData,
    Track,
    TrackPoint,
)
from light_tracker.io.export import export_to_csv, export_to_latex
from light_tracker.io.project import load_project, save_project


def test_project_save_and_load() -> None:
    track = Track(id="track-test", name="Masa 1", color="#3B82F6", mass=2.5)
    p0 = TrackPoint(
        frame_idx=0,
        t=0.0,
        pixel_pos=Point2D(100.0, 150.0),
        world_pos=Point2D(0.0, 0.0),
        vx=1.2,
        vy=3.4,
        v=3.6,
        ax=0.1,
        ay=-9.8,
        a=9.8,
        score=0.98,
    )
    track.add_point(p0)

    stick = CalibrationStick(p1=Point2D(10, 10), p2=Point2D(110, 10), real_length=1.5, unit="m")
    origin = CoordinateOrigin(origin=Point2D(50, 50), angle_deg=15.0)

    proj = ProjectData(
        video_path="test_video.mp4",
        fps=60.0,
        total_frames=120,
        start_frame=0,
        end_frame=119,
        step_size=2,
        calibration=stick,
        coordinate_origin=origin,
        tracks=[track],
    )

    with tempfile.NamedTemporaryFile(suffix=".trk.json", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        save_project(proj, tmp_path)
        loaded = load_project(tmp_path)

        assert loaded.fps == 60.0
        assert loaded.total_frames == 120
        assert loaded.step_size == 2
        assert loaded.coordinate_origin.angle_deg == 15.0
        assert loaded.calibration is not None
        assert loaded.calibration.real_length == 1.5

        assert len(loaded.tracks) == 1
        t_loaded = loaded.tracks[0]
        assert t_loaded.name == "Masa 1"
        assert t_loaded.mass == 2.5
        assert 0 in t_loaded.points
        p_loaded = t_loaded.points[0]
        assert p_loaded.pixel_pos.x == 100.0
        assert p_loaded.ay == -9.8
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_export_to_csv() -> None:
    track = Track(id="t1", name="Péndulo")
    track.add_point(
        TrackPoint(
            frame_idx=5,
            t=0.1667,
            pixel_pos=Point2D(50, 60),
            world_pos=Point2D(0.5, 0.6),
            vx=1.0,
            vy=2.0,
            v=2.236,
            ax=0.0,
            ay=-9.8,
            a=9.8,
            score=0.95,
        )
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)

    try:
        export_to_csv(track, csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            assert reader[0]["frame"] == "5"
            assert float(reader[0]["x"]) == 0.5
            assert float(reader[0]["score"]) == 0.95
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_export_to_latex() -> None:
    track = Track(id="t1", name="Esfera")
    track.add_point(
        TrackPoint(
            frame_idx=0,
            t=0.0,
            pixel_pos=Point2D(10, 10),
            world_pos=Point2D(0.0, 1.0),
            vx=0.0,
            vy=0.0,
            v=0.0,
            ax=0.0,
            ay=-9.8,
            a=9.8,
        )
    )
    latex_code = export_to_latex(track)
    assert "\\begin{table}" in latex_code
    assert "\\caption{Datos cinemáticos de Esfera}" in latex_code
    assert "\\end{table}" in latex_code
