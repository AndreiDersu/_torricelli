"""Command-line interface (CLI) for headless video tracking and batch processing."""

from __future__ import annotations

import argparse
import sys

from light_tracker.core.coordinate import CoordinateSystem
from light_tracker.core.kinematics import KinematicsEngine
from light_tracker.core.models import Point2D, Track, TrackPoint
from light_tracker.core.video import VideoReader
from light_tracker.io.export import export_to_csv, export_to_latex
from light_tracker.io.project import load_project
from light_tracker.tracking.template_tracker import TemplateTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="light-tracker-cli",
        description="LightTracker: Headless CLI for automated object tracking in physics videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: info
    parser_info = subparsers.add_parser("info", help="Muestra metadatos del video.")
    parser_info.add_argument("video", type=str, help="Ruta al archivo de video.")

    # Subcommand: track
    parser_track = subparsers.add_parser("track", help="Rastreo desatendido por plantilla.")
    parser_track.add_argument("video", type=str, help="Ruta al archivo de video.")
    parser_track.add_argument(
        "--output", "-o", type=str, required=True, help="Archivo CSV de salida."
    )
    parser_track.add_argument(
        "--start-x", type=float, required=True, help="Coordenada X inicial en píxeles."
    )
    parser_track.add_argument(
        "--start-y", type=float, required=True, help="Coordenada Y inicial en píxeles."
    )
    parser_track.add_argument("--start-frame", type=int, default=0, help="Fotograma inicial.")
    parser_track.add_argument(
        "--end-frame", type=int, default=-1, help="Fotograma final (-1 para fin de video)."
    )
    parser_track.add_argument(
        "--scale", type=float, default=1.0, help="Escala de calibración (px / metro)."
    )
    parser_track.add_argument(
        "--score-threshold", type=float, default=0.55, help="Umbral de correlación."
    )

    # Subcommand: export
    parser_export = subparsers.add_parser("export", help="Exportar datos de un proyecto guardado.")
    parser_export.add_argument("project", type=str, help="Ruta al archivo .trk.json.")
    parser_export.add_argument("--csv", type=str, default="", help="Ruta de salida CSV.")
    parser_export.add_argument(
        "--latex", action="store_true", help="Imprimir código de tabla LaTeX."
    )

    args = parser.parse_args()

    if args.command == "info":
        with VideoReader(args.video) as reader:
            print(f"Video: {reader.path.name}")
            print(f"Resolución: {reader.width} x {reader.height} px")
            print(f"Tasa de cuadros (FPS): {reader.fps:.2f}")
            print(f"Total fotogramas: {reader.total_frames}")
            print(f"Duración: {reader.duration:.2f} s")

    elif args.command == "track":
        with VideoReader(args.video) as reader:
            start_f = max(0, min(reader.total_frames - 1, args.start_frame))
            end_f = (
                reader.total_frames
                if args.end_frame < 0
                else min(reader.total_frames, args.end_frame + 1)
            )

            coord_sys = CoordinateSystem(scale_px_per_unit=args.scale)
            tracker = TemplateTracker(min_score=args.score_threshold)

            # Initialize on first frame
            init_frame = reader.get_frame(start_f)
            if init_frame is None:
                print(f"Error: no se pudo leer fotograma {start_f}", file=sys.stderr)
                sys.exit(1)

            init_pos = Point2D(args.start_x, args.start_y)
            tracker.initialize(init_frame, init_pos)

            track = Track(id="track-0", name="Particula CLI")
            # Add initial point
            p0 = TrackPoint(
                frame_idx=start_f,
                t=start_f / reader.fps,
                pixel_pos=init_pos,
                world_pos=coord_sys.pixel_to_world(init_pos),
                score=1.0,
            )
            track.add_point(p0)

            print(f"Iniciando seguimiento desde frame {start_f} hasta {end_f - 1}...")

            curr_pos = init_pos
            for f_idx in range(start_f + 1, end_f):
                frame = reader.get_frame(f_idx)
                if frame is None:
                    break

                # Lookahead velocity estimate
                vel_est = None
                pts = track.get_sorted_points()
                if len(pts) >= 2:
                    vel_est = Point2D(
                        pts[-1].pixel_pos.x - pts[-2].pixel_pos.x,
                        pts[-1].pixel_pos.y - pts[-2].pixel_pos.y,
                    )

                pos, score = tracker.track_frame(
                    frame, prev_pos=curr_pos, velocity_estimate=vel_est
                )
                if pos is not None and score >= args.score_threshold:
                    curr_pos = pos
                    pt = TrackPoint(
                        frame_idx=f_idx,
                        t=f_idx / reader.fps,
                        pixel_pos=pos,
                        world_pos=coord_sys.pixel_to_world(pos),
                        score=score,
                    )
                    track.add_point(pt)
                else:
                    print(f"Pérdida de seguimiento en frame {f_idx} (score: {score:.2f})")
                    break

            # Kinematics
            pts = track.get_sorted_points()
            computed = KinematicsEngine.compute(pts)
            track.points = {p.frame_idx: p for p in computed}

            export_to_csv(track, args.output)
            print(f"Seguimiento completado. {len(track.points)} puntos guardados en {args.output}")

    elif args.command == "export":
        proj = load_project(args.project)
        if not proj.tracks:
            print("El proyecto no contiene masas registradas.", file=sys.stderr)
            sys.exit(1)

        track = proj.tracks[0]
        if args.csv:
            export_to_csv(track, args.csv)
            print(f"Exportado a {args.csv}")

        if args.latex:
            print(export_to_latex(track))


if __name__ == "__main__":
    main()
