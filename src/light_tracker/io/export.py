"""Data export routines for scientific reports, CSV, and Pandas."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from light_tracker.core.models import Track


def export_to_csv(track: Track, file_path: str | Path) -> None:
    """Exports kinematic track data to a clean CSV file."""
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    points = track.get_sorted_points()
    fieldnames = [
        "frame",
        "t",
        "x",
        "y",
        "vx",
        "vy",
        "v",
        "ax",
        "ay",
        "a",
        "pixel_x",
        "pixel_y",
        "score",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "frame": p.frame_idx,
                    "t": f"{p.t:.4f}",
                    "x": f"{p.world_pos.x:.6f}",
                    "y": f"{p.world_pos.y:.6f}",
                    "vx": f"{p.vx:.6f}",
                    "vy": f"{p.vy:.6f}",
                    "v": f"{p.v:.6f}",
                    "ax": f"{p.ax:.6f}",
                    "ay": f"{p.ay:.6f}",
                    "a": f"{p.a:.6f}",
                    "pixel_x": f"{p.pixel_pos.x:.2f}",
                    "pixel_y": f"{p.pixel_pos.y:.2f}",
                    "score": f"{p.score:.3f}",
                }
            )


def export_to_dataframe(track: Track) -> pd.DataFrame:
    """Converts kinematic track data into a pandas DataFrame."""
    import pandas as pd

    points = track.get_sorted_points()
    data = [
        {
            "frame": p.frame_idx,
            "t": p.t,
            "x": p.world_pos.x,
            "y": p.world_pos.y,
            "vx": p.vx,
            "vy": p.vy,
            "v": p.v,
            "ax": p.ax,
            "ay": p.ay,
            "a": p.a,
            "pixel_x": p.pixel_pos.x,
            "pixel_y": p.pixel_pos.y,
            "score": p.score,
        }
        for p in points
    ]
    return pd.DataFrame(data)


def export_to_latex(track: Track, max_rows: int = 30) -> str:
    """Generates a publication-quality LaTeX table for physics reports."""
    points = track.get_sorted_points()
    if not points:
        return "% No data available\n"

    # If points exceed max_rows, downsample evenly
    if len(points) > max_rows:
        step = max(1, len(points) // max_rows)
        subset = points[::step]
        if points[-1] not in subset:
            subset.append(points[-1])
    else:
        subset = points

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Datos cinemáticos de {track.name}}}",
        "\\label{tab:cinematica_" + track.name.lower().replace(" ", "_") + "}",
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "$t$ (\\unit{\\second}) & $x$ (\\unit{\\meter}) & $y$ (\\unit{\\meter}) & $v_x$ (\\unit{\\meter\\per\\second}) & $v_y$ (\\unit{\\meter\\per\\second}) & $a_y$ (\\unit{\\meter\\per\\second\\squared}) \\\\",
        "\\midrule",
    ]

    for p in subset:
        row = f"{p.t:.3f} & {p.world_pos.x:.3f} & {p.world_pos.y:.3f} & {p.vx:.3f} & {p.vy:.3f} & {p.ay:.3f} \\\\"
        lines.append(row)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)
