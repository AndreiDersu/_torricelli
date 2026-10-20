"""Interactive Matplotlib plotting panel for kinematics analysis."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import ClassVar

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from light_tracker.core.models import Track, TrackPoint

matplotlib.use("TkAgg")


class PlotPanel(ttk.Frame):
    """Embeds an interactive Matplotlib figure to graph trajectory and kinematic variables."""

    VARIABLES: ClassVar[dict[str, tuple[str, str]]] = {
        "t": ("Tiempo", "s"),
        "x": ("Posición X", "m"),
        "y": ("Posición Y", "m"),
        "vx": ("Velocidad Vx", "m/s"),
        "vy": ("Velocidad Vy", "m/s"),
        "v": ("Rapidez v", "m/s"),
        "ax": ("Aceleración Ax", "m/s²"),
        "ay": ("Aceleración Ay", "m/s²"),
        "a": ("Aceleración Total a", "m/s²"),
    }

    def __init__(self, master: tk.Widget, **kwargs: object) -> None:
        super().__init__(master, **kwargs)

        self._current_track: Track | None = None
        self._current_frame: int | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        # Top toolbar: axis variable selectors
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=6, pady=4)

        ttk.Label(top_bar, text="Eje X:").pack(side="left", padx=(0, 2))
        self.combo_x = ttk.Combobox(
            top_bar,
            values=list(self.VARIABLES.keys()),
            width=5,
            state="readonly",
        )
        self.combo_x.set("t")
        self.combo_x.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        self.combo_x.pack(side="left", padx=(0, 10))

        ttk.Label(top_bar, text="Eje Y:").pack(side="left", padx=(0, 2))
        self.combo_y = ttk.Combobox(
            top_bar,
            values=list(self.VARIABLES.keys()),
            width=5,
            state="readonly",
        )
        self.combo_y.set("y")
        self.combo_y.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        self.combo_y.pack(side="left", padx=(0, 10))

        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

    def update_data(self, track: Track | None, current_frame: int | None = None) -> None:
        """Updates stored data and redraws the graph."""
        self._current_track = track
        self._current_frame = current_frame
        self.redraw()

    def redraw(self) -> None:
        """Renders the selected X vs Y plot."""
        self.ax.clear()
        self.ax.grid(True, linestyle="--", alpha=0.6)

        var_x = self.combo_x.get()
        var_y = self.combo_y.get()

        label_x, unit_x = self.VARIABLES.get(var_x, (var_x, ""))
        label_y, unit_y = self.VARIABLES.get(var_y, (var_y, ""))

        self.ax.set_xlabel(f"{label_x} [{unit_x}]" if unit_x else label_x)
        self.ax.set_ylabel(f"{label_y} [{unit_y}]" if unit_y else label_y)

        if self._current_track is None:
            self.ax.set_title("Sin datos")
            self.canvas.draw_idle()
            return

        points = self._current_track.get_sorted_points()
        if not points:
            self.ax.set_title(f"{self._current_track.name}: Sin puntos registrados")
            self.canvas.draw_idle()
            return

        x_vals = [self._extract_val(p, var_x) for p in points]
        y_vals = [self._extract_val(p, var_y) for p in points]

        track_color = self._current_track.color or "#E63946"

        # Trajectory line and scatter markers
        self.ax.plot(
            x_vals,
            y_vals,
            color=track_color,
            alpha=0.8,
            linestyle="-",
            label=self._current_track.name,
        )
        self.ax.scatter(
            x_vals, y_vals, color=track_color, s=24, edgecolors="white", linewidths=0.5, zorder=3
        )

        # Highlight current frame position
        if self._current_frame is not None:
            for p, x_v, y_v in zip(points, x_vals, y_vals):
                if p.frame_idx == self._current_frame:
                    self.ax.scatter(
                        [x_v],
                        [y_v],
                        color="#F59E0B",
                        s=80,
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=4,
                        label=f"Frame {self._current_frame}",
                    )
                    break

        self.ax.set_title(f"{label_y} vs {label_x} ({self._current_track.name})", fontsize=10)
        self.ax.legend(loc="best", fontsize=8)
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _extract_val(self, p: TrackPoint, var_name: str) -> float:
        mapping: dict[str, float] = {
            "t": p.t,
            "x": p.world_pos.x,
            "y": p.world_pos.y,
            "vx": p.vx,
            "vy": p.vy,
            "v": p.v,
            "ax": p.ax,
            "ay": p.ay,
            "a": p.a,
        }
        return mapping.get(var_name, 0.0)
