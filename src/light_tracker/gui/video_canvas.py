"""Interactive video canvas widget with overlays for axes, stick, and tracks."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageTk

from light_tracker.core.coordinate import CoordinateSystem
from light_tracker.core.models import CalibrationStick, Point2D, Track


class VideoCanvas(tk.Canvas):
    """Tkinter Canvas with hardware-like fast rendering and scientific overlays."""

    def __init__(self, master: tk.Widget, **kwargs: object) -> None:
        super().__init__(master, bg="#1E1E1E", highlightthickness=0, **kwargs)

        self._current_frame: np.ndarray | None = None
        self._photo_image: ImageTk.PhotoImage | None = None
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # Overlays state
        self.show_axes: bool = True
        self.show_stick: bool = True
        self.show_tracks: bool = True
        self.template_rect: tuple[float, float, float, float] | None = None  # (cx, cy, w, h)
        self.search_rect: tuple[float, float, float, float] | None = None  # (cx, cy, w, h)

        # Callbacks
        self.on_click_canvas: Callable[[Point2D], None] | None = None
        self.on_drag_canvas: Callable[[Point2D], None] | None = None

        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)

    def set_frame(self, frame_rgb: np.ndarray | None) -> None:
        """Updates the video frame displayed on the canvas."""
        self._current_frame = frame_rgb
        self.redraw()

    def frame_to_canvas(self, p: Point2D) -> tuple[float, float]:
        """Converts frame pixel coordinates to canvas widget coordinates."""
        cx = self._offset_x + p.x * self._scale
        cy = self._offset_y + p.y * self._scale
        return cx, cy

    def canvas_to_frame(self, cx: float, cy: float) -> Point2D:
        """Converts canvas widget coordinates to frame pixel coordinates."""
        if self._scale <= 0.0:
            return Point2D(0.0, 0.0)
        fx = (cx - self._offset_x) / self._scale
        fy = (cy - self._offset_y) / self._scale
        return Point2D(fx, fy)

    def redraw(
        self,
        coord_sys: CoordinateSystem | None = None,
        stick: CalibrationStick | None = None,
        tracks: list[Track] | None = None,
        current_frame_idx: int = 0,
    ) -> None:
        """Redraws the frame image and all visible overlays."""
        self.delete("all")

        if self._current_frame is None:
            w = self.winfo_width()
            h = self.winfo_height()
            self.create_text(
                w / 2,
                h / 2,
                text="Sin video cargado\n(Archivo > Abrir Video)",
                fill="#888888",
                font=("sans-serif", 13, "bold"),
                justify="center",
            )
            return

        fh, fw = self._current_frame.shape[:2]
        cw = max(1, self.winfo_width())
        ch = max(1, self.winfo_height())

        # Preserve aspect ratio
        self._scale = min(cw / fw, ch / fh)
        dw = round(fw * self._scale)
        dh = round(fh * self._scale)
        self._offset_x = (cw - dw) / 2.0
        self._offset_y = (ch - dh) / 2.0

        # Resize image using fast bilinear filtering
        pil_img = Image.fromarray(self._current_frame)
        pil_resized = pil_img.resize((dw, dh), Image.Resampling.BILINEAR)
        self._photo_image = ImageTk.PhotoImage(pil_resized)

        self.create_image(
            self._offset_x,
            self._offset_y,
            anchor="nw",
            image=self._photo_image,
        )

        # Draw overlays
        if self.show_stick and stick is not None:
            self._draw_stick(stick)

        if self.show_axes and coord_sys is not None:
            self._draw_axes(coord_sys)

        if self.show_tracks and tracks:
            self._draw_tracks(tracks, current_frame_idx)

        self._draw_autotracker_boxes()

    def _draw_axes(self, coord_sys: CoordinateSystem) -> None:
        """Renders coordinate axes with origin and direction arrows."""
        ox, oy = self.frame_to_canvas(coord_sys.origin)

        # Axis length on screen
        axis_len = 100.0
        rad = math.radians(coord_sys.angle_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)

        # X-axis (direction cos_t, -sin_t due to inverted screen Y)
        xx = ox + axis_len * cos_t
        xy = oy - axis_len * sin_t

        # Y-axis (direction sin_t, cos_t)
        yx = ox + axis_len * sin_t
        yy = oy + axis_len * cos_t

        # X-axis line (magenta)
        self.create_line(
            ox, oy, xx, xy, fill="#D946EF", width=2, arrow="last", arrowshape=(8, 10, 3)
        )
        self.create_text(
            xx + 10 * cos_t,
            xy - 10 * sin_t,
            text="x",
            fill="#D946EF",
            font=("sans-serif", 10, "bold"),
        )

        # Y-axis line (cyan/magenta)
        self.create_line(
            ox, oy, yx, yy, fill="#D946EF", width=2, arrow="last", arrowshape=(8, 10, 3)
        )
        self.create_text(
            yx + 10 * sin_t,
            yy + 10 * cos_t,
            text="y",
            fill="#D946EF",
            font=("sans-serif", 10, "bold"),
        )

        # Origin point handle
        r = 5
        self.create_oval(ox - r, oy - r, ox + r, oy + r, fill="#D946EF", outline="#FFFFFF", width=1)

    def _draw_stick(self, stick: CalibrationStick) -> None:
        """Renders the calibration stick line and endpoints."""
        x1, y1 = self.frame_to_canvas(stick.p1)
        x2, y2 = self.frame_to_canvas(stick.p2)

        # Line
        self.create_line(x1, y1, x2, y2, fill="#06B6D4", width=2, dash=(4, 4))

        # End handles
        r = 6
        self.create_oval(
            x1 - r, y1 - r, x1 + r, y1 + r, fill="#06B6D4", outline="#FFFFFF", width=1.5
        )
        self.create_oval(
            x2 - r, y2 - r, x2 + r, y2 + r, fill="#06B6D4", outline="#FFFFFF", width=1.5
        )

        # Length label at midpoint
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        label_text = f"{stick.real_length:.2f} {stick.unit}"
        self.create_text(
            mx, my - 12, text=label_text, fill="#06B6D4", font=("sans-serif", 9, "bold")
        )

    def _draw_tracks(self, tracks: list[Track], current_frame_idx: int) -> None:
        """Renders points and path trajectories for all tracks."""
        for track in tracks:
            pts = track.get_sorted_points()
            if not pts:
                continue

            color = track.color

            # Draw trajectory trail up to current frame
            coords: list[float] = []
            for p in pts:
                if p.frame_idx <= current_frame_idx:
                    cx, cy = self.frame_to_canvas(p.pixel_pos)
                    coords.extend([cx, cy])

            if len(coords) >= 4:
                self.create_line(*coords, fill=color, width=1.5, smooth=False)

            # Draw points
            for p in pts:
                cx, cy = self.frame_to_canvas(p.pixel_pos)
                if p.frame_idx == current_frame_idx:
                    # Current frame highlighted point: crosshair + circle
                    r = 7
                    self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
                    self.create_line(cx - 10, cy, cx + 10, cy, fill=color, width=1.5)
                    self.create_line(cx, cy - 10, cx, cy + 10, fill=color, width=1.5)
                else:
                    # Past or future points: small solid circles
                    r = 3
                    self.create_oval(
                        cx - r, cy - r, cx + r, cy + r, fill=color, outline="#FFFFFF", width=0.5
                    )

    def _draw_autotracker_boxes(self) -> None:
        """Draws template and search area boxes if active."""
        if self.search_rect is not None:
            cx, cy, sw, sh = self.search_rect
            p1 = Point2D(cx - sw / 2.0, cy - sh / 2.0)
            p2 = Point2D(cx + sw / 2.0, cy + sh / 2.0)
            x1, y1 = self.frame_to_canvas(p1)
            x2, y2 = self.frame_to_canvas(p2)
            self.create_rectangle(x1, y1, x2, y2, outline="#FBBF24", width=1.5, dash=(4, 2))

        if self.template_rect is not None:
            cx, cy, tw, th = self.template_rect
            p1 = Point2D(cx - tw / 2.0, cy - th / 2.0)
            p2 = Point2D(cx + tw / 2.0, cy + th / 2.0)
            x1, y1 = self.frame_to_canvas(p1)
            x2, y2 = self.frame_to_canvas(p2)
            self.create_rectangle(x1, y1, x2, y2, outline="#10B981", width=2)

    def _on_resize(self, event: tk.Event) -> None:
        self.redraw()

    def _on_mouse_down(self, event: tk.Event) -> None:
        if self._current_frame is not None and self.on_click_canvas:
            pos_frame = self.canvas_to_frame(event.x, event.y)
            self.on_click_canvas(pos_frame)

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self._current_frame is not None and self.on_drag_canvas:
            pos_frame = self.canvas_to_frame(event.x, event.y)
            self.on_drag_canvas(pos_frame)
