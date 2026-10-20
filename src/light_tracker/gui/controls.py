"""Transport playback control bar for video navigation."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk


class PlaybackControls(ttk.Frame):
    """Playback transport bar with play/pause, stepping, and seeking slider."""

    def __init__(self, master: tk.Widget, **kwargs: object) -> None:
        super().__init__(master, **kwargs)

        self.on_seek: Callable[[int], None] | None = None
        self.on_step: Callable[[int], None] | None = None
        self.on_play_toggle: Callable[[], None] | None = None
        self.on_step_size_changed: Callable[[int], None] | None = None

        self.is_playing: bool = False
        self.step_size: int = 1
        self.total_frames: int = 1
        self.current_frame: int = 0
        self.fps: float = 30.0

        self._build_ui()

    def _build_ui(self) -> None:
        # Top row: Transport slider
        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill="x", padx=6, pady=(4, 2))

        self.scale_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=1,
            variable=self.scale_var,
            orient="horizontal",
            command=self._on_slider_move,
        )
        self.slider.pack(side="left", fill="x", expand=True)

        self.lbl_info = ttk.Label(
            slider_frame, text="Frame: 0 / 0 | t: 0.000 s", width=30, anchor="e"
        )
        self.lbl_info.pack(side="right", padx=(8, 0))

        # Bottom row: Control buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=6, pady=(0, 4))

        self.btn_first = ttk.Button(
            btn_frame, text="|<<", width=4, command=lambda: self._trigger_step_abs(0)
        )
        self.btn_first.pack(side="left", padx=2)

        self.btn_prev = ttk.Button(
            btn_frame, text="<", width=4, command=lambda: self._trigger_step(-self.step_size)
        )
        self.btn_prev.pack(side="left", padx=2)

        self.btn_play = ttk.Button(btn_frame, text="▶ Play", width=8, command=self._trigger_play)
        self.btn_play.pack(side="left", padx=4)

        self.btn_next = ttk.Button(
            btn_frame, text=">", width=4, command=lambda: self._trigger_step(self.step_size)
        )
        self.btn_next.pack(side="left", padx=2)

        self.btn_last = ttk.Button(
            btn_frame,
            text=">>|",
            width=4,
            command=lambda: self._trigger_step_abs(self.total_frames - 1),
        )
        self.btn_last.pack(side="left", padx=2)

        # Step size selector
        ttk.Label(btn_frame, text="Paso:").pack(side="left", padx=(16, 4))
        self.combo_step = ttk.Combobox(
            btn_frame, values=["1", "2", "5", "10"], width=4, state="readonly"
        )
        self.combo_step.set("1")
        self.combo_step.bind("<<ComboboxSelected>>", self._on_combo_step)
        self.combo_step.pack(side="left")

    def configure_video(self, total_frames: int, fps: float) -> None:
        """Configures slider range and FPS metadata."""
        self.total_frames = max(1, total_frames)
        self.fps = max(1.0, fps)
        self.slider.configure(to=max(1, self.total_frames - 1))
        self.set_frame_index(0)

    def set_frame_index(self, frame_idx: int) -> None:
        """Updates slider position and information text without triggering callbacks."""
        self.current_frame = max(0, min(self.total_frames - 1, frame_idx))
        self.scale_var.set(self.current_frame)
        t = self.current_frame / self.fps
        self.lbl_info.configure(
            text=f"Frame: {self.current_frame} / {self.total_frames - 1} | t: {t:.3f} s"
        )

    def set_playing_state(self, playing: bool) -> None:
        """Updates Play/Pause button appearance."""
        self.is_playing = playing
        self.btn_play.configure(text="⏸ Pausa" if playing else "▶ Play")

    def _on_slider_move(self, value_str: str) -> None:
        try:
            val = round(float(value_str))
            if self.on_seek and val != self.current_frame:
                self.on_seek(val)
        except ValueError:
            pass

    def _trigger_step(self, delta: int) -> None:
        if self.on_step:
            self.on_step(delta)

    def _trigger_step_abs(self, frame_idx: int) -> None:
        if self.on_seek:
            self.on_seek(frame_idx)

    def _trigger_play(self) -> None:
        if self.on_play_toggle:
            self.on_play_toggle()

    def _on_combo_step(self, event: tk.Event) -> None:
        try:
            val = int(self.combo_step.get())
            self.step_size = val
            if self.on_step_size_changed:
                self.on_step_size_changed(val)
        except ValueError:
            pass
