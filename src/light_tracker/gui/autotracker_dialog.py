"""AutoTracker wizard dialog and controller."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from light_tracker.tracking.template_tracker import TemplateTracker


class AutoTrackerDialog(tk.Toplevel):
    """Wizard window for configuring and driving the automated NCC tracking process."""

    def __init__(
        self,
        master: tk.Widget,
        tracker: TemplateTracker,
        on_capture_template: Callable[[], None],
        on_step_search: Callable[[], bool],
        on_run_search: Callable[[], None],
        on_stop_search: Callable[[], None],
    ) -> None:
        super().__init__(master)
        self.title("AutoTracker - Asistente de Rastreo Automático")
        self.geometry("380x350")
        self.resizable(False, False)

        self.tracker = tracker
        self.on_capture_template = on_capture_template
        self.on_step_search = on_step_search
        self.on_run_search = on_run_search
        self.on_stop_search = on_stop_search

        self.is_running: bool = False

        self._build_ui()
        self.transient(master)

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # Header info
        lbl_info = ttk.Label(
            self,
            text="Configura la plantilla de búsqueda y ejecuta el rastreo automático.",
            wraplength=350,
            justify="left",
        )
        lbl_info.pack(fill="x", padx=10, pady=(10, 6))

        # Capture button
        self.btn_capture = ttk.Button(
            self,
            text="1. Capturar plantilla en el cursor / fotograma actual",
            command=self.on_capture_template,
        )
        self.btn_capture.pack(fill="x", **pad)

        # Settings Group
        grp_settings = ttk.LabelFrame(self, text="Parámetros de Búsqueda")
        grp_settings.pack(fill="x", padx=10, pady=6)

        # Template size
        f_ts = ttk.Frame(grp_settings)
        f_ts.pack(fill="x", padx=6, pady=2)
        ttk.Label(f_ts, text="Tamaño plantilla (px):").pack(side="left")
        self.spin_tw = ttk.Spinbox(f_ts, from_=8, to=128, width=5)
        self.spin_tw.set(str(self.tracker.template_size[0]))
        self.spin_tw.pack(side="right")
        self.spin_tw.bind("<FocusOut>", self._apply_settings)

        # Search window size
        f_ss = ttk.Frame(grp_settings)
        f_ss.pack(fill="x", padx=6, pady=2)
        ttk.Label(f_ss, text="Ventana de búsqueda (px):").pack(side="left")
        self.spin_sw = ttk.Spinbox(f_ss, from_=16, to=256, width=5)
        self.spin_sw.set(str(self.tracker.search_size[0]))
        self.spin_sw.pack(side="right")
        self.spin_sw.bind("<FocusOut>", self._apply_settings)

        # Match threshold
        f_th = ttk.Frame(grp_settings)
        f_th.pack(fill="x", padx=6, pady=2)
        ttk.Label(f_th, text="Umbral de coincidencia:").pack(side="left")
        self.var_score = tk.DoubleVar(value=self.tracker.min_score)
        self.lbl_score_val = ttk.Label(f_th, text=f"{self.tracker.min_score:.2f}", width=5)
        self.lbl_score_val.pack(side="right")
        self.scale_score = ttk.Scale(
            f_th,
            from_=0.10,
            to=0.95,
            variable=self.var_score,
            command=self._on_score_slider,
        )
        self.scale_score.pack(side="right", fill="x", expand=True, padx=4)

        # Status & Quality Indicator
        self.lbl_status = ttk.Label(
            self, text="Estado: Esperando captura de plantilla", font=("sans-serif", 9, "bold")
        )
        self.lbl_status.pack(fill="x", padx=10, pady=(6, 4))

        # Action Buttons
        btn_box = ttk.Frame(self)
        btn_box.pack(fill="x", padx=10, pady=10)

        self.btn_step = ttk.Button(btn_box, text="Paso a Paso", command=self.on_step_search)
        self.btn_step.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_run = ttk.Button(btn_box, text="▶ Rastreo Continuo", command=self._toggle_run)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(4, 0))

    def set_status(self, text: str, color: str = "#000000") -> None:
        """Updates dialog status message."""
        self.lbl_status.configure(text=f"Estado: {text}")

    def update_running_state(self, is_running: bool) -> None:
        """Updates run button text."""
        self.is_running = is_running
        if is_running:
            self.btn_run.configure(text="⏹ Detener")
            self.btn_step.configure(state="disabled")
        else:
            self.btn_run.configure(text="▶ Rastreo Continuo")
            self.btn_step.configure(state="normal")

    def _toggle_run(self) -> None:
        if self.is_running:
            self.on_stop_search()
        else:
            self._apply_settings()
            self.on_run_search()

    def _on_score_slider(self, val_str: str) -> None:
        try:
            val = float(val_str)
            self.lbl_score_val.configure(text=f"{val:.2f}")
            self.tracker.min_score = val
        except ValueError:
            pass

    def _apply_settings(self, event: tk.Event | None = None) -> None:
        try:
            tw = int(self.spin_tw.get())
            sw = int(self.spin_sw.get())
            self.tracker.template_size = (tw, tw)
            self.tracker.search_size = (sw, sw)
            self.tracker.min_score = self.var_score.get()
        except ValueError:
            pass
