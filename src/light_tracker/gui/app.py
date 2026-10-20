"""Main application window for LightTracker."""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2

from light_tracker.core.coordinate import CoordinateSystem
from light_tracker.core.kinematics import KinematicsEngine
from light_tracker.core.models import (
    CalibrationStick,
    Point2D,
    ProjectData,
    Track,
    TrackPoint,
)
from light_tracker.core.video import VideoReader
from light_tracker.gui.autotracker_dialog import AutoTrackerDialog
from light_tracker.gui.controls import PlaybackControls
from light_tracker.gui.data_table import DataTablePanel
from light_tracker.gui.plot_view import PlotPanel
from light_tracker.gui.video_canvas import VideoCanvas
from light_tracker.io.export import export_to_csv, export_to_latex
from light_tracker.io.project import load_project, save_project
from light_tracker.tracking.color_tracker import ColorTracker
from light_tracker.tracking.manual_tracker import ManualTracker
from light_tracker.tracking.template_tracker import TemplateTracker


class LightTrackerApp:
    """LightTracker main application window and coordinator."""

    def __init__(self, root: tk.Tk, video_path: str | None = None) -> None:
        self.root = root
        self.root.title("LightTracker - Video-Análisis y Seguimiento Cinemático")
        self.root.geometry("1180x780")
        self.root.minsize(800, 600)

        # Domain State
        self.video: VideoReader | None = None
        self.coord_sys = CoordinateSystem()
        self.calibration_stick: CalibrationStick | None = None
        self.tracks: list[Track] = [Track(id="track-0", name="Masa A", color="#E63946")]
        self.active_track: Track = self.tracks[0]
        self.current_frame: int = 0
        self.step_size: int = 1
        self.smooth_kinematics: bool = False

        # Interaction Mode
        self.interaction_mode: str = "none"  # "none", "manual_track", "set_stick", "set_origin"
        self._stick_point_buffer: list[Point2D] = []

        # Tracking Engines
        self.manual_tracker = ManualTracker(step_size=self.step_size)
        self.template_tracker = TemplateTracker()
        self.color_tracker = ColorTracker()
        self.autotracker_dialog: AutoTrackerDialog | None = None
        self._autotracking_active: bool = False

        # Playback animation state
        self._play_job: str | None = None

        self._apply_theme()
        self._build_menu()
        self._build_layout()

        if video_path:
            self.load_video(video_path)

    def _apply_theme(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        # File menu
        menu_file = tk.Menu(menubar, tearoff=0)
        menu_file.add_command(
            label="Abrir Video...", accelerator="Ctrl+O", command=self.on_open_video
        )
        menu_file.add_command(label="Cargar Proyecto (.trk.json)...", command=self.on_load_project)
        menu_file.add_command(
            label="Guardar Proyecto (.trk.json)...",
            accelerator="Ctrl+S",
            command=self.on_save_project,
        )
        menu_file.add_separator()
        menu_file.add_command(label="Exportar a CSV...", command=self.on_export_csv)
        menu_file.add_command(label="Exportar Tabla LaTeX...", command=self.on_export_latex)
        menu_file.add_separator()
        menu_file.add_command(label="Salir", command=self.root.quit)
        menubar.add_cascade(label="Archivo", menu=menu_file)

        # Calibration menu
        menu_calib = tk.Menu(menubar, tearoff=0)
        menu_calib.add_command(
            label="Definir Vara de Calibración (2 clics)", command=self.on_set_stick_mode
        )
        menu_calib.add_command(
            label="Fijar Origen de Coordenadas (1 clic)", command=self.on_set_origin_mode
        )
        menu_calib.add_command(
            label="Configurar Ángulo de Ejes...", command=self.on_set_angle_dialog
        )
        menu_calib.add_separator()
        self.var_show_axes = tk.BooleanVar(value=True)
        menu_calib.add_checkbutton(
            label="Mostrar Ejes", variable=self.var_show_axes, command=self._toggle_axes_vis
        )
        self.var_show_stick = tk.BooleanVar(value=True)
        menu_calib.add_checkbutton(
            label="Mostrar Vara", variable=self.var_show_stick, command=self._toggle_stick_vis
        )
        menubar.add_cascade(label="Calibración", menu=menu_calib)

        # Tracking menu
        menu_track = tk.Menu(menubar, tearoff=0)
        menu_track.add_command(
            label="Modo Rastreo Manual (Clic y avanzar)",
            accelerator="M",
            command=self.on_manual_track_mode,
        )
        menu_track.add_command(
            label="Abrir Asistente AutoTracker (NCC)",
            accelerator="A",
            command=self.on_open_autotracker,
        )
        menu_track.add_separator()
        menu_track.add_command(label="Nueva Masa Puntual...", command=self.on_new_track)
        menu_track.add_command(
            label="Limpiar Puntos de Masa Actual", command=self.on_clear_active_track
        )
        menubar.add_cascade(label="Rastreo", menu=menu_track)

        # Kinematics menu
        menu_kin = tk.Menu(menubar, tearoff=0)
        self.var_smooth = tk.BooleanVar(value=False)
        menu_kin.add_checkbutton(
            label="Suavizado Savitzky-Golay (Filtro de Ruido)",
            variable=self.var_smooth,
            command=self._toggle_smooth,
        )
        menubar.add_cascade(label="Cinemática", menu=menu_kin)

        # Help menu
        menu_help = tk.Menu(menubar, tearoff=0)
        menu_help.add_command(label="Acerca de LightTracker", command=self.on_about)
        menubar.add_cascade(label="Ayuda", menu=menu_help)

        self.root.config(menu=menubar)

        # Shortcuts
        self.root.bind("<Control-o>", lambda e: self.on_open_video())
        self.root.bind("<Control-s>", lambda e: self.on_save_project())
        self.root.bind("<space>", lambda e: self.toggle_playback())
        self.root.bind("<Right>", lambda e: self.step_frame(self.step_size))
        self.root.bind("<Left>", lambda e: self.step_frame(-self.step_size))
        self.root.bind("m", lambda e: self.on_manual_track_mode())
        self.root.bind("a", lambda e: self.on_open_autotracker())

    def _build_layout(self) -> None:
        # Main PanedWindow: Left Video & Controls; Right Data & Plots
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left Container
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        # Status Bar above canvas
        self.lbl_mode_status = ttk.Label(
            left_frame,
            text="Modo: Normal | Selecciona 'Rastreo > Modo Rastreo Manual' o usa 'Calibración'",
            relief="sunken",
            anchor="w",
            padding=(6, 4),
        )
        self.lbl_mode_status.pack(fill="x")

        # Video Canvas
        self.canvas = VideoCanvas(left_frame)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.on_click_canvas = self._on_canvas_click

        # Playback Controls
        self.controls = PlaybackControls(left_frame)
        self.controls.pack(fill="x")
        self.controls.on_seek = self.seek_frame
        self.controls.on_step = self.step_frame
        self.controls.on_play_toggle = self.toggle_playback
        self.controls.on_step_size_changed = self._on_step_size_changed

        # Right Container: Tabs for Data Table and Kinematic Plots
        right_frame = ttk.Frame(paned, width=420)
        paned.add(right_frame, weight=2)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill="both", expand=True)

        # Tab 1: DataTablePanel
        self.data_table = DataTablePanel(notebook)
        notebook.add(self.data_table, text="Tabla de Datos")
        self.data_table.on_select_frame = self.seek_frame
        self.data_table.on_delete_point = self.delete_track_point

        # Tab 2: PlotPanel
        self.plot_panel = PlotPanel(notebook)
        notebook.add(self.plot_panel, text="Gráficas Cinemáticas")

    # --- Video & Playback Actions ---

    def on_open_video(self) -> None:
        filetypes = [
            ("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Todos los archivos", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Seleccionar Video", filetypes=filetypes)
        if path:
            self.load_video(path)

    def load_video(self, video_path: str) -> None:
        try:
            if self.video:
                self.video.close()
            self.video = VideoReader(video_path)
            self.current_frame = 0
            self.controls.configure_video(self.video.total_frames, self.video.fps)
            self.coord_sys.origin = Point2D(self.video.width / 2.0, self.video.height / 2.0)
            self._update_display()
        except (RuntimeError, FileNotFoundError, cv2.error) as e:
            messagebox.showerror("Error al abrir video", str(e))

    def seek_frame(self, frame_idx: int) -> None:
        if not self.video:
            return
        self.current_frame = max(0, min(self.video.total_frames - 1, frame_idx))
        self.controls.set_frame_index(self.current_frame)
        self._update_display()

    def step_frame(self, delta: int) -> None:
        if not self.video:
            return
        new_f = self.current_frame + delta
        self.seek_frame(new_f)

    def toggle_playback(self) -> None:
        if not self.video:
            return
        if self.controls.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if not self.video:
            return
        self.controls.set_playing_state(True)
        self._playback_loop()

    def _stop_playback(self) -> None:
        self.controls.set_playing_state(False)
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _playback_loop(self) -> None:
        if not self.controls.is_playing or not self.video:
            return

        if self.current_frame + self.step_size >= self.video.total_frames:
            self._stop_playback()
            return

        self.step_frame(self.step_size)
        delay_ms = max(10, int(1000.0 / self.video.fps * self.step_size))
        self._play_job = self.root.after(delay_ms, self._playback_loop)

    def _on_step_size_changed(self, step: int) -> None:
        self.step_size = step
        self.manual_tracker.step_size = step

    # --- Mode & Canvas Interactions ---

    def on_set_stick_mode(self) -> None:
        self.interaction_mode = "set_stick"
        self._stick_point_buffer.clear()
        self.lbl_mode_status.configure(
            text="Modo: Calibración Vara -> Haz clic en el PRIMER extremo de la vara."
        )

    def on_set_origin_mode(self) -> None:
        self.interaction_mode = "set_origin"
        self.lbl_mode_status.configure(
            text="Modo: Calibración Origen -> Haz clic en el punto del origen (0,0)."
        )

    def on_manual_track_mode(self) -> None:
        self.interaction_mode = "manual_track"
        self.lbl_mode_status.configure(
            text="Modo: Rastreo Manual -> Haz clic en el objeto para marcar y avanzar."
        )

    def _on_canvas_click(self, p: Point2D) -> None:
        if not self.video:
            return

        if self.interaction_mode == "set_origin":
            self.coord_sys.origin = p
            self.interaction_mode = "none"
            self.lbl_mode_status.configure(text="Modo: Normal | Origen fijado exitosamente.")
            self._recalculate_kinematics()
            self._update_display()

        elif self.interaction_mode == "set_stick":
            self._stick_point_buffer.append(p)
            if len(self._stick_point_buffer) == 1:
                self.lbl_mode_status.configure(
                    text="Modo: Calibración Vara -> Haz clic en el SEGUNDO extremo de la vara."
                )
            elif len(self._stick_point_buffer) == 2:
                p1, p2 = self._stick_point_buffer
                dist_px = p1.distance_to(p2)
                length_str = simpledialog.askstring(
                    "Longitud Real de la Vara",
                    f"Distancia en píxeles: {dist_px:.1f} px\nIngresa la longitud real en metros:",
                    initialvalue="1.0",
                )
                if length_str:
                    try:
                        val = float(length_str)
                        if val > 0:
                            self.calibration_stick = CalibrationStick(
                                p1=p1, p2=p2, real_length=val, unit="m"
                            )
                            self.coord_sys.set_calibration_stick(self.calibration_stick)
                            messagebox.showinfo(
                                "Calibración completada",
                                f"Escala: {self.coord_sys.scale_px_per_unit:.2f} px/m",
                            )
                    except ValueError:
                        messagebox.showerror("Error", "Longitud numérica inválida.")
                self.interaction_mode = "none"
                self._stick_point_buffer.clear()
                self.lbl_mode_status.configure(text="Modo: Normal | Calibración guardada.")
                self._recalculate_kinematics()
                self._update_display()

        elif self.interaction_mode == "manual_track":
            # Add point to active track
            self.manual_tracker.mark_point(
                track=self.active_track,
                frame_idx=self.current_frame,
                fps=self.video.fps,
                pixel_pos=p,
                coord_sys=self.coord_sys,
            )
            self._recalculate_kinematics()
            self._update_display()
            # Advance frame
            if self.current_frame + self.step_size < self.video.total_frames:
                self.step_frame(self.step_size)

        elif self.interaction_mode == "select_template":
            # Autotracker capture center
            frame = self.video.get_frame(self.current_frame)
            if frame is not None:
                self.template_tracker.initialize(frame, p)
                self.canvas.template_rect = (
                    p.x,
                    p.y,
                    self.template_tracker.template_size[0],
                    self.template_tracker.template_size[1],
                )
                self.canvas.search_rect = (
                    p.x,
                    p.y,
                    self.template_tracker.search_size[0],
                    self.template_tracker.search_size[1],
                )
                if self.autotracker_dialog:
                    self.autotracker_dialog.set_status("Plantilla capturada. Listo para rastrear.")
                self.interaction_mode = "none"
                self.lbl_mode_status.configure(
                    text="Modo: Normal | Plantilla de AutoTracker capturada."
                )
                self._update_display()

    # --- AutoTracker Actions ---

    def on_open_autotracker(self) -> None:
        if not self.video:
            messagebox.showwarning("Atención", "Carga un video primero.")
            return

        if self.autotracker_dialog is None or not self.autotracker_dialog.winfo_exists():
            self.autotracker_dialog = AutoTrackerDialog(
                master=self.root,
                tracker=self.template_tracker,
                on_capture_template=self._on_at_capture,
                on_step_search=self._on_at_step,
                on_run_search=self._on_at_run,
                on_stop_search=self._on_at_stop,
            )
        self.autotracker_dialog.deiconify()
        self.autotracker_dialog.lift()

    def _on_at_capture(self) -> None:
        self.interaction_mode = "select_template"
        self.lbl_mode_status.configure(
            text="Modo: AutoTracker -> Haz clic en el centro del objeto a seguir."
        )

    def _on_at_step(self) -> bool:
        if not self.video or not self.template_tracker.is_initialized:
            messagebox.showwarning("Atención", "Captura una plantilla antes de rastrear.")
            return False

        if self.current_frame + self.step_size >= self.video.total_frames:
            if self.autotracker_dialog:
                self.autotracker_dialog.set_status("Fin del video alcanzado.")
            return False

        next_frame_idx = self.current_frame + self.step_size
        next_frame = self.video.get_frame(next_frame_idx)
        if next_frame is None:
            return False

        # Lookahead velocity vector
        vel_est = None
        pts = self.active_track.get_sorted_points()
        if len(pts) >= 2:
            dx = pts[-1].pixel_pos.x - pts[-2].pixel_pos.x
            dy = pts[-1].pixel_pos.y - pts[-2].pixel_pos.y
            vel_est = Point2D(dx, dy)

        pos, score = self.template_tracker.track_frame(
            frame=next_frame,
            prev_pos=self.active_track.points.get(
                self.current_frame, pts[-1] if pts else None
            ).pixel_pos
            if (pts or self.current_frame in self.active_track.points)
            else None,
            velocity_estimate=vel_est,
        )

        if pos is not None and score >= self.template_tracker.min_score:
            # Advance and record
            self.seek_frame(next_frame_idx)
            world_pos = self.coord_sys.pixel_to_world(pos)
            point = TrackPoint(
                frame_idx=self.current_frame,
                t=self.current_frame / self.video.fps,
                pixel_pos=pos,
                world_pos=world_pos,
                score=score,
            )
            self.active_track.add_point(point)

            # Update boxes
            tw, th = self.template_tracker.template_size
            sw, sh = self.template_tracker.search_size
            self.canvas.template_rect = (pos.x, pos.y, tw, th)
            self.canvas.search_rect = (pos.x, pos.y, sw, sh)

            self._recalculate_kinematics()
            self._update_display()
            if self.autotracker_dialog:
                self.autotracker_dialog.set_status(
                    f"Match OK: {score:.2f} en Frame {self.current_frame}"
                )
            return True
        else:
            if self.autotracker_dialog:
                score_str = f"{score:.2f}" if score > 0 else "0.00"
                self.autotracker_dialog.set_status(
                    f"Perdido en Frame {next_frame_idx} (Score {score_str} < Umbral)"
                )
            return False

    def _on_at_run(self) -> None:
        self._autotracking_active = True
        if self.autotracker_dialog:
            self.autotracker_dialog.update_running_state(True)
        self._autotracker_loop()

    def _on_at_stop(self) -> None:
        self._autotracking_active = False
        if self.autotracker_dialog:
            self.autotracker_dialog.update_running_state(False)

    def _autotracker_loop(self) -> None:
        if not self._autotracking_active:
            return

        success = self._on_at_step()
        if (
            success
            and self._autotracking_active
            and self.current_frame + self.step_size < (self.video.total_frames if self.video else 0)
        ):
            self.root.after(20, self._autotracker_loop)
        else:
            self._on_at_stop()

    # --- Project & Track Management ---

    def on_new_track(self) -> None:
        name = simpledialog.askstring(
            "Nueva Masa Puntual",
            "Nombre de la masa:",
            initialvalue=f"Masa {chr(65 + len(self.tracks))}",
        )
        if name:
            new_t = Track(id=f"track-{len(self.tracks)}", name=name, color="#3B82F6")
            self.tracks.append(new_t)
            self.active_track = new_t
            self._update_display()

    def on_clear_active_track(self) -> None:
        if messagebox.askyesno(
            "Confirmar", f"¿Eliminar todos los puntos de {self.active_track.name}?"
        ):
            self.active_track.clear()
            self._recalculate_kinematics()
            self._update_display()

    def delete_track_point(self, frame_idx: int) -> None:
        self.active_track.remove_point(frame_idx)
        self._recalculate_kinematics()
        self._update_display()

    def _recalculate_kinematics(self) -> None:
        if not self.video:
            return

        # Re-transform world positions with current coordinate system
        for pt in self.active_track.points.values():
            pt.world_pos = self.coord_sys.pixel_to_world(pt.pixel_pos)

        pts = self.active_track.get_sorted_points()
        computed = KinematicsEngine.compute(
            pts,
            smooth=self.smooth_kinematics,
            window_length=5,
            polyorder=2,
        )
        for p in computed:
            self.active_track.points[p.frame_idx] = p

    def _update_display(self) -> None:
        frame_rgb = self.video.get_frame(self.current_frame) if self.video else None
        self.canvas.set_frame(frame_rgb)
        self.canvas.redraw(
            coord_sys=self.coord_sys,
            stick=self.calibration_stick,
            tracks=self.tracks,
            current_frame_idx=self.current_frame,
        )
        self.data_table.update_data(self.active_track, highlight_frame=self.current_frame)
        self.plot_panel.update_data(self.active_track, current_frame=self.current_frame)

    def on_set_angle_dialog(self) -> None:
        val_str = simpledialog.askstring(
            "Ángulo de Coordenadas",
            f"Ángulo actual: {self.coord_sys.angle_deg:.1f}°\nIngresa el nuevo ángulo en grados (-180° a 180°):",
            initialvalue=str(self.coord_sys.angle_deg),
        )
        if val_str:
            try:
                angle = float(val_str)
                self.coord_sys.angle_deg = angle
                self._recalculate_kinematics()
                self._update_display()
            except ValueError:
                messagebox.showerror("Error", "Ángulo numérico inválido.")

    def _toggle_axes_vis(self) -> None:
        self.canvas.show_axes = self.var_show_axes.get()
        self.canvas.redraw(self.coord_sys, self.calibration_stick, self.tracks, self.current_frame)

    def _toggle_stick_vis(self) -> None:
        self.canvas.show_stick = self.var_show_stick.get()
        self.canvas.redraw(self.coord_sys, self.calibration_stick, self.tracks, self.current_frame)

    def _toggle_smooth(self) -> None:
        self.smooth_kinematics = self.var_smooth.get()
        self._recalculate_kinematics()
        self._update_display()

    # --- Project IO & Export ---

    def on_save_project(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Guardar Proyecto",
            defaultextension=".trk.json",
            filetypes=[("Proyectos LightTracker", "*.trk.json"), ("Archivos JSON", "*.json")],
        )
        if not path:
            return

        proj = ProjectData(
            video_path=str(self.video.path) if self.video else "",
            fps=self.video.fps if self.video else 30.0,
            total_frames=self.video.total_frames if self.video else 0,
            start_frame=0,
            end_frame=self.video.total_frames - 1 if self.video else 0,
            step_size=self.step_size,
            calibration=self.calibration_stick,
            coordinate_origin=self.coord_sys.to_origin_model(),
            tracks=self.tracks,
        )
        save_project(proj, path)
        messagebox.showinfo("Guardado", f"Proyecto guardado en:\n{path}")

    def on_load_project(self) -> None:
        path = filedialog.askopenfilename(
            title="Cargar Proyecto",
            filetypes=[("Proyectos LightTracker", "*.trk.json"), ("Archivos JSON", "*.json")],
        )
        if not path:
            return

        try:
            proj = load_project(path)
            if proj.video_path and Path(proj.video_path).exists():
                self.load_video(proj.video_path)

            self.coord_sys.origin = proj.coordinate_origin.origin
            self.coord_sys.angle_deg = proj.coordinate_origin.angle_deg
            self.calibration_stick = proj.calibration
            if self.calibration_stick:
                self.coord_sys.set_calibration_stick(self.calibration_stick)

            self.step_size = proj.step_size
            self.tracks = proj.tracks if proj.tracks else [Track(id="track-0", name="Masa A")]
            self.active_track = self.tracks[0]

            self._recalculate_kinematics()
            self._update_display()
            messagebox.showinfo("Cargado", "Proyecto cargado correctamente.")
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            messagebox.showerror("Error al cargar proyecto", str(e))

    def on_export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Exportar Datos a CSV",
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv")],
        )
        if path:
            export_to_csv(self.active_track, path)
            messagebox.showinfo("Exportado", f"Datos exportados a:\n{path}")

    def on_export_latex(self) -> None:
        code = export_to_latex(self.active_track)
        win = tk.Toplevel(self.root)
        win.title("Código Tabla LaTeX")
        win.geometry("600x400")

        text = tk.Text(win, wrap="none", font=("monospace", 10))
        text.insert("1.0", code)
        text.pack(fill="both", expand=True, padx=8, pady=8)

        btn_copy = ttk.Button(
            win,
            text="Copiar al portapapeles",
            command=lambda: (
                win.clipboard_clear(),
                win.clipboard_append(code),
                messagebox.showinfo("Copiado", "Tabla LaTeX copiada."),
            ),
        )
        btn_copy.pack(pady=(0, 8))

    def on_about(self) -> None:
        msg = (
            "LightTracker v0.1.0\n\n"
            "Sistema ligero de video-análisis y tracking cinemático para física.\n"
            "Desarrollado en Python 3.12 y OpenCV con aceleración C++.\n\n"
            "Alternativa ultraligera y moderna a Tracker (Open Source Physics)."
        )
        messagebox.showinfo("Acerca de LightTracker", msg)


def main() -> None:
    """Entry point for the GUI application."""
    root = tk.Tk()
    video_arg = sys.argv[1] if len(sys.argv) > 1 else None
    LightTrackerApp(root, video_path=video_arg)
    root.mainloop()


if __name__ == "__main__":
    main()
