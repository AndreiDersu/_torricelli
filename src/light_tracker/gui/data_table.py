"""Data table view for kinematic measurements."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from light_tracker.core.models import Track


class DataTablePanel(ttk.Frame):
    """Tabular presentation of kinematic measurements with selection and clipboard support."""

    def __init__(self, master: tk.Widget, **kwargs: object) -> None:
        super().__init__(master, **kwargs)

        self.on_select_frame: Callable[[int], None] | None = None
        self.on_delete_point: Callable[[int], None] | None = None
        self._current_track: Track | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        columns = ("frame", "t", "x", "y", "vx", "vy", "v", "ax", "ay", "score")

        self.tree = ttk.Treeview(self, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("frame", text="Frame")
        self.tree.heading("t", text="t [s]")
        self.tree.heading("x", text="x [m]")
        self.tree.heading("y", text="y [m]")
        self.tree.heading("vx", text="vx [m/s]")
        self.tree.heading("vy", text="vy [m/s]")
        self.tree.heading("v", text="v [m/s]")
        self.tree.heading("ax", text="ax [m/s²]")
        self.tree.heading("ay", text="ay [m/s²]")
        self.tree.heading("score", text="Score")

        widths = {
            "frame": 55,
            "t": 65,
            "x": 75,
            "y": 75,
            "vx": 75,
            "vy": 75,
            "v": 75,
            "ax": 75,
            "ay": 75,
            "score": 60,
        }

        for col, w in widths.items():
            self.tree.column(col, width=w, anchor="e" if col != "frame" else "center")

        # Scrollbars
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_row_selected)
        self.tree.bind("<Button-3>", self._show_context_menu)

        # Context menu
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Copiar datos al portapapeles", command=self.copy_to_clipboard)
        self.menu.add_command(label="Eliminar punto", command=self._delete_selected_point)

    def update_data(self, track: Track | None, highlight_frame: int | None = None) -> None:
        """Refreshes table rows with the points from the specified track."""
        self._current_track = track

        # Clear existing rows
        for item in self.tree.get_children():
            self.tree.delete(item)

        if track is None:
            return

        points = track.get_sorted_points()
        for p in points:
            values = (
                p.frame_idx,
                f"{p.t:.3f}",
                f"{p.world_pos.x:.4f}",
                f"{p.world_pos.y:.4f}",
                f"{p.vx:.4f}",
                f"{p.vy:.4f}",
                f"{p.v:.4f}",
                f"{p.ax:.4f}",
                f"{p.ay:.4f}",
                f"{p.score:.2f}",
            )
            item_id = str(p.frame_idx)
            self.tree.insert("", "end", iid=item_id, values=values)

        if highlight_frame is not None and str(highlight_frame) in self.tree.get_children():
            self.tree.selection_set(str(highlight_frame))
            self.tree.see(str(highlight_frame))

    def copy_to_clipboard(self) -> None:
        """Copies all table contents as tab-separated values to system clipboard."""
        if self._current_track is None:
            return

        points = self._current_track.get_sorted_points()
        if not points:
            return

        headers = [
            "frame",
            "t_s",
            "x_m",
            "y_m",
            "vx_mps",
            "vy_mps",
            "v_mps",
            "ax_mps2",
            "ay_mps2",
            "score",
        ]
        lines = ["\t".join(headers)]
        for p in points:
            row = [
                str(p.frame_idx),
                f"{p.t:.4f}",
                f"{p.world_pos.x:.6f}",
                f"{p.world_pos.y:.6f}",
                f"{p.vx:.6f}",
                f"{p.vy:.6f}",
                f"{p.v:.6f}",
                f"{p.ax:.6f}",
                f"{p.ay:.6f}",
                f"{p.score:.3f}",
            ]
            lines.append("\t".join(row))

        text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Portapapeles", "Datos copiados al portapapeles exitosamente.")

    def _on_row_selected(self, event: tk.Event) -> None:
        sel = self.tree.selection()
        if sel and self.on_select_frame:
            frame_idx = int(sel[0])
            self.on_select_frame(frame_idx)

    def _show_context_menu(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.menu.tk_popup(event.x_root, event.y_root)

    def _delete_selected_point(self) -> None:
        sel = self.tree.selection()
        if sel and self.on_delete_point:
            frame_idx = int(sel[0])
            self.on_delete_point(frame_idx)
