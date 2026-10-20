"""Automated high-precision water level meniscus tracking for Torricelli experiment."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def track_water_level(
    video_path: str | Path = "/home/dersu/Downloads/20261007_082021.mp4",
    output_dir: str | Path = "/home/dersu/Code/Python/light_tracker/analisis_torricelli",
    sample_interval_sec: float = 1.0,
) -> Path:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"Video cargado: {duration:.2f} s, {total_frames} frames a {fps:.2f} FPS.")

    # Geometric calibration:
    # Crop in full 4K frame: [900:2700, 200:600]
    # Scale: 2500 pixels per meter (25.0 px/cm)
    scale_px_per_m = 2500.0

    # Orifice nozzle y-coordinate inside the crop: y = 1558.0 px
    y_orifice = 1558.0

    # Start meniscus at t=0 s: y ~ 809.2 px
    curr_y = 809.2

    timestamps: list[float] = []
    y_pixels_crop: list[float] = []
    y_pixels_full: list[float] = []
    heights_m: list[float] = []
    heights_cm: list[float] = []

    # Step in seconds
    num_steps = int(duration / sample_interval_sec)

    print(f"Rastreando nivel de agua en {num_steps} instantes de tiempo...")

    for step_idx in range(num_steps):
        t = step_idx * sample_interval_sec
        f_idx = min(total_frames - 1, round(t * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        crop = frame[900:2700, 200:600]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Clear water column to the left of the yellow tape: x in [90:170]
        col = gray[:, 90:170].astype(np.float32)
        prof = np.mean(col, axis=1)

        # Constrained search window: meniscus can only stay or drop downwards
        win_min = max(0, int(curr_y - 6))
        win_max = min(len(prof) - 2, int(curr_y + 20))

        # Sobel/gradient in y
        grad = np.abs(prof[win_min + 1 : win_max + 1] - prof[win_min - 1 : win_max - 1])
        local_idx = int(np.argmax(grad))
        best_y = float(win_min + local_idx)

        # Subpixel parabolic interpolation
        if 0 < local_idx < len(grad) - 1:
            denom = 2.0 * grad[local_idx] - grad[local_idx - 1] - grad[local_idx + 1]
            if abs(denom) > 1e-4:
                sub = (grad[local_idx + 1] - grad[local_idx - 1]) / (2.0 * denom)
                best_y += float(np.clip(sub, -0.5, 0.5))

        # Update if valid
        if best_y >= curr_y - 3.0:
            curr_y = best_y

        # Height above orifice in meters
        h_m = max(0.0, (y_orifice - curr_y) / scale_px_per_m)
        h_cm = h_m * 100.0

        timestamps.append(t)
        y_pixels_crop.append(curr_y)
        y_pixels_full.append(900.0 + curr_y)
        heights_m.append(h_m)
        heights_cm.append(h_cm)

        if step_idx % 30 == 0:
            print(f"  t = {t:5.1f} s | Nivel = {h_cm:5.2f} cm | y_px = {curr_y:.1f}")

        # Stop if water has reached the orifice
        if curr_y >= y_orifice - 5.0:
            print("  El nivel de agua alcanzó el orificio de salida.")
            break

    cap.release()

    t_arr = np.array(timestamps)
    h_arr = np.array(heights_m)
    h_cm_arr = np.array(heights_cm)
    sqrt_h = np.sqrt(h_arr)

    # Torricelli Linear Fit: sqrt(h) = sqrt(h0) - C * t
    slope, intercept = np.polyfit(t_arr, sqrt_h, 1)
    fit_sqrt_h = intercept + slope * t_arr
    fit_h = fit_sqrt_h**2

    # R^2 correlation coefficient
    ss_res = np.sum((sqrt_h - fit_sqrt_h) ** 2)
    ss_tot = np.sum((sqrt_h - np.mean(sqrt_h)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot)

    print("\n--- Resultados del Modelo de Torricelli ---")
    print(f"Altura inicial: h0 = {(intercept**2)*100:.2f} cm")
    print(f"Coeficiente C: {-slope:.5e} s^-1")
    print(f"Coeficiente de determinación R² = {r_squared:.4f}")

    # Save CSV
    csv_path = out_dir / "nivel_agua_torricelli.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_s,y_crop_px,y_full_px,altura_m,altura_cm,sqrt_h_m05\n")
        f.writelines(f"{t_arr[i]:.2f},{y_pixels_crop[i]:.2f},{y_pixels_full[i]:.2f},{h_arr[i]:.5f},{h_cm_arr[i]:.3f},{sqrt_h[i]:.5f}\n" for i in range(len(t_arr)))
    print(f"Datos exportados a: {csv_path}")

    # Generate Scientific Plot
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Plot 1: h(t) vs t
    ax1.scatter(t_arr, h_cm_arr, s=12, color="#0284C7", label="Datos experimentales (Menisco)", alpha=0.8)
    ax1.plot(t_arr, fit_h * 100.0, color="#DC2626", linewidth=2, label=f"Ajuste Torricelli (R² = {r_squared:.4f})")
    ax1.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax1.set_ylabel("Altura de la columna de agua $h(t)$ [cm]", fontsize=11)
    ax1.set_title("Vaciado del Tanque: $h(t)$ vs $t$", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Plot 2: sqrt(h) vs t (Linearization)
    ax2.scatter(t_arr, sqrt_h, s=12, color="#059669", label="$\\sqrt{h}$ experimental", alpha=0.8)
    ax2.plot(t_arr, fit_sqrt_h, color="#7C3AED", linewidth=2, label=f"Regresión lineal: $\\sqrt{{h}} = {intercept:.3f} - {-slope:.4f}t$")
    ax2.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax2.set_ylabel("$\\sqrt{h(t)}$ [m$^{1/2}$]", fontsize=11)
    ax2.set_title("Linealización de la Ley de Torricelli: $\\sqrt{h}$ vs $t$", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plot_path = out_dir / "torricelli_analisis.png"
    plt.savefig(str(plot_path), bbox_inches="tight")
    plt.close()
    print(f"Gráfica guardada en: {plot_path}")

    return csv_path


if __name__ == "__main__":
    track_water_level()
