"""Complete analysis of the 4 Torricelli-Bernoulli videos in Data/ at 1080x1920 and 60 fps.

Tracks:
1. Meniscus position in the water tube referenced to the 60 cm yellow ruler
   (where position 254 on the ruler = 54 cm from the top of the ruler).
2. Water jet trajectory, horizontal reach X(t), and parabolic arc length L(t)
   ("longitud del chorro de agua en sí mismo"), ensuring smooth, jump-free curves.
Generates:
- Individual CSVs and a combined master CSV with all data.
- 3-position visual tracking comparison images (frames) for each of the 4 videos.
- Time-evolution plots of meniscus position and water jet length for each video
  and a comparative summary plot.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Scale in 1080x1920 resolution:
# The 60 cm yellow ruler (marks 200 cm to 260 cm) spans 750.0 pixels vertically:
# 750.0 px / 60.0 cm = 12.50 px/cm (1250.0 px/m).
SCALE_PX_PER_CM = 12.50
SCALE_PX_PER_M = 1250.0
FPS_NOMINAL = 60.0
GRAVITY = 9.80665
RHO_WATER = 1000.0
ORIFICE_DIAM_M = 0.00252
ORIFICE_AREA_M2 = math.pi * (ORIFICE_DIAM_M / 2.0) ** 2

VIDEO_CONFIGS: dict[str, dict] = {
    "20261007_082021": {
        "label": "Video 1 (082021)",
        "x_col": (135, 175),
        "y_top_ruler": 552.0,   # Mark 200 (0 cm from top of ruler)
        "y_pos_254": 1227.0,    # Mark 254 (54 cm from top of ruler, orifice level)
        "y_bot_ruler": 1302.0,  # Mark 260 (60 cm from top of ruler)
        "y_men_start": 854.0,
        "nozzle_xy": (255.0, 1227.0),
        "y_jet_ref": 1332.0,
        "t_end": 310,
        "three_times": [10, 150, 290],
    },
    "20261007_083231": {
        "label": "Video 2 (083231)",
        "x_col": (145, 185),
        "y_top_ruler": 558.0,
        "y_pos_254": 1233.0,
        "y_bot_ruler": 1308.0,
        "y_men_start": 854.0,
        "nozzle_xy": (250.0, 1233.0),
        "y_jet_ref": 1329.0,
        "t_end": 360,
        "three_times": [10, 175, 340],
    },
    "20261007_084607": {
        "label": "Video 3 (084607)",
        "x_col": (135, 175),
        "y_top_ruler": 561.0,
        "y_pos_254": 1236.0,
        "y_bot_ruler": 1311.0,
        "y_men_start": 904.0,
        "nozzle_xy": (242.0, 1236.0),
        "y_jet_ref": 1345.0,
        "t_end": 325,
        "three_times": [10, 160, 310],
    },
    "20261007_085723": {
        "label": "Video 4 (085723)",
        "x_col": (210, 250),
        "y_top_ruler": 555.0,
        "y_pos_254": 1230.0,
        "y_bot_ruler": 1305.0,
        "y_men_start": 733.0,
        "nozzle_xy": (312.0, 1230.0),
        "y_jet_ref": 1329.0,
        "t_end": 410,
        "three_times": [15, 200, 390],
    },
}


def parabolic_arc_length(dx: float, dy: float) -> float:
    """Compute exact arc length of parabola y(x) = (dy / dx^2) * x^2 for x in [0, dx]."""
    if dx <= 1e-6:
        return float(dy)
    k = (2.0 * dy) / dx
    sqrt_term = math.sqrt(1.0 + k * k)
    return 0.5 * dx * (sqrt_term + math.log(k + sqrt_term) / k)


def smooth_monotonic_increasing(t_sample: np.ndarray, y_sample: np.ndarray, t_dense: np.ndarray) -> np.ndarray:
    """Reject outliers, smooth with Savitzky-Golay, interpolate to 1s grid, and enforce non-decreasing."""
    y = y_sample.copy()
    poly_coeffs = np.polyfit(t_sample, y, 2)
    poly_trend = np.polyval(poly_coeffs, t_sample)
    residuals = y - poly_trend
    mad = np.median(np.abs(residuals - np.median(residuals))) + 1e-6
    outliers = np.abs(residuals) > 2.8 * mad
    y[outliers] = poly_trend[outliers]

    win = min(19, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if win >= 5:
        y = savgol_filter(y, window_length=win, polyorder=2)

    y_dense = np.interp(t_dense, t_sample, y)
    win_dense = min(31, len(y_dense) if len(y_dense) % 2 == 1 else len(y_dense) - 1)
    if win_dense >= 7:
        y_dense = savgol_filter(y_dense, window_length=win_dense, polyorder=2)

    for i in range(1, len(y_dense)):
        if y_dense[i] < y_dense[i - 1]:
            y_dense[i] = y_dense[i - 1]
    return y_dense


def smooth_monotonic_decreasing(t_sample: np.ndarray, x_sample: np.ndarray, t_dense: np.ndarray) -> np.ndarray:
    """Reject outliers, smooth with Savitzky-Golay, interpolate to 1s grid, and enforce non-increasing."""
    return -smooth_monotonic_increasing(t_sample, -x_sample, t_dense)


def process_single_video(
    video_stem: str,
    cfg: dict,
    data_dir: Path,
    out_dir: Path,
    sample_step_s: int = 3,
) -> pd.DataFrame:
    vpath = data_dir / f"{video_stem}.mp4"
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {vpath}")

    fps_actual = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_end = min(cfg["t_end"], int(total_frames / fps_actual) - 1)

    x0, x1 = cfg["x_col"]
    y_top = cfg["y_top_ruler"]
    y_254 = cfg["y_pos_254"]
    x_noz, y_noz = cfg["nozzle_xy"]
    y_ref = cfg["y_jet_ref"]
    dy_px = y_ref - y_noz
    dy_cm = dy_px / SCALE_PX_PER_CM
    dy_m = dy_px / SCALE_PX_PER_M
    t_flight = math.sqrt(2.0 * dy_m / GRAVITY)

    curr_y_men = float(cfg["y_men_start"])
    prev_dx_jet: float | None = None

    sample_times: list[float] = []
    raw_y_men: list[float] = []
    raw_dx_jet: list[float] = []

    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Build sample grid including exact 3 verification timestamps and endpoints
    sample_grid = sorted(
        set(range(0, t_end + 1, sample_step_s))
        | {t_end}
        | set(cfg["three_times"])
    )

    print(f"\nProcesando {video_stem} (0 a {t_end} s, resolución normalizada 1080x1920 @ 60 fps)...")

    prev_t = 0.0
    for sec in sample_grid:
        t = float(sec)
        dt = max(1.0, t - prev_t)
        prev_t = t

        f_idx_actual = min(total_frames - 1, round(t * fps_actual))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx_actual)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame.shape[1] != 1080 or frame.shape[0] != 1920:
            f1080 = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_AREA)
        else:
            f1080 = frame

        gray = cv2.cvtColor(f1080, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. Track meniscus inside bounded physical descent window (approx 0.3 to 2.8 px/s)
        max_drop = int(round(3.0 * dt)) + 3
        w_min = max(int(y_top), int(round(curr_y_men)))
        w_max = min(int(y_254), int(round(curr_y_men + max_drop)))
        if w_min < w_max - 2:
            col_slice = gray[w_min : w_max + 2, x0:x1]
            prof_y = np.mean(col_slice, axis=1)
            grad_y = np.abs(prof_y[2:] - prof_y[:-2])
            loc_idx = int(np.argmax(grad_y))
            best_y = float(w_min + 1 + loc_idx)
            if 0 < loc_idx < len(grad_y) - 1:
                denom = 2.0 * grad_y[loc_idx] - grad_y[loc_idx - 1] - grad_y[loc_idx + 1]
                if abs(denom) > 1e-4:
                    sub = (grad_y[loc_idx + 1] - grad_y[loc_idx - 1]) / (2.0 * denom)
                    best_y += float(np.clip(sub, -0.5, 0.5))
            curr_y_men = best_y

        # 2. Track water jet via parabolic-arc top-hat integral around physical prior
        h_px = max(1.0, y_254 - curr_y_men)
        dx_prior = 2.0 * 0.78 * math.sqrt(dy_px * h_px)
        center_dx = dx_prior if prev_dx_jet is None else (0.50 * prev_dx_jet + 0.50 * dx_prior)

        roi_y0 = max(0, int(y_noz - 10))
        roi_y1 = min(1920, int(y_ref + 15))
        roi_x0 = max(0, int(x_noz))
        roi_x1 = min(1080, int(x_noz + 420))
        jet_roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
        tophat_roi = cv2.morphologyEx(jet_roi, cv2.MORPH_TOPHAT, kernel_tophat)
        # Mask out the static scissor-jack plate at the bottom-left of the ROI
        tophat_roi[int(y_noz - roi_y0 + 25) :, :42] = 0.0

        cand_min = max(35.0, center_dx - 32.0)
        cand_max = min(390.0, center_dx + 32.0)
        cands = np.linspace(cand_min, cand_max, 65)

        best_dx = center_dx
        best_score = -1e9
        for dx_c in cands:
            xs_local = np.linspace(12.0, dx_c, 40)
            ys_local = (y_noz - roi_y0) + dy_px * (xs_local / dx_c) ** 2
            xi = np.clip(np.round(xs_local).astype(int), 0, tophat_roi.shape[1] - 1)
            yi = np.clip(np.round(ys_local).astype(int), 0, tophat_roi.shape[0] - 1)
            reg_penalty = 0.12 * abs(dx_c - dx_prior)
            score = float(np.mean(tophat_roi[yi, xi])) - reg_penalty
            if score > best_score:
                best_score = score
                best_dx = float(dx_c)

        prev_dx_jet = best_dx

        sample_times.append(t)
        raw_y_men.append(curr_y_men)
        raw_dx_jet.append(best_dx)

    cap.release()

    # Smooth & evaluate on dense 1-second grid (zero abrupt jumps!)
    t_sample_arr = np.array(sample_times, dtype=float)
    t_dense_arr = np.arange(0, t_end + 1, 1.0, dtype=float)
    y_men_smooth = smooth_monotonic_increasing(t_sample_arr, np.array(raw_y_men, dtype=float), t_dense_arr)
    dx_jet_smooth = smooth_monotonic_decreasing(t_sample_arr, np.array(raw_dx_jet, dtype=float), t_dense_arr)

    records: list[dict] = []
    for i, t in enumerate(t_dense_arr):
        f60 = round(t * FPS_NOMINAL)
        ym = float(y_men_smooth[i])
        dx_px = float(dx_jet_smooth[i])

        # Ruler coordinates (60 cm ruler from mark 200 to 260; mark 254 = 54 cm from top)
        dist_from_top_cm = (ym - y_top) / SCALE_PX_PER_CM
        ruler_mark_cm = 200.0 + dist_from_top_cm
        h_water_cm = max(0.0, 54.0 - dist_from_top_cm)
        h_water_m = h_water_cm / 100.0

        # Jet coordinates
        x_impact_px = x_noz + dx_px
        reach_X_cm = dx_px / SCALE_PX_PER_CM
        reach_X_m = dx_px / SCALE_PX_PER_M
        arc_len_px = parabolic_arc_length(dx_px, dy_px)
        arc_len_cm = arc_len_px / SCALE_PX_PER_CM

        v_exit_ms = reach_X_m / t_flight
        v_ideal_ms = math.sqrt(2.0 * GRAVITY * h_water_m)
        power_mW = 0.5 * RHO_WATER * ORIFICE_AREA_M2 * (v_exit_ms**3) * 1000.0

        records.append({
            "video": video_stem,
            "frame_60fps": f60,
            "tiempo_s": t,
            "y_menisco_px": round(ym, 2),
            "posicion_regla_cm": round(ruler_mark_cm, 3),
            "distancia_tope_regla_cm": round(dist_from_top_cm, 3),
            "altura_agua_h_cm": round(h_water_cm, 3),
            "x_impacto_chorro_px": round(x_impact_px, 2),
            "alcance_horizontal_chorro_cm": round(reach_X_cm, 3),
            "longitud_chorro_cm": round(arc_len_cm, 3),
            "caida_vertical_chorro_cm": round(dy_cm, 3),
            "velocidad_salida_ms": round(v_exit_ms, 4),
            "velocidad_ideal_torricelli_ms": round(v_ideal_ms, 4),
            "potencia_chorro_mW": round(power_mW, 4),
        })

    df = pd.DataFrame(records)
    csv_path = out_dir / f"datos_{video_stem}.csv"
    df.to_csv(csv_path, index=False)
    print(
        f"  -> CSV guardado: {csv_path.name} ({len(df)} filas) | "
        f"Menisco: {df['posicion_regla_cm'].iloc[0]:.1f} -> {df['posicion_regla_cm'].iloc[-1]:.1f} cm "
        f"(dist tope: {df['distancia_tope_regla_cm'].iloc[0]:.1f} -> {df['distancia_tope_regla_cm'].iloc[-1]:.1f} cm, "
        f"h: {df['altura_agua_h_cm'].iloc[0]:.1f} -> {df['altura_agua_h_cm'].iloc[-1]:.1f} cm) | "
        f"Longitud chorro: {df['longitud_chorro_cm'].iloc[0]:.2f} -> {df['longitud_chorro_cm'].iloc[-1]:.2f} cm"
    )

    # Generate 3-position visual comparison image (frames)
    generate_three_position_image(video_stem, cfg, df, fps_actual, data_dir, out_dir)

    # Generate individual time-evolution plot
    generate_video_plot(video_stem, cfg, df, out_dir)

    return df


def generate_three_position_image(
    video_stem: str,
    cfg: dict,
    df: pd.DataFrame,
    fps_actual: float,
    data_dir: Path,
    out_dir: Path,
) -> Path:
    vpath = data_dir / f"{video_stem}.mp4"
    cap = cv2.VideoCapture(str(vpath))
    x0, x1 = cfg["x_col"]
    y_top = cfg["y_top_ruler"]
    y_254 = cfg["y_pos_254"]
    y_bot = cfg["y_bot_ruler"]
    x_noz, y_noz = cfg["nozzle_xy"]
    y_ref = cfg["y_jet_ref"]
    dy_px = y_ref - y_noz

    panels: list[np.ndarray] = []
    stage_names = ["Posición 1 (Inicio)", "Posición 2 (Intermedia)", "Posición 3 (Final)"]

    for stage_label, t_target in zip(stage_names, cfg["three_times"]):
        row = df.iloc[(df["tiempo_s"] - t_target).abs().argsort()[:1]].iloc[0]
        t_val = float(row["tiempo_s"])
        f_idx = round(t_val * fps_actual)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        f1080 = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_AREA)
        vis = f1080.copy()

        ym = round(row["y_menisco_px"])
        x_imp = float(row["x_impacto_chorro_px"])
        dx_px = max(1.0, x_imp - x_noz)

        # 1. Draw 60 cm yellow ruler reference markers (Top=200/0cm, Orifice=254/54cm, Bottom=260/60cm)
        cv2.line(vis, (x0 - 30, int(y_top)), (x1 + 45, int(y_top)), (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            "Tope regla: 0 cm (Marca 200)",
            (x0 - 30, int(y_top) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.line(vis, (x0 - 30, int(y_254)), (x1 + 65, int(y_254)), (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            "Marca 254 (54 cm del tope)",
            (x0 - 30, int(y_254) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 180, 50),
            1,
            cv2.LINE_AA,
        )

        cv2.line(vis, (x0 - 30, int(y_bot)), (x1 + 45, int(y_bot)), (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            "Base regla: 60 cm (Marca 260)",
            (x0 - 30, int(y_bot) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )

        # 2. Draw tracked meniscus position
        cv2.line(vis, (x0 - 25, ym), (x1 + 85, ym), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(vis, (int((x0 + x1) / 2), ym), 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"Menisco: reg={row['posicion_regla_cm']:.1f}cm (d={row['distancia_tope_regla_cm']:.1f}cm, h={row['altura_agua_h_cm']:.1f}cm)",
            (x1 + 92, ym + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )

        # 3. Draw tracked water jet trajectory & length
        xs = np.linspace(x_noz, x_noz + dx_px, 60)
        ys = y_noz + dy_px * ((xs - x_noz) / dx_px) ** 2
        pts = np.column_stack([xs, ys]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], False, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(vis, (round(x_noz), round(y_noz)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (round(x_imp), round(y_ref)), 7, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(vis, (round(x_noz), round(y_ref)), (round(x_imp), round(y_ref)), (180, 180, 180), 1, cv2.LINE_AA)

        # Crop region showing entire 60cm ruler, cylinder, meniscus, and water jet [430:1390, 70:750]
        crop = vis[430:1390, 70:750].copy()

        # Header info box
        cv2.rectangle(crop, (10, 8), (665, 108), (15, 15, 15), -1)
        cv2.rectangle(crop, (10, 8), (665, 108), (0, 255, 255), 1)
        cv2.putText(
            crop,
            f"{cfg['label']} - {stage_label} | t = {t_val:.0f} s (Frame {int(row['frame_60fps'])})",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            crop,
            f"Menisco: Marca {row['posicion_regla_cm']:.1f} cm | Dist. tope: {row['distancia_tope_regla_cm']:.2f} cm | Columna h: {row['altura_agua_h_cm']:.2f} cm",
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (100, 255, 100),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            crop,
            f"Chorro: Longitud L = {row['longitud_chorro_cm']:.2f} cm | Alcance X = {row['alcance_horizontal_chorro_cm']:.2f} cm | v = {row['velocidad_salida_ms']:.2f} m/s",
            (18, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        panels.append(crop)

    cap.release()
    grid = np.hstack(panels)
    img_out = out_dir / f"comparacion_3_posiciones_{video_stem}.jpg"
    cv2.imwrite(str(img_out), grid)
    print(f"  -> Imagen de 3 posiciones guardada: {img_out.name}")
    return img_out


def generate_video_plot(
    video_stem: str,
    cfg: dict,
    df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    t = df["tiempo_s"].to_numpy()
    dist_top = df["distancia_tope_regla_cm"].to_numpy()
    h_col = df["altura_agua_h_cm"].to_numpy()
    len_jet = df["longitud_chorro_cm"].to_numpy()
    reach_x = df["alcance_horizontal_chorro_cm"].to_numpy()

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)

    # Panel 1: Meniscus position vs Time
    ax1.plot(t, h_col, color="#0284C7", linewidth=2.5, label="Altura de agua $h(t)$ sobre marca 254 [cm]")
    ax1.plot(t, dist_top, color="#059669", linewidth=2.2, linestyle="--", label="Distancia desde tope de regla $d(t)$ [cm]")
    ax1.axhline(54.0, color="#DC2626", linestyle=":", linewidth=1.2, label="Posición 254 de la regla (54 cm del tope)")
    ax1.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax1.set_ylabel("Posición en la regla de 60 cm [cm]", fontsize=11)
    ax1.set_title(f"{cfg['label']}: Posición del Menisco vs Tiempo", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.55)
    ax1.legend(loc="center right", fontsize=9.5)

    # Panel 2: Water Jet Length & Horizontal Reach vs Time
    ax2.plot(t, len_jet, color="#E11D48", linewidth=2.5, label="Longitud total del chorro $L(t)$ (arco) [cm]")
    ax2.plot(t, reach_x, color="#7C3AED", linewidth=2.2, linestyle="--", label="Alcance horizontal del chorro $X(t)$ [cm]")
    ax2.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax2.set_ylabel("Longitud del chorro de agua [cm]", fontsize=11)
    ax2.set_title(f"{cfg['label']}: Longitud del Chorro vs Tiempo", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.55)
    ax2.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    plot_out = out_dir / f"grafica_{video_stem}.png"
    plt.savefig(plot_out, dpi=160)
    plt.close()
    print(f"  -> Gráfica guardada: {plot_out.name}")
    return plot_out


def generate_summary_comparison_plot(all_dfs: list[pd.DataFrame], out_dir: Path) -> Path:
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=160)
    colors = ["#0284C7", "#059669", "#7C3AED", "#E11D48"]

    for df, color in zip(all_dfs, colors):
        stem = df["video"].iloc[0]
        label = VIDEO_CONFIGS[stem]["label"]
        t = df["tiempo_s"].to_numpy()
        ax1.plot(t, df["altura_agua_h_cm"].to_numpy(), color=color, linewidth=2.2, label=f"{label} ($h(t)$)")
        ax2.plot(t, df["longitud_chorro_cm"].to_numpy(), color=color, linewidth=2.2, label=f"{label} ($L(t)$)")

    ax1.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax1.set_ylabel("Altura del menisco sobre marca 254 $h(t)$ [cm]", fontsize=11)
    ax1.set_title("Comparativa de los 4 Videos: Descenso del Menisco vs Tiempo", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.55)
    ax1.legend(fontsize=10)

    ax2.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax2.set_ylabel("Longitud del chorro de agua $L(t)$ [cm]", fontsize=11)
    ax2.set_title("Comparativa de los 4 Videos: Longitud del Chorro vs Tiempo", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.55)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    summary_path = out_dir / "grafica_comparativa_4_videos.png"
    plt.savefig(summary_path, dpi=160)
    plt.close()
    print(f"\nGráfica comparativa general guardada en: {summary_path}")
    return summary_path


def main() -> None:
    data_dir = Path("/home/dersu/Code/Python/light_tracker/Data")
    out_dir = Path("/home/dersu/Code/Python/light_tracker/analisis_torricelli")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    for stem, cfg in VIDEO_CONFIGS.items():
        df_v = process_single_video(stem, cfg, data_dir, out_dir, sample_step_s=3)
        all_dfs.append(df_v)
        df_v.to_csv(data_dir / f"datos_{stem}.csv", index=False)

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_csv_out = out_dir / "datos_completos_4_videos.csv"
    master_csv_data = data_dir / "datos_completos_4_videos.csv"
    master_df.to_csv(master_csv_out, index=False)
    master_df.to_csv(master_csv_data, index=False)
    print(f"\nArchivo CSV maestro guardado en:\n  - {master_csv_out}\n  - {master_csv_data}\nTotal de registros: {len(master_df)}")

    generate_summary_comparison_plot(all_dfs, out_dir)


if __name__ == "__main__":
    main()
