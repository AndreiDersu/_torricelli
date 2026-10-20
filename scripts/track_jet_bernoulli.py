"""High-precision water jet tracking and Bernoulli-Torricelli hydrodynamics analysis."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_jet_and_bernoulli(
    video_path: str | Path = "/home/dersu/Downloads/20261007_082021.mp4",
    meniscus_csv_path: str | Path = "/home/dersu/Code/Python/light_tracker/analisis_torricelli/nivel_agua_torricelli.csv",
    output_dir: str | Path = "/home/dersu/Code/Python/light_tracker/analisis_torricelli",
    max_time_sec: float = 300.0,
) -> dict:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load meniscus tracking data
    meniscus_df = pd.read_csv(meniscus_csv_path)
    print(f"Cargados {len(meniscus_df)} registros de menisco desde {meniscus_csv_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video abierto: {total_frames} frames a {fps:.2f} FPS")

    # Physical parameters & geometry
    scale_px_per_m = 2500.0  # 25.0 px/cm
    g = 9.80665  # m/s^2
    rho_water = 1000.0  # kg/m^3

    # Geometry inside crop [2430:2750, 500:1250]:
    # Full-frame nozzle: y_full = 2458.0 px, x_full = 510.0 px
    # Local nozzle: y_nozzle = 28.0 px, x_nozzle = 10.0 px
    x_nozzle = 10.0
    y_nozzle = 28.0

    # Reference crossing horizontal plane at the tray top rim:
    # y_ref = 230.0 px (in crop)
    y_ref = 230.0
    dy_px = y_ref - y_nozzle  # 202.0 px
    dy_m = dy_px / scale_px_per_m  # 0.0808 m (8.08 cm)
    t_flight = math.sqrt(2.0 * dy_m / g)  # flight time ~ 0.1284 s

    # Orifice effective area (derived from Torricelli drainage slope and cylinder D = 9.2 cm):
    # d_orif ~ 2.52 mm -> A_orif = pi/4 * d^2 ~ 4.98e-6 m^2
    d_orif_m = 0.00252
    A_orif = math.pi * (d_orif_m / 2.0) ** 2  # ~ 4.988e-6 m^2

    records: list[dict] = []
    prev_x_meas: float | None = None

    # Tracking across each integer second
    valid_df = meniscus_df[meniscus_df["t_s"] <= max_time_sec].copy()
    print(f"Iniciando rastreo del chorro de agua en {len(valid_df)} instantes de tiempo (t=0 a t={max_time_sec:.0f} s)...")

    for _, row in valid_df.iterrows():
        t = float(row["t_s"])
        h_m = float(row["altura_m"])
        h_cm = float(row["altura_cm"])

        # Ideal Torricelli exit velocity
        v_ideal = math.sqrt(2.0 * g * h_m) if h_m > 0 else 0.0

        # Theoretical ideal range at y_ref
        x_nozzle + (v_ideal * t_flight) * scale_px_per_m
        # Physics-guided prior using nominal Cd ~ 0.78
        x_prior = x_nozzle + 0.78 * (v_ideal * t_flight) * scale_px_per_m

        if prev_x_meas is not None:
            x_center = 0.65 * prev_x_meas + 0.35 * x_prior
        else:
            x_center = x_prior

        # Extract frame
        f_idx = min(total_frames - 1, round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        crop = frame[2430:2750, 500:1250]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Average horizontal band across y in [225:235]
        band = gray[225:235, :]
        prof = np.mean(band, axis=0)  # length 750

        # Narrow search window around dynamic physics center (+/- 35 px)
        w_min = max(15, int(x_center - 35))
        w_max = min(735, int(x_center + 35))

        sub_prof = prof[w_min:w_max]
        local_peak = int(np.argmax(sub_prof))
        x_meas = float(w_min + local_peak)

        # Sub-pixel parabolic peak interpolation
        if 0 < local_peak < len(sub_prof) - 1:
            denom = 2.0 * sub_prof[local_peak] - sub_prof[local_peak - 1] - sub_prof[local_peak + 1]
            if abs(denom) > 1e-4:
                subpix = (sub_prof[local_peak + 1] - sub_prof[local_peak - 1]) / (2.0 * denom)
                x_meas += float(np.clip(subpix, -0.5, 0.5))

        prev_x_meas = x_meas

        # Horizontal distance from orifice
        dx_px = max(0.0, x_meas - x_nozzle)
        X_m = dx_px / scale_px_per_m
        X_cm = X_m * 100.0

        # Kinematic exit velocity
        v_exp = X_m / t_flight

        # Discharge / velocity coefficient
        Cd = (v_exp / v_ideal) if v_ideal > 1e-4 else 0.0

        # Hydrodynamic power and dynamic quantities (Bernoulli):
        # 1. Dynamic pressure (Bernoulli kinetic head): q = 0.5 * rho * v^2 [Pa]
        dynamic_pressure_Pa = 0.5 * rho_water * (v_exp**2)

        # 2. Volumetric flow rate: Q = A_orif * v_exp [m^3/s and L/min]
        flow_rate_m3s = A_orif * v_exp
        flow_rate_Lmin = flow_rate_m3s * 60000.0

        # 3. Kinetic power of the water jet: P = 0.5 * m_dot * v^2 = 0.5 * rho * A * v^3 [Watts and milliWatts]
        power_W = 0.5 * rho_water * A_orif * (v_exp**3)
        power_mW = power_W * 1000.0

        # 4. Momentum thrust / impact force: F = m_dot * v = rho * A * v^2 [mN]
        thrust_mN = rho_water * A_orif * (v_exp**2) * 1000.0

        records.append({
            "t_s": t,
            "h_columna_m": h_m,
            "h_columna_cm": h_cm,
            "x_meas_px": x_meas,
            "alcance_X_m": X_m,
            "alcance_X_cm": X_cm,
            "v_exp_ms": v_exp,
            "v_ideal_torricelli_ms": v_ideal,
            "coeficiente_Cd": Cd,
            "presion_dinamica_Pa": dynamic_pressure_Pa,
            "caudal_Lmin": flow_rate_Lmin,
            "potencia_cinetica_mW": power_mW,
            "empuje_impacto_mN": thrust_mN,
        })

        if int(t) % 30 == 0:
            print(f"  t = {t:5.1f} s | h = {h_cm:5.2f} cm | X = {X_cm:5.2f} cm | v_exp = {v_exp:5.3f} m/s | Potencia = {power_mW:6.2f} mW | Cd = {Cd:.3f}")

    cap.release()

    df_results = pd.DataFrame(records)

    # Statistical fits:
    # 1. Horizontal reach X(t) vs t:
    t_vals = df_results["t_s"].values
    X_vals = df_results["alcance_X_cm"].values
    slope_X, intercept_X = np.polyfit(t_vals, X_vals, 1)
    fit_X = intercept_X + slope_X * t_vals
    ss_res_X = np.sum((X_vals - fit_X) ** 2)
    ss_tot_X = np.sum((X_vals - np.mean(X_vals)) ** 2)
    r2_X = 1.0 - (ss_res_X / ss_tot_X)

    # 2. X vs sqrt(h):
    sqrt_h_vals = np.sqrt(df_results["h_columna_cm"].values)
    slope_X_sqrt, intercept_X_sqrt = np.polyfit(sqrt_h_vals, X_vals, 1)
    fit_X_sqrt = intercept_X_sqrt + slope_X_sqrt * sqrt_h_vals
    ss_res_sqrt = np.sum((X_vals - fit_X_sqrt) ** 2)
    ss_tot_sqrt = np.sum((X_vals - np.mean(X_vals)) ** 2)
    r2_sqrt = 1.0 - (ss_res_sqrt / ss_tot_sqrt)

    print("\n--- Resultados del Análisis de Bernoulli y Potencia del Chorro ---")
    print(f"Alcance inicial (t=0): X_0 = {df_results['alcance_X_cm'].iloc[0]:.2f} cm")
    print(f"Alcance final (t=300): X_f = {df_results['alcance_X_cm'].iloc[-1]:.2f} cm")
    print(f"Ajuste X(t) vs t: X(t) = {intercept_X:.2f} - {-slope_X:.4f} * t  (R² = {r2_X:.4f})")
    print(f"Ajuste X vs sqrt(h): R² = {r2_sqrt:.4f}")
    print(f"Potencia cinética inicial: P_0 = {df_results['potencia_cinetica_mW'].iloc[0]:.2f} mW")
    print(f"Potencia cinética final: P_f = {df_results['potencia_cinetica_mW'].iloc[-1]:.2f} mW")
    print(f"Reducción total de potencia: {df_results['potencia_cinetica_mW'].iloc[0] / df_results['potencia_cinetica_mW'].iloc[-1]:.1f} veces")
    print(f"Coeficiente de descarga medio: Cd = {df_results['coeficiente_Cd'].mean():.3f} +/- {df_results['coeficiente_Cd'].std():.3f}")

    # Export CSV
    csv_out = out_dir / "bernoulli_chorro_agua.csv"
    df_results.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"Tabla completa exportada a: {csv_out}")

    # 3. Generate Scientific 4-Panel Plot
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # Subplot (a): Horizontal range X(t) vs t
    ax_a = axes[0, 0]
    ax_a.scatter(t_vals, X_vals, color="#0284C7", s=14, alpha=0.75, label="Datos experimentales $X(t)$")
    ax_a.plot(t_vals, fit_X, color="#DC2626", linewidth=2.0, label=f"Ajuste Lineal ($R^2 = {r2_X:.4f}$)\n$X(t) = {intercept_X:.1f} {slope_X:.3f} t$ cm")
    ax_a.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax_a.set_ylabel("Alcance horizontal $X$ [cm]", fontsize=11)
    ax_a.set_title("(a) Disminución del Alcance del Chorro vs Tiempo", fontsize=12, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.55)
    ax_a.legend(loc="upper right", fontsize=9.5)

    # Subplot (b): X vs sqrt(h) (Torricelli-Bernoulli scaling)
    ax_b = axes[0, 1]
    ax_b.scatter(sqrt_h_vals, X_vals, color="#7C3AED", s=14, alpha=0.75, label="Datos experimentales")
    ax_b.plot(sqrt_h_vals, fit_X_sqrt, color="#059669", linewidth=2.0, label=f"Relación Bernoulli ($R^2 = {r2_sqrt:.4f}$)\n$X \\propto \\sqrt{{h}}$")
    ax_b.set_xlabel("Raíz de la altura $\\sqrt{h}$ [cm$^{1/2}$]", fontsize=11)
    ax_b.set_ylabel("Alcance horizontal $X$ [cm]", fontsize=11)
    ax_b.set_title("(b) Validación de Bernoulli: $X$ vs $\\sqrt{h}$", fontsize=12, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.55)
    ax_b.legend(loc="upper left", fontsize=9.5)

    # Subplot (c): Velocity comparison (v_exp vs v_ideal)
    ax_c = axes[1, 0]
    ax_c.plot(t_vals, df_results["v_ideal_torricelli_ms"].values, color="#F59E0B", linestyle="--", linewidth=2.0, label="Velocidad ideal Torricelli $\\sqrt{2gh}$")
    ax_c.scatter(t_vals, df_results["v_exp_ms"].values, color="#0284C7", s=14, alpha=0.75, label="Velocidad experimental $v_{\\text{exit}}$")
    ax_c.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax_c.set_ylabel("Velocidad de salida $v$ [m/s]", fontsize=11)
    ax_c.set_title("(c) Velocidad de Salida del Agua vs Tiempo", fontsize=12, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.55)
    ax_c.legend(loc="upper right", fontsize=9.5)

    # Subplot (d): Kinetic Power decay P(t) vs t
    ax_d = axes[1, 1]
    power_vals = df_results["potencia_cinetica_mW"].values
    ax_d.plot(t_vals, power_vals, color="#E11D48", linewidth=2.2, label="Potencia cinética $P(t) = \\frac{1}{2} \\dot{m} v^2$ [mW]")
    ax_d.fill_between(t_vals, power_vals, color="#E11D48", alpha=0.15)
    ax_d.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax_d.set_ylabel("Potencia cinética del chorro $P$ [mW]", fontsize=11)
    ax_d.set_title("(d) Decaimiento de la Potencia del Chorro", fontsize=12, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.55)
    ax_d.legend(loc="upper right", fontsize=9.5)

    plot_out = out_dir / "bernoulli_chorro_analisis.png"
    plt.tight_layout()
    plt.savefig(plot_out, dpi=180)
    plt.close()
    print(f"Gráfica científica guardada en: {plot_out}")

    # 4. Generate Visual Verification Overlay Image
    cap_verif = cv2.VideoCapture(str(video_path))
    verif_times = [5, 60, 120, 180, 240, 300]
    verif_panels = []

    for vt in verif_times:
        cap_verif.set(cv2.CAP_PROP_POS_FRAMES, round(vt * fps))
        ret, frame = cap_verif.read()
        if not ret:
            continue
        c = frame[2430:2750, 500:1250].copy()

        # Find row in dataframe
        row_vt = df_results.iloc[(df_results["t_s"] - vt).abs().argsort()[:1]].iloc[0]
        x_m_px = round(row_vt["x_meas_px"])
        X_cm_vt = row_vt["alcance_X_cm"]
        P_mW_vt = row_vt["potencia_cinetica_mW"]
        v_vt = row_vt["v_exp_ms"]

        # Draw trajectory parabola from nozzle (10, 28) to landing (x_m_px, 230)
        # y(x) = y0 + (dy / dx^2) * (x - x0)^2
        dx_tot = max(1.0, float(x_m_px - 10))
        pts_parabola = []
        for xi in np.linspace(10, x_m_px, 50):
            yi = 28.0 + (dy_px / (dx_tot**2)) * ((xi - 10.0) ** 2)
            pts_parabola.append((round(xi), round(yi)))

        for k in range(len(pts_parabola) - 1):
            cv2.line(c, pts_parabola[k], pts_parabola[k + 1], (0, 255, 255), 2, cv2.LINE_AA)

        # Mark reference line y_ref = 230
        cv2.line(c, (10, int(y_ref)), (740, int(y_ref)), (100, 100, 100), 1, cv2.LINE_AA)

        # Mark nozzle
        cv2.circle(c, (10, 28), 5, (0, 0, 255), -1, cv2.LINE_AA)

        # Mark landing point
        cv2.circle(c, (x_m_px, int(y_ref)), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(c, (x_m_px, int(y_ref) - 15), (x_m_px, int(y_ref) + 15), (0, 255, 0), 2, cv2.LINE_AA)

        # Info badge
        cv2.rectangle(c, (8, 8), (340, 75), (0, 0, 0), -1)
        cv2.rectangle(c, (8, 8), (340, 75), (0, 255, 255), 1)
        cv2.putText(c, f"t = {vt:3d} s  |  h = {row_vt['h_columna_cm']:4.1f} cm", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(c, f"Alcance X = {X_cm_vt:4.1f} cm  (v = {v_vt:4.2f} m/s)", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(c, f"Potencia P = {P_mW_vt:5.2f} mW", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 100), 1, cv2.LINE_AA)

        verif_panels.append(c)

    cap_verif.release()

    if len(verif_panels) == 6:
        v_row1 = np.hstack(verif_panels[:3])
        v_row2 = np.hstack(verif_panels[3:])
        v_grid = np.vstack([v_row1, v_row2])
        v_path = out_dir / "verificacion_chorro_bernoulli.jpg"
        cv2.imwrite(str(v_path), v_grid)
        print(f"Verificación visual del chorro guardada en: {v_path}")

    # 5. Generate LaTeX Table
    tex_out = out_dir / "bernoulli_tabla.tex"
    sample_tex_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write("% Tabla de Hidrodinámica: Principio de Bernoulli y Ley de Torricelli\n")
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\caption{Evolución temporal del alcance horizontal, velocidad de salida y potencia del chorro de agua.}\n")
        f.write("\\label{tab:bernoulli_chorro}\n")
        f.write("\\begin{tabular}{ccccccc}\n\\hline\\hline\n")
        f.write("$t$ [s] & $h$ [cm] & $X_{\\text{exp}}$ [cm] & $v_{\\text{exp}}$ [m/s] & $v_{\\text{ideal}}$ [m/s] & $C_d$ & $P_{\\text{cin}}$ [mW] \\\\ \\hline\n")
        for st in sample_tex_times:
            row_st = df_results.iloc[(df_results["t_s"] - st).abs().argsort()[:1]].iloc[0]
            f.write(
                f"{row_st['t_s']:5.0f} & {row_st['h_columna_cm']:5.2f} & {row_st['alcance_X_cm']:5.2f} & "
                f"{row_st['v_exp_ms']:5.3f} & {row_st['v_ideal_torricelli_ms']:5.3f} & "
                f"{row_st['coeficiente_Cd']:5.3f} & {row_st['potencia_cinetica_mW']:6.2f} \\\\\n"
            )
        f.write("\\hline\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"Tabla LaTeX guardada en: {tex_out}")

    return {
        "r2_X_vs_t": r2_X,
        "r2_X_vs_sqrt_h": r2_sqrt,
        "Cd_mean": df_results["coeficiente_Cd"].mean(),
        "P_initial_mW": df_results["potencia_cinetica_mW"].iloc[0],
        "P_final_mW": df_results["potencia_cinetica_mW"].iloc[-1],
        "X_initial_cm": df_results["alcance_X_cm"].iloc[0],
        "X_final_cm": df_results["alcance_X_cm"].iloc[-1],
    }


if __name__ == "__main__":
    analyze_jet_and_bernoulli()
