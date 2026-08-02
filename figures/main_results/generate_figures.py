"""
generate_figures.py
===================
Figuras de publicacion para la tesis.
Lee directamente de analisis/output_log/*.csv — sin valores hardcodeados.

Escenario: log(precio de oferta) + holdout 80/20
Tabla principal (6 modelos): OLS, MLP, GWR-27, GNNWR, SANNWR, GSAWR
Anexo: GWR-17 (17 vars VIF-clean, analisis de sensibilidad)
SAR/SEM excluidos del documento (usan y_train en prediccion; archivados en archivo/).

Figures
-------
  Fig1 — Ranking holdout: RMSE + R2 + Moran I
  Fig2 — RandomKFold vs SpatialBlock (CV comparison)
  Fig3 — Moran I residuos holdout
"""
from __future__ import annotations

# --- normalizacion de etiquetas SOLO para display (no toca claves de datos .loc[]) ---
import matplotlib.text as _mtext
_ORIG_SETTEXT = _mtext.Text.set_text
def _fix_label(s):
    if isinstance(s, str):
        for a, b in ((" "+chr(0x2014)+" ", " - "), (chr(0x2014), " - "),
                     (" "+chr(0xB7)+" ", ", "), (chr(0xB7), ", "),
                     (chr(0x2013), "-"),
                     ("SANNWR-"+chr(0x3B1), "SANNWR"), ("GWR-27", "GWR"),
                     ("n=1.011", "n=1,011"), ("n=1.007", "n=1,007")):
            s = s.replace(a, b)
        import re as _re
        s = _re.sub(r" +,", ",", s)
        s = _re.sub(r" {2,}", " ", s)
    return s
def _set_text_fixed(self, s):
    return _ORIG_SETTEXT(self, _fix_label(s))
_mtext.Text.set_text = _set_text_fixed
# --- fin normalizacion ---

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
OUT_DIR   = Path(__file__).parent
ROOT      = OUT_DIR.parent.parent
ANALISIS  = ROOT / "analisis" / "output_log"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

C_RKF   = "#2166AC"
C_SB    = "#D6604D"
C_HOLD  = "#1B7837"
ALPHA   = 0.82

COLOR_MODEL = {
    "OLS":       "#9E9E9E",
    "MLP":       "#00897B",
    "GWR-17":    "#90A4AE",
    "GWR-27":    "#607D8B",
    "GNNWR":     "#2196F3",
    "SANNWR":    "#4CAF50",
    "GSAWR":     "#9C27B0",
    "RF":        "#F57C00",
}

# Modelos focales de la comparación principal (conjunto pedido por el tutor):
# OLS, GWR, GNNWR, SANNWR, Random Forest. MLP, GWR-17, GSAWR y HGB son modelos
# de referencia/variantes exploratorias, reportados solo en el Anexo G -> no
# incluir aquí ni en las figuras principales.
PRINCIPAL = ["OLS","GWR-27","GNNWR","SANNWR","RF"]

def save_fig(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"  -> {name}.png / .pdf")

# ── Cargar datos ───────────────────────────────────────────────────────────────
# Preferir CSVs "principal" (5 modelos core, sin GWR-17/GSAWR); fallback al completo.
def _read(csv_principal, csv_full):
    p = ANALISIS / csv_principal
    return pd.read_csv(p if p.exists() else ANALISIS / csv_full)

hdf = _read("comparativo_holdout_principal.csv", "comparativo_holdout.csv")
cdf = _read("comparativo_cv_principal.csv",      "comparativo_cv.csv")
mdf = _read("comparativo_moran_principal.csv",   "comparativo_moran.csv")

# Override con metricas SMEARED de Duan (definitivas, consistentes con la Tabla 1)
_sm = ANALISIS / "comparativo_holdout_smeared.csv"
if _sm.exists():
    _smdf = pd.read_csv(_sm).set_index("modelo")
    for _i, _row in hdf.iterrows():
        _m = _row["modelo_display"]
        if _m in _smdf.index:
            hdf.at[_i, "RMSE"] = float(_smdf.loc[_m, "RMSE_smeared"])
            hdf.at[_i, "R2"]   = float(_smdf.loc[_m, "R2_smeared"])
            if "MAE" in hdf.columns:
                hdf.at[_i, "MAE"] = float(_smdf.loc[_m, "MAE_smeared"])

# Random Forest no pasa por analisis_log.py (pipeline separado de baselines
# tabulares): se inyecta aqui desde sus propios artefactos, ya con smearing
# de Duan aplicado internamente por baselines_tabulares.py.
_rf_hold = ROOT / "modelos" / "baselines" / "output_log" / "rf_log_holdout.csv"
if _rf_hold.exists():
    _rf_h = pd.read_csv(_rf_hold)
    _rf_h["modelo_display"] = "RF"
    hdf = pd.concat([hdf, _rf_h], ignore_index=True)

_rf_res = ROOT / "modelos" / "baselines" / "output_log" / "rf_log_results.csv"
if _rf_res.exists():
    _rf_r = pd.read_csv(_rf_res)
    _rf_cv = (_rf_r.groupby("estrategia")[["MAE", "RMSE", "MAPE", "R2"]]
              .agg(["mean", "std"]))
    _rf_cv.columns = [f"{c}_{s}" for c, s in _rf_cv.columns]
    _rf_cv = _rf_cv.reset_index()
    _rf_cv["modelo_display"] = "RF"
    _rf_cv["modelo"] = "RF"
    cdf = pd.concat([cdf, _rf_cv], ignore_index=True)

_moran_sig = ANALISIS / "moran_holdout_significancia.csv"
if _moran_sig.exists():
    _msig = pd.read_csv(_moran_sig)
    _rf_m = _msig[_msig["modelo"] == "RF"].copy()
    if not _rf_m.empty:
        _rf_m["estrategia"] = "Holdout20%"
        _rf_m["modelo_display"] = "RF"
        _rf_m["EI"] = -1.0 / (float(_rf_m["n_test"].iloc[0]) - 1)
        mdf = pd.concat([mdf, _rf_m[["modelo", "estrategia", "I", "EI", "modelo_display",
                                      "p_sim", "z_sim"]]], ignore_index=True)

# Filtrar a los 5 modelos focales de esta comparacion (MLP queda solo en Anexo G).
hdf = hdf[hdf["modelo_display"].isin(PRINCIPAL)].reset_index(drop=True)
cdf = cdf[cdf["modelo_display"].isin(PRINCIPAL)].reset_index(drop=True)
mdf = mdf[mdf["modelo_display"].isin(PRINCIPAL)].reset_index(drop=True)

# Réplicas CV (5 seeds × 5 folds, RTX 4090) para modelos neurales en fig2.
# SANNWR usa la adaptacion canonica (Ni et al. 2022, output_log_real), no la
# variante de grilla SANNWR* (output_log) que reporta el Anexo G. OLS, GWR-27
# y RF son deterministas o se reportan con su run base (Tabla 2) y vienen de cdf.
NEURAL_REPLICA_CV = {}
_REPLICA_SRC = {
    "GNNWR":  ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_cv_replicas_summary.csv",
    "SANNWR": ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_cv_replicas_summary.csv",
}
for _name, _p in _REPLICA_SRC.items():
    if _p.exists():
        NEURAL_REPLICA_CV[_name] = pd.read_csv(_p).set_index("estrategia")

# Ordenar por RMSE holdout ascendente (mejor -> peor)
ORDER_HOLD = (hdf.sort_values("RMSE")["modelo_display"]
              .tolist() if "RMSE" in hdf.columns else PRINCIPAL)

# ── Fig1 — Ranking holdout (RMSE + R2 + Moran I) ─────────────────────────────
def figure1():
    """Ranking interno de modelos por RMSE holdout (smeared)."""
    # Ordenar por RMSE
    ordered = hdf.set_index("modelo_display").reindex(ORDER_HOLD).reset_index()

    modelos = ordered["modelo_display"].tolist()
    rmse    = ordered["RMSE"].tolist()
    r2      = ordered["R2"].tolist()

    # Moran I holdout
    moran_hold = mdf[mdf["estrategia"]=="Holdout20%"].set_index("modelo_display")["I"]
    moran_vals = [float(moran_hold.get(m, np.nan)) for m in modelos]

    colors = [COLOR_MODEL.get(m, "#333") for m in modelos]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: RMSE
    ax = axes[0]
    bars = ax.barh(range(len(modelos)), rmse, color=colors, alpha=ALPHA,
                   edgecolor="black", lw=0.5)
    for bar, v in zip(bars, rmse):
        ax.text(v + 1.5, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}", va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(modelos)))
    ax.set_yticklabels(modelos, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (USD/m2)", fontsize=11)
    ax.set_title("RMSE — Holdout 20%", fontsize=11, fontweight="bold")
    ax.axvline(min(v for v in rmse if not np.isnan(v)),
               color="black", lw=0.8, ls=":", alpha=0.5)

    # Panel 2: R2
    ax = axes[1]
    bars = ax.barh(range(len(modelos)), r2, color=colors, alpha=ALPHA,
                   edgecolor="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, r2):
        xpos = v + 0.01 if v >= 0 else v - 0.04
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(modelos)))
    ax.set_yticklabels(modelos, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("R2 (escala original)", fontsize=11)
    ax.set_title("R2 — Holdout 20%", fontsize=11, fontweight="bold")

    # Panel 3: Moran I residuos holdout (sin umbral inferencial arbitrario)
    ax = axes[2]
    moran_colors = [COLOR_MODEL.get(m, "#607D8B") for m in modelos]
    bars = ax.barh(range(len(modelos)), moran_vals, color=moran_colors, alpha=ALPHA,
                   edgecolor="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, moran_vals):
        if not np.isnan(v):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", ha="left", fontsize=9)
    ax.set_yticks(range(len(modelos)))
    ax.set_yticklabels(modelos, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Indice de Moran I (residuos)", fontsize=11)
    ax.set_title("Moran I — Residuos (holdout; todos p<0.001)", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Evaluación interna (holdout 20%, conjunto de desarrollo, n=1,011)\n"
        f"DMQ Quito · log(precio de oferta) · {len(modelos)} modelos focales · RMSE con smearing de Duan",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    save_fig(fig, "fig1_holdout_ranking")
    plt.close(fig)


# ── Fig2 — RandomKFold vs SpatialBlock ────────────────────────────────────────
def figure2():
    """CV comparison: interpolacion vs generalizacion espacial.

    Modelos neurales: media de réplicas CV (5 seeds × 5 folds, RTX 4090) —
    consistente con la Tabla 2 del README. OLS y GWR-27 (deterministas)
    vienen del run base (comparativo_cv_principal.csv).
    """
    MODELS_CV = PRINCIPAL

    rkf = cdf[cdf["estrategia"]=="RandomKFold"].set_index("modelo_display")
    sb  = cdf[cdf["estrategia"]=="SpatialBlock"].set_index("modelo_display")

    def safe(df, m, col):
        try: return float(df.loc[m, col])
        except: return np.nan

    def cv_val(m, estrategia, col):
        # Neurales: réplicas. Deterministas: run base.
        if m in NEURAL_REPLICA_CV:
            try: return float(NEURAL_REPLICA_CV[m].loc[estrategia, col])
            except: return np.nan
        return safe(rkf if estrategia == "RandomKFold" else sb, m, col)

    rmse_rkf = [cv_val(m, "RandomKFold",  "RMSE_mean") for m in MODELS_CV]
    rmse_sb  = [cv_val(m, "SpatialBlock", "RMSE_mean") for m in MODELS_CV]
    r2_rkf   = [cv_val(m, "RandomKFold",  "R2_mean")   for m in MODELS_CV]
    r2_sb    = [cv_val(m, "SpatialBlock", "R2_mean")   for m in MODELS_CV]

    n     = len(MODELS_CV)
    x     = np.arange(n)
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (rkf_vals, sb_vals, ylabel, title, ylim) in zip(axes, [
        (rmse_rkf, rmse_sb, "RMSE (USD/m²)", "RMSE — RandomKFold vs SpatialBlock", (0, 200)),
        (r2_rkf,   r2_sb,   "R²",            "R² — RandomKFold vs SpatialBlock",   (0, 1.0)),
    ]):
        # Clipear barras al ylim para evitar que rompan el layout
        rkf_plot = [min(v, ylim[1]*0.98) if not np.isnan(v) else np.nan for v in rkf_vals]
        sb_plot  = [min(v, ylim[1]*0.98) if not np.isnan(v) else np.nan for v in sb_vals]

        b1 = ax.bar(x - width/2, rkf_plot, width, label="RandomKFold (interpolación)",
                    color=C_RKF, alpha=ALPHA, edgecolor="black", lw=0.5)
        b2 = ax.bar(x + width/2, sb_plot,  width, label="SpatialBlock (generalización)",
                    color=C_SB,  alpha=ALPHA, edgecolor="black", lw=0.5)

        label_offset = (ylim[1] - ylim[0]) * 0.015
        for bar, v, vp in zip(b1, rkf_vals, rkf_plot):
            if not np.isnan(v):
                label = f"{v:.2f}" if ylabel == "R²" else f"{v:.0f}"
                if v > ylim[1] * 0.9:  # barra clipeada — label en blanco dentro
                    ax.text(bar.get_x() + bar.get_width()/2, vp * 0.5,
                            label, ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, vp + label_offset,
                            label, ha="center", va="bottom", fontsize=8,
                            color=C_RKF, fontweight="bold")
        for bar, v, vp in zip(b2, sb_vals, sb_plot):
            if not np.isnan(v):
                label = f"{v:.2f}" if ylabel == "R²" else f"{v:.0f}"
                if v > ylim[1] * 0.9:
                    ax.text(bar.get_x() + bar.get_width()/2, vp * 0.5,
                            label, ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, vp + label_offset,
                            label, ha="center", va="bottom", fontsize=8,
                            color=C_SB)

        if ylabel == "R²":
            ax.axhline(0, color="black", lw=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS_CV, fontsize=10, rotation=0)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3, ls="--"); ax.set_axisbelow(True)

    # Sin titulo general: lo que decia (hardware, numero de semillas) va en el
    # pie de figura del documento, que es donde el lector lo busca.
    # Una sola leyenda para los dos paneles, fuera del area de datos: dentro
    # tapaba la etiqueta de GNNWR, y una por panel repetia lo mismo dos veces.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.93, bottom=0.20, left=0.07, right=0.97, wspace=0.28)
    fig.savefig(OUT_DIR / "fig2_rkf_vs_sb_comparison.png", dpi=180)
    fig.savefig(OUT_DIR / "fig2_rkf_vs_sb_comparison.pdf")
    print("  -> fig2_rkf_vs_sb_comparison.png / .pdf")
    plt.close(fig)


# ── Fig3 — Moran I progression across models ─────────────────────────────────
def figure3():
    """Moran I de residuos: cuanto autocorrelacion espacial queda en cada modelo."""
    # Usar holdout Moran I (mas honesto)
    hold_moran = mdf[mdf["estrategia"]=="Holdout20%"].copy()
    hold_moran = hold_moran.set_index("modelo_display").reindex(ORDER_HOLD).reset_index()
    hold_moran = hold_moran.dropna(subset=["I"])

    modelos = hold_moran["modelo_display"].tolist()
    I_vals  = hold_moran["I"].tolist()
    colors  = [COLOR_MODEL.get(m, "#333") for m in modelos]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(modelos))
    bars = ax.bar(x, I_vals, color=colors, alpha=ALPHA, edgecolor="black", lw=0.6)

    # Etiquetas
    for bar, v in zip(bars, I_vals):
        ypos = v + 0.005 if v >= 0 else v - 0.015
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Linea de referencia: residuos aleatorios (sin umbral inferencial arbitrario)
    ax.axhline(0, color="black", lw=1.0, label="I=0 (residuos aleatorios)")

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=11, rotation=20, ha="right")
    ax.set_ylabel("Indice de Moran I", fontsize=12)
    ax.set_title(
        "Autocorrelación residual (holdout 20%, seed=42)\n"
        "Todos los modelos reducen la autocorrelación vs OLS, pero ninguno la elimina (todos p=0.001, mínimo con 999 permutaciones)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.05, 0.75)

    plt.tight_layout()
    save_fig(fig, "fig3_moran_residuals")
    plt.close(fig)


# ── Resumen texto ──────────────────────────────────────────────────────────────
def write_summary():
    hdf_s = hdf.set_index("modelo_display").reindex(ORDER_HOLD)
    lines = ["RESUMEN — Figuras Principales de Tesis",
             "=" * 45,
             "Escenario: log(precio de oferta) + holdout 80/20",
             "DMQ Quito · 5,051 obs · CRS EPSG:32717",
             "5 modelos focales: OLS, GWR, GNNWR, SANNWR, RF (MLP/GWR-17/GSAWR/HGB en Anexo G)",
             "",
             "HOLDOUT 20% (n=1,011) — ranking interno (smeared):",
             f"{'Modelo':15s} {'RMSE':>8s} {'R2':>7s} {'MAPE':>8s}",
             "-" * 42]
    for m in ORDER_HOLD:
        row = hdf_s.loc[m] if m in hdf_s.index else None
        if row is not None:
            lines.append(f"{m:15s} {row['RMSE']:8.2f} {row['R2']:7.4f} {row['MAPE']:7.2f}%")
    lines += [
        "",
        "MORAN I HOLDOUT (residuos):",
        f"{'Modelo':15s} {'I':>7s} {'Interpretacion':20s}",
        "-" * 45,
    ]
    mhold = mdf[mdf["estrategia"]=="Holdout20%"].set_index("modelo_display")
    for m in ORDER_HOLD:
        if m in mhold.index:
            I = mhold.loc[m, "I"]
            interp = "autocorrelacion residual (p=0.001)"
            lines.append(f"{m:15s} {I:7.4f} {interp}")
    (OUT_DIR / "figures_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  -> figures_summary.txt")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando figuras de tesis...")
    print("\n[Fig1] Ranking holdout")
    figure1()
    print("\n[Fig2] RKF vs SpatialBlock")
    figure2()
    print("\n[Fig3] Moran I residuos")
    figure3()
    print("\n[Resumen]")
    write_summary()
    print(f"\nGuardado en: {OUT_DIR}")
