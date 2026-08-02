"""
fig5b_gradient_validation.py
Gradiente de validación: MAE media de folds por modelo y esquema (escala lineal).
Con la especificación corregida de GWR (intercepto no penalizado, λ por CV espacial anidado) ningún
modelo diverge; bajo separación estricta los modelos convergen a un rango común (~93-110
USD/m²) y el ordenamiento se aplana. La figura ilustra el hallazgo central: el ranking
depende del esquema de validación y de la especificación, sin un ganador robusto.
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

OUT_DIR  = Path(__file__).parent
ROOT     = OUT_DIR.parent.parent
ANALISIS = ROOT / "analisis" / "output_log"

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

COLOR_MODEL = {"SANNWR": "#4CAF50", "GNNWR": "#2196F3", "RF": "#F57C00",
               "OLS": "#9E9E9E", "GWR-27": "#E53935"}
MARKER = {"SANNWR": "o", "GNNWR": "s", "RF": "^", "OLS": "D", "GWR-27": "X"}

cdf = pd.read_csv(ANALISIS / "comparativo_cv_principal.csv")

# Random Forest no pasa por analisis_log.py (pipeline separado de baselines
# tabulares): se inyecta aqui desde su propio results.csv (fold-level, run
# base seed=42, igual que en la Tabla 2/6, que no reportan sigma para RF).
_rf_res = ROOT / "modelos" / "baselines" / "output_log" / "rf_log_results.csv"
if _rf_res.exists():
    _rf_r = pd.read_csv(_rf_res)
    _rf_cv = _rf_r.groupby("estrategia")[["MAE", "RMSE", "MAPE", "R2"]].mean().reset_index()
    _rf_cv.columns = ["estrategia", "MAE_mean", "RMSE_mean", "MAPE_mean", "R2_mean"]
    _rf_cv["modelo_display"] = "RF"
    _rf_cv["modelo"] = "RF"
    cdf = pd.concat([cdf, _rf_cv], ignore_index=True)

SCHEMES_DISPLAY = {
    "RandomKFold":      "RandomKFold\n(interpolación)",
    "SpatialBlock":     "SpatialBlock\n(separación parcial)",
    "SpatialBlock_buf": "SpatialBlock_buf\n(separación estricta)",
}
SCHEMES = list(SCHEMES_DISPLAY.keys())

def _place_labels_no_overlap(ax, x_pos, entries, min_gap=4.0, base_offset=2.6,
                              marker_clear=3.2, leader_thresh=4.0, step=0.6):
    """entries: list of (value, color). Coloca etiquetas apiladas de abajo hacia arriba,
    evitando tanto el solape entre etiquetas como el solape con marcadores de OTRAS series
    (p. ej. una etiqueta empujada hacia arriba que caería justo sobre otro marcador),
    con una línea guía punteada cuando el desplazamiento respecto a su propio punto es grande."""
    order = sorted(entries, key=lambda e: e[0])
    marker_vals = [v for v, _ in entries]
    placed = []

    def clashes(y):
        if any(abs(y - m) < marker_clear for m in marker_vals):
            return True
        if placed and (y - placed[-1]) < min_gap:
            return True
        return False

    for val, color in order:
        y = val + base_offset
        while clashes(y):
            y += step
        placed.append(y)
        if y - val > leader_thresh:
            ax.plot([x_pos, x_pos], [val + 0.8, y - 1.0], color=color,
                    lw=0.7, ls=":", alpha=0.6, zorder=2)
        ax.text(x_pos, y, f"{val:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=color, fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5))

    series = {}
    for model in ["SANNWR", "GNNWR", "RF", "OLS", "GWR-27"]:
        vals = []
        for sch in SCHEMES:
            row = cdf[(cdf["modelo_display"] == model) & (cdf["estrategia"] == sch)]
            vals.append(float(row["MAE_mean"].iloc[0]) if not row.empty else np.nan)
        series[model] = vals
        x_pos = list(range(len(SCHEMES)))
        ax.plot(x_pos, vals, color=COLOR_MODEL[model], lw=2.0,
                marker=MARKER[model], ms=9, label=model, zorder=3)

    for xi in range(len(SCHEMES)):
        entries = [(series[m][xi], COLOR_MODEL[m]) for m in series if not np.isnan(series[m][xi])]
        _place_labels_no_overlap(ax, xi, entries)

    # Rango descriptivo bajo extrapolación estricta (1 sola seed neuronal; no confirmatorio)
    # Rango real de los 5 modelos focales en SpatialBlock_buf: 93.1-106.0 USD/m² (consistente con el texto, §5.5)
    ax.axhspan(93, 106, alpha=0.06, color="#1565C0", zorder=0)
    ax.text(2.35, 88, "rango descriptivo\n93–106 USD/m²\n(1 seed; no confirmatorio)",
            ha="center", va="center",
            fontsize=8, color="#1565C0", style="italic")

    ax.axvspan(1.5, 2.5, alpha=0.05, color="#D32F2F", zorder=0)
    ax.text(2, 40, "extrapolación\ngeográfica estricta", ha="center", va="bottom",
            fontsize=8.5, color="#D32F2F", style="italic", alpha=0.8)

    ax.set_xticks(range(len(SCHEMES)))
    ax.set_xticklabels([SCHEMES_DISPLAY[s] for s in SCHEMES], fontsize=10)
    ax.set_ylabel("MAE media de folds (USD/m²)", fontsize=11)
    ax.set_ylim(35, 120)
    ax.set_xlim(-0.3, 2.8)
    ax.set_title("Gradiente MAE por esquema de validación\n"
                 "(de interpolación a extrapolación estricta)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.85)

    fig.suptitle(
        "Gradiente de validación espacial: MAE por modelo y esquema (5 modelos focales)\n"
        "DMQ Quito · log(precio de oferta) · SpatialBlock_buf: 1 seed neuronal (exploratorio) · HGB (control) en Tabla 6/Anexo G",
        fontsize=11, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5b_gradient_validation.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5b_gradient_validation.pdf", bbox_inches="tight")
    print("  -> fig5b_gradient_validation.png / .pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()
