"""
fig4_replicas_stability.py
==========================
Estabilidad de los modelos focales estocásticos con réplicas (10 seeds holdout).
Lee los *_log_replicas_summary.csv de cada modelo y grafica
RMSE_mean ± std en holdout. Documenta que el resultado no depende de la
inicialización aleatoria.

Modelos focales con réplicas neuronales: GNNWR, SANNWR (Tabla 3). Random
Forest tambien tiene réplicas de 10 semillas, con variabilidad despreciable
(sección 5.3); no se grafica aquí porque su escala de σ (~0.6 USD/m²) haría
sus barras de error invisibles junto a las neuronales (σ de 3-4 USD/m²). MLP
y GSAWR son modelos de referencia/variantes, reportados solo en el Anexo G.

Fuentes:
  modelos/gnnwr/output_log/gnnwr_log_replicas_summary.csv
  modelos/sannwr/output_log_real/sannwr_real_log_replicas_summary.csv
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
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT  = Path(__file__).parent

SOURCES = [
    ("GNNWR",  ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_replicas_summary.csv", "#2196F3"),
    ("SANNWR", ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_replicas_summary.csv","#4CAF50"),
]

DETAIL_PATHS = {
    "GNNWR":  ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_replicas.csv",
    "SANNWR": ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_replicas.csv",
}

def seed42_rmse(name: str) -> float | None:
    path = DETAIL_PATHS.get(name)
    if path and path.exists():
        df = pd.read_csv(path)
        row = df[df["seed"] == 42]
        if not row.empty:
            return float(row["RMSE"].iloc[0])
    return None

def main():
    rows = []
    for name, path, color in SOURCES:
        if not path.exists():
            print(f"[--] {name}: {path.name} NO encontrado"); continue
        df = pd.read_csv(path)
        rows.append({
            "modelo": name,
            "RMSE_mean": float(df["RMSE_mean"].iloc[0]),
            "RMSE_std":  float(df["RMSE_std"].iloc[0]),
            "R2_mean":   float(df["R2_mean"].iloc[0]),
            "color": color,
            "ref": seed42_rmse(name),
        })
    if not rows:
        print("Sin datos de réplicas."); return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(rows))
    means = [r["RMSE_mean"] for r in rows]
    stds  = [r["RMSE_std"]  for r in rows]
    colors = [r["color"] for r in rows]

    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.82,
                  edgecolor="black", lw=0.6, error_kw={"elinewidth": 1.4})

    for i, r in enumerate(rows):
        ax.text(i, r["RMSE_mean"] + r["RMSE_std"] + 1.5,
                f"{r['RMSE_mean']:.1f} ± {r['RMSE_std']:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        cv = 100 * r["RMSE_std"] / r["RMSE_mean"]
        ax.text(i, r["RMSE_mean"]/2, f"CV={cv:.1f}%",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if r["ref"] is not None:
            ax.hlines(r["ref"], i-0.4, i+0.4, color="black", ls=":", lw=1.4)
            ax.text(i+0.42, r["ref"], f"seed42={r['ref']:.1f}",
                    ha="left", va="center", fontsize=8, color="black")

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["modelo"] for r in rows], fontsize=11)
    ax.set_ylabel("RMSE sobre la partición reservada (USD/m²)", fontsize=11)
    # Sin titulo interno: enumeraba las diez semillas y usaba jerga que el pie
    # de figura del documento ya cubre en castellano.
    ax.grid(axis="y", alpha=0.3, ls="--"); ax.set_axisbelow(True)
    ax.set_ylim(0, max(m+s for m, s in zip(means, stds)) * 1.25)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_replicas_stability.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] fig4_replicas_stability.png/pdf")
    for r in rows:
        console_name = r["modelo"].replace("α", "alpha")
        print(f"  {console_name:8s} RMSE={r['RMSE_mean']:.2f} +/- {r['RMSE_std']:.2f}  (CV={100*r['RMSE_std']/r['RMSE_mean']:.1f}%)")

if __name__ == "__main__":
    main()
