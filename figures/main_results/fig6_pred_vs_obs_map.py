"""
fig6_pred_vs_obs_map.py
=======================
Mapa 2x2 sobre el holdout test (n=1,011):
  (a) valor observado (USD/m2)
  (b) prediccion GWR-27 + Duan  — equivalente estadístico en interpolación (holdout)
  (c) prediccion GNNWR + Duan   — menor RMSE puntual bajo SpatialBlock (98.1±1.88), no significativo tras Holm-Bonferroni
  (d) |error relativo| de GNNWR (%)

GWR-27, GNNWR (Du 2020), SANNWR (Ni 2022) y GSAWR (Xu 2025) son todos
arquitecturas de referencia replicadas; el aporte de la tesis es el
protocolo de evaluación, no un modelo nuevo. La elección GWR-27 vs GNNWR
para el mapa ilustra el contraste central: el modelo lineal interpretable
(GWR-27) y una red de pesos espaciales (GNNWR) producen mapas visualmente
similares en interpolación; bajo extrapolación el ordenamiento depende
del esquema de validación y de la especificación del estimador, sin un
ganador robusto (Sección 8.5).
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

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT     = Path(__file__).parent.parent.parent
DATA     = ROOT / "datos" / "dataset.gpkg"
SPLIT    = ROOT / "data_split" / "split.csv"
ANALISIS = ROOT / "analisis" / "output_log"
OUT      = ROOT / "figures" / "main_results"
OUT.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

def load_preds(model_name, path):
    df = pd.read_csv(path)
    df = df[df["split"] == "test"].copy()
    df["predio_join"] = df["predio_join"].astype(int)
    df = df.rename(columns={"y_pred_log": f"yhat_{model_name}"})
    return df[["predio_join", f"yhat_{model_name}"]]

def main():
    gdf = gpd.read_file(DATA, layer="puntos_mercado").to_crs(epsg=32717)
    split_df = pd.read_csv(SPLIT)
    gdf["predio_join"]      = gdf["predio_join"].astype(int)
    split_df["predio_join"] = split_df["predio_join"].astype(int)
    gdf = gdf.merge(split_df[["predio_join", "split"]], on="predio_join", how="left")
    gdf_te = gdf[gdf["split"] == "test"].copy().reset_index(drop=True)
    gdf_te["x"] = gdf_te.geometry.x / 1000   # km para etiquetas
    gdf_te["y"] = gdf_te.geometry.y / 1000
    gdf_te["valor_m2"] = gdf_te["valor_m2"].astype(float)

    # Predicciones GWR-27 y GNNWR (menor RMSE puntual bajo SpatialBlock: 98.1±1.88, no significativo)
    gwr27_df = load_preds("gwr27", ROOT/"modelos"/"gwr"/"output_log_27vars"/"gwr27_log_predictions.csv")
    gnnwr_df = load_preds("gnnwr", ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_predictions.csv")

    smeared = pd.read_csv(ANALISIS / "comparativo_holdout_smeared.csv").set_index("modelo")
    s_gwr27 = float(smeared.loc["GWR-27", "smearing_factor"])
    s_gnnwr = float(smeared.loc["GNNWR",  "smearing_factor"])
    rmse_gwr27 = float(smeared.loc["GWR-27", "RMSE_smeared"])
    rmse_gnnwr = float(smeared.loc["GNNWR",  "RMSE_smeared"])

    gdf_te = gdf_te.merge(gwr27_df, on="predio_join", how="left")
    gdf_te = gdf_te.merge(gnnwr_df, on="predio_join", how="left")
    gdf_te["pred_gwr27"] = np.exp(gdf_te["yhat_gwr27"]) * s_gwr27
    gdf_te["pred_gnnwr"] = np.exp(gdf_te["yhat_gnnwr"]) * s_gnnwr
    gdf_te["err_rel_gnnwr"] = np.abs(gdf_te["pred_gnnwr"] - gdf_te["valor_m2"]) / gdf_te["valor_m2"] * 100

    vmin = max(float(min(gdf_te["valor_m2"].min(),
                         gdf_te["pred_gwr27"].min(),
                         gdf_te["pred_gnnwr"].min())), 1.0)
    vmax = float(max(gdf_te["valor_m2"].max(),
                     gdf_te["pred_gwr27"].max(),
                     gdf_te["pred_gnnwr"].max()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    def panel(ax, col, title, cmap="viridis", use_norm=True, vmin_=None, vmax_=None, unit="USD/m²"):
        kw = {"norm": norm} if use_norm else {"vmin": vmin_, "vmax": vmax_}
        sc = ax.scatter(gdf_te["x"], gdf_te["y"], c=gdf_te[col],
                        s=7, cmap=cmap, alpha=0.88, edgecolors="none", **kw)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Este (km, UTM 17S)", fontsize=9)
        ax.set_ylabel("Norte (km, UTM 17S)", fontsize=9)
        ax.tick_params(labelsize=8)
        cb = plt.colorbar(sc, ax=ax, shrink=0.80, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(unit, fontsize=9)
        return sc

    panel(axes[0, 0], "valor_m2",    "(a) Valor observado")
    panel(axes[0, 1], "pred_gwr27",  f"(b) GWR-27 · RMSE={rmse_gwr27:.1f} + Duan\n(lineal espacial — equivalente en holdout)")
    panel(axes[1, 0], "pred_gnnwr",  f"(c) GNNWR (Du 2020) · RMSE={rmse_gnnwr:.1f} + Duan\n(red de pesos espaciales — equivalente en holdout)")
    # Calcular cuantos puntos superan el umbral de saturacion (80%)
    if "err_rel_gnnwr" in gdf_te.columns:
        n_over80 = int((gdf_te["err_rel_gnnwr"].abs() > 80).sum())
        n_total  = len(gdf_te)
        lbl_err  = f"(d) |error relativo| GNNWR (%)  [saturado en 80%; {n_over80}/{n_total}={100*n_over80/n_total:.1f}% superan umbral]"
    else:
        lbl_err = "(d) |error relativo| GNNWR (%)"
    panel(axes[1, 1], "err_rel_gnnwr", lbl_err,
          cmap="RdYlGn_r", use_norm=False, vmin_=0, vmax_=80, unit="|error| (%)")

    fig.suptitle(
        "Predicción vs observación en holdout (n=1,011)\n"
        "GWR-27 (lineal espacial) vs GNNWR (ponderación espacial aprendida) · "
        "DMQ · EPSG:32717 · smearing Duan (1983)",
        fontsize=10, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig6_pred_vs_obs_map.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] fig6_pred_vs_obs_map  GWR-27 vs GNNWR")

if __name__ == "__main__":
    main()
