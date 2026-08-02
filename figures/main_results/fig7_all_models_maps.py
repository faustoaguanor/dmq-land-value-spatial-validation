"""
fig7_all_models_maps.py
=======================
Figura comparativa: predicciones y errores de todos los modelos sobre el
holdout test (n=1,011 predios del DMQ).

Layout 3 figuras independientes (más legible que un solo mosaico):

  fig7a_predictions.png — grilla 2×3: predicción en USD/m² de los 5 modelos
                           focales (RF, SANNWR, GWR-27, GNNWR, OLS); el sexto
                           panel se oculta.
                           Mismo rango de color en todos → diferencias visibles.

  fig7b_errors.png      — grilla 2×3: |error relativo| % de cada modelo
                           Colores divergentes: verde=bajo, rojo=alto.
                           Permite ver dónde falla cada arquitectura en el DMQ.

  fig7c_residuals.png   — grilla 2×3: residuo log-escala (y_obs_log - y_pred_log)
                           Revela sesgo sistemático por zona (norte/sur/valles).
                           Directamente relacionado con el Moran I de cada modelo.

Todos los mapas usan EPSG:32717 (metros UTM), eje en km para legibilidad.
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
                     ("SANNWR-"+chr(0x3B1), "SANNWR"), ("GWR", "GWR"),
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
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, TwoSlopeNorm

ROOT    = Path(__file__).parent.parent.parent
DATA    = ROOT / "datos" / "dataset.gpkg"
SPLIT   = ROOT / "data_split" / "split.csv"
ANALISIS = ROOT / "analisis" / "output_log"
OUT     = ROOT / "figures" / "main_results"
OUT.mkdir(exist_ok=True, parents=True)

# Los cinco modelos focales, en orden de mejor a peor RMSE en holdout.
# MLP y GSAWR quedan fuera (modelos de referencia/variantes, Anexo G) y entra
# Random Forest, que es el modelo focal recomendado para interpolacion y antes
# faltaba en estos mapas.
# Nombre que se dibuja, distinto de la clave interna con que los CSV de
# resultados identifican al modelo (nomenclatura anterior a la reduccion a
# cinco modelos focales).
ETIQUETA = {"GWR-27": "GWR"}
def lbl(n):
    return ETIQUETA.get(n, n)

MODELS = [
    ("RF",     ROOT/"modelos"/"baselines"/"output_log"/"rf_log_predictions.csv",         "#F57C00"),
    ("SANNWR", ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_predictions.csv", "#4CAF50"),
    ("GWR-27", ROOT/"modelos"/"gwr"/"output_log_27vars"/"gwr27_log_predictions.csv", "#607D8B"),
    ("GNNWR",  ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_predictions.csv",          "#2196F3"),
    ("OLS",    ROOT/"modelos"/"ols"/"output_log"/"ols_log_predictions.csv",              "#9E9E9E"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Cargando datos ...")
gdf = gpd.read_file(DATA, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT)
gdf["predio_join"]      = gdf["predio_join"].astype(int)
split_df["predio_join"] = split_df["predio_join"].astype(int)
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")
gdf_te = gdf[gdf["split"]=="test"].copy().reset_index(drop=True)
gdf_te["x"] = gdf_te.geometry.x / 1000   # → km
gdf_te["y"] = gdf_te.geometry.y / 1000

y_obs_orig = gdf_te["valor_m2"].astype(float).values
y_obs_log  = np.log(y_obs_orig)

# Smearing factors
smeared = pd.read_csv(ANALISIS/"comparativo_holdout_smeared.csv").set_index("modelo")

# Cargar predicciones de todos los modelos
preds = {}
for name, path, _ in MODELS:
    df = pd.read_csv(path)
    df = df[df["split"]=="test"].copy()
    df["predio_join"] = df["predio_join"].astype(int)
    merged = gdf_te[["predio_join"]].merge(df[["predio_join","y_pred_log"]], on="predio_join", how="left")
    # Factor de smearing. RF no pasa por analisis_log.py (pipeline separado de
    # baselines tabulares) y por tanto no esta en comparativo_holdout_smeared.csv:
    # su factor se lee de su propio holdout. Sin esto caeria al fallback s=1.0.
    try:
        if name == "RF":
            s = float(pd.read_csv(ROOT/"modelos"/"baselines"/"output_log"/"rf_log_holdout.csv")
                      ["smearing_factor"].iloc[0])
        else:
            s = float(smeared.loc[name, "smearing_factor"])
    except (KeyError, FileNotFoundError):
        raise SystemExit(f"[ERROR] factor de smearing no encontrado para {name}; abortado "
                         f"para no generar mapas con retransformacion naive")
    pred_log = merged["y_pred_log"].values
    pred_orig = np.exp(pred_log) * s
    preds[name] = {"log": pred_log, "orig": pred_orig,
                   "err_rel": np.abs(pred_orig - y_obs_orig) / y_obs_orig * 100,
                   "resid_log": y_obs_log - pred_log}
    rmse = np.sqrt(np.nanmean((y_obs_orig - pred_orig)**2))
    print(f"  {name:8s} RMSE={rmse:.1f}  s={s:.3f}")

x = gdf_te["x"].values
y = gdf_te["y"].values
XLIM = np.percentile(x, [1, 99]) + np.array([-0.9, 0.9])
YLIM = np.percentile(y, [1, 99]) + np.array([-0.9, 0.9])

# Rango común de valores para fig7a (escala log)
all_vals = np.concatenate([preds[n]["orig"] for n,_,_ in MODELS] + [y_obs_orig])
vmin_val = max(np.nanpercentile(all_vals, 1), 5.0)
vmax_val = np.nanpercentile(all_vals, 99)

# ── fig7a — Predicciones ─────────────────────────────────────────────────────
print("\nGenerando fig7a (predicciones) ...")
fig, axes = plt.subplots(1, 5, figsize=(15.5, 4.6))
axes_flat = axes.flatten()

norm_val = LogNorm(vmin=vmin_val, vmax=vmax_val)

for ax, (name, _, color) in zip(axes_flat, MODELS):
    rmse = np.sqrt(np.nanmean((y_obs_orig - preds[name]["orig"])**2))
    sc = ax.scatter(x, y, c=preds[name]["orig"], s=6, cmap="plasma",
                    norm=norm_val, alpha=0.88, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(f"{lbl(name)}  ·  RMSE={rmse:.1f}", fontsize=10, fontweight="bold",
                 color=color, pad=4)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    _ultimo = sc

_ETIQ = "USD/m²"
plt.tight_layout(w_pad=0.5, rect=[0, 0.11, 1, 1])
cax = fig.add_axes([0.34, 0.055, 0.32, 0.032])
cb = fig.colorbar(_ultimo, cax=cax, orientation="horizontal")
cb.set_label(_ETIQ, fontsize=10)
cb.ax.tick_params(labelsize=8.5)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig7a_predictions.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  -> fig7a_predictions.png/pdf")

# ── fig7b — Errores relativos ─────────────────────────────────────────────────
print("Generando fig7b (errores) ...")
fig, axes = plt.subplots(1, 5, figsize=(15.5, 4.6))
axes_flat = axes.flatten()

for ax, (name, _, color) in zip(axes_flat, MODELS):
    err = np.clip(preds[name]["err_rel"], 0, 100)
    p50 = np.nanpercentile(err, 50)
    sc = ax.scatter(x, y, c=err, s=6, cmap="RdYlGn_r",
                    vmin=0, vmax=80, alpha=0.88, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(f"{lbl(name)}  ·  p50={p50:.0f}% error",
                 fontsize=10, fontweight="bold", color=color, pad=4)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    _ultimo = sc

_ETIQ = "|error relativo| (%)"
plt.tight_layout(w_pad=0.5, rect=[0, 0.11, 1, 1])
cax = fig.add_axes([0.34, 0.055, 0.32, 0.032])
cb = fig.colorbar(_ultimo, cax=cax, orientation="horizontal")
cb.set_label(_ETIQ, fontsize=10)
cb.ax.tick_params(labelsize=8.5)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig7b_errors.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  -> fig7b_errors.png/pdf")

# ── fig7c — Residuos log-escala ───────────────────────────────────────────────
print("Generando fig7c (residuos log) ...")
fig, axes = plt.subplots(1, 5, figsize=(15.5, 4.6))
axes_flat = axes.flatten()

for ax, (name, _, color) in zip(axes_flat, MODELS):
    r = preds[name]["resid_log"]
    p5, p95 = np.nanpercentile(r, 5), np.nanpercentile(r, 95)
    vlim = max(abs(p5), abs(p95), 0.3)
    norm_r = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    sc = ax.scatter(x, y, c=r, s=6, cmap="RdBu",
                    norm=norm_r, alpha=0.88, edgecolors="none")
    ax.set_aspect("equal")
    # Random Forest se evalua en un pipeline separado y no figura en
    # comparativo_moran.csv; su indice se calcula aparte con el mismo criterio
    # (residuos log del test, k=8 vecinos, 999 permutaciones) y coincide con el
    # reportado en la tabla de Moran del Capitulo 5.
    MORAN_APARTE = {"RF": 0.0555}
    moran_str = ""
    try:
        mo = pd.read_csv(ROOT/"analisis"/"output_log"/"comparativo_moran.csv")
        mo_h = mo[(mo["modelo_display"]==name) & (mo["estrategia"]=="Holdout20%")]
        if len(mo_h):
            moran_str = f"  I={float(mo_h['I'].iloc[0]):.3f}"
        elif name in MORAN_APARTE:
            moran_str = f"  I={MORAN_APARTE[name]:.3f}"
    except Exception:
        pass
    ax.set_title(f"{lbl(name)}{moran_str}", fontsize=10, fontweight="bold", color=color, pad=4)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    _ultimo = sc

_ETIQ = "residuo log (obs−pred)"
plt.tight_layout(w_pad=0.5, rect=[0, 0.11, 1, 1])
cax = fig.add_axes([0.34, 0.055, 0.32, 0.032])
cb = fig.colorbar(_ultimo, cax=cax, orientation="horizontal")
cb.set_label(_ETIQ, fontsize=10)
cb.ax.tick_params(labelsize=8.5)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig7c_residuals.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  -> fig7c_residuals.png/pdf")

print(f"\n[OK] 3 figuras guardadas en {OUT}")
