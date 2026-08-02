"""
tost_equivalencia.py
=====================
Prueba de equivalencia práctica (TOST) entre los modelos focales.

Problema metodológico: TOST exige un margen de equivalencia PREESPECIFICADO. Fijarlo
ahora, con los resultados ya a la vista, sería exactamente la práctica que esta tesis
evita. Por eso NO se fija un margen a dedo: se reporta el **margen mínimo de equivalencia**
Delta_min, es decir, el menor margen para el cual TOST declararía equivalencia al 5%.
Delta_min es el borde exterior del intervalo de confianza al 90% de la diferencia pareada
(la equivalencia TOST al alfa=0.05 equivale a que el IC90 caiga dentro de +/-Delta).
Asi el lector juzga si Delta_min es practicamente irrelevante, sin que el autor haya
elegido el umbral despues de ver los datos.

Dos niveles:
  (A) Holdout, n=1,011 predios pareados. Diferencia del error absoluto por predio, en USD/m2.
      IC90 por t pareado y por bootstrap de BLOQUES ESPACIALES de 2,530 m (la tesis ya
      documenta que el bootstrap i.i.d. subestima la incertidumbre 2.0-2.6x).
  (B) SpatialBlock, n=5 folds como unidad (la unidad pertinente para generalizacion).
      Se espera potencia muy baja: el resultado esperable es que NI diferencia NI
      equivalencia puedan establecerse, lo que es en si mismo el hallazgo.

Salidas: analisis/output_log/
  tost_equivalencia_holdout.csv
  tost_equivalencia_spatialblock.csv
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analisis" / "output_log"
OUT.mkdir(exist_ok=True)

ALPHA = 0.05          # TOST al 5%  ->  IC al 90%
CELL_M = 2530         # celda del bootstrap espacial (= buffer de SpatialBlock_buf)
N_BOOT = 2000
RNG = np.random.default_rng(42)

# modelo -> (archivo de predicciones holdout, factor de smearing, archivo de folds)
MODELOS = {
    "OLS":    ("modelos/ols/output_log/ols_log_predictions.csv",                  1.089736, "modelos/ols/output_log/ols_log_results.csv"),
    "GWR":    ("modelos/gwr/output_log_27vars/gwr27_log_predictions.csv",         1.026509, "modelos/gwr/output_log_27vars/gwr27_log_results.csv"),
    "GNNWR":  ("modelos/gnnwr/output_log/gnnwr_log_predictions.csv",              1.031205, "modelos/gnnwr/output_log/gnnwr_log_results.csv"),
    "SANNWR": ("modelos/sannwr/output_log_real/sannwr_real_log_predictions.csv",  0.997983, "modelos/sannwr/output_log_real/sannwr_real_log_results.csv"),
    "RF":     ("modelos/baselines/output_log/rf_log_predictions.csv",             1.003332, "modelos/baselines/output_log/rf_log_results.csv"),
    "MLP":    ("modelos/mlp/output_log/mlp_log_predictions.csv",                  0.989259, "modelos/mlp/output_log/mlp_log_results.csv"),
}

# ---------------- (A) holdout: error absoluto pareado por predio ----------------
gdf = gpd.read_file(ROOT / "datos" / "dataset.gpkg", layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"] = gdf["predio_join"].astype(int)     # contrato de clave canonica
geo = gdf[["predio_join"]].copy()
geo["x"] = gdf.geometry.x.values
geo["y"] = gdf.geometry.y.values

ae = {}
for name, (pred_f, s_M, _) in MODELOS.items():
    df = pd.read_csv(ROOT / pred_f)
    df = df[df["split"] == "test"].copy()
    df["predio_join"] = df["predio_join"].astype(int)
    y_obs = np.exp(df["y_obs_log"].values)
    y_hat = np.exp(df["y_pred_log"].values) * s_M
    s = pd.Series(np.abs(y_obs - y_hat), index=df["predio_join"].values)
    ae[name] = s[~s.index.duplicated()].sort_index()

common = None
for s in ae.values():
    common = s.index if common is None else common.intersection(s.index)
common = np.sort(np.asarray(common))
print(f"[A] holdout pareado: n={len(common)} predios comunes a los {len(MODELOS)} modelos")

AE = pd.DataFrame({k: v.reindex(common).values for k, v in ae.items()}, index=common)
g = geo.set_index("predio_join").reindex(common)
cell = (g["x"].values // CELL_M).astype(int).astype(str) + "_" + (g["y"].values // CELL_M).astype(int).astype(str)
cells, cell_idx = np.unique(cell, return_inverse=True)
groups = [np.where(cell_idx == k)[0] for k in range(len(cells))]
print(f"    bootstrap espacial: {len(cells)} celdas de {CELL_M} m")

def boot_ci_spatial(d):
    """IC90 por bootstrap de bloques espaciales (remuestrea celdas completas)."""
    means = np.empty(N_BOOT)
    n_cells = len(groups)
    for b in range(N_BOOT):
        pick = RNG.integers(0, n_cells, n_cells)
        idx = np.concatenate([groups[k] for k in pick])
        means[b] = d[idx].mean()
    return np.percentile(means, [5, 95])

rows = []
for a, b in combinations(MODELOS.keys(), 2):
    d = AE[a].values - AE[b].values          # >0 => a peor que b
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - ALPHA, n - 1)    # IC al 90%
    lo, hi = m - tcrit * se, m + tcrit * se
    slo, shi = boot_ci_spatial(d)
    t_stat, p_dif = stats.ttest_rel(AE[a].values, AE[b].values)
    rows.append({
        "modelo_A": a, "modelo_B": b, "n": n,
        "dif_MAE_media": round(m, 3),
        "IC90_iid_inf": round(lo, 3), "IC90_iid_sup": round(hi, 3),
        "delta_min_iid": round(max(abs(lo), abs(hi)), 3),
        "IC90_espacial_inf": round(slo, 3), "IC90_espacial_sup": round(shi, 3),
        "delta_min_espacial": round(max(abs(slo), abs(shi)), 3),
        "p_diferencia": round(float(p_dif), 5),
    })

hold = pd.DataFrame(rows).sort_values("delta_min_espacial").reset_index(drop=True)
hold.to_csv(OUT / "tost_equivalencia_holdout.csv", index=False)

# ---------------- (B) SpatialBlock: folds como unidad ----------------
fold_mae = {}
for name, (_, _, res_f) in MODELOS.items():
    df = pd.read_csv(ROOT / res_f)
    sb = df[df["estrategia"] == "SpatialBlock"].sort_values("fold")
    fold_mae[name] = sb["MAE"].values.astype(float)

rows_b = []
for a, b in combinations(MODELOS.keys(), 2):
    d = fold_mae[a] - fold_mae[b]
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - ALPHA, n - 1)
    lo, hi = m - tcrit * se, m + tcrit * se
    _, p_dif = stats.ttest_rel(fold_mae[a], fold_mae[b])
    rows_b.append({
        "modelo_A": a, "modelo_B": b, "n_regiones": n,
        "dif_MAE_media": round(m, 3),
        "IC90_inf": round(lo, 3), "IC90_sup": round(hi, 3),
        "delta_min": round(max(abs(lo), abs(hi)), 3),
        "p_diferencia": round(float(p_dif), 5),
    })

sb = pd.DataFrame(rows_b).sort_values("delta_min").reset_index(drop=True)
sb.to_csv(OUT / "tost_equivalencia_spatialblock.csv", index=False)

# ---------------- resumen ----------------
pd.set_option("display.width", 200)
print("\n=== (A) HOLDOUT: diferencia de MAE pareada, USD/m2 ===")
print(hold[["modelo_A", "modelo_B", "dif_MAE_media", "delta_min_iid", "delta_min_espacial", "p_diferencia"]].to_string(index=False))
print("\n=== (B) SPATIALBLOCK: 5 regiones como unidad ===")
print(sb[["modelo_A", "modelo_B", "dif_MAE_media", "delta_min", "p_diferencia"]].to_string(index=False))
print(f"\nDelta_min = menor margen con el que TOST declararia equivalencia (alfa={ALPHA}).")
print(f"Razon de anchura espacial/iid (mediana): {np.median(hold.delta_min_espacial/hold.delta_min_iid):.2f}x")
print(f"\n-> {OUT/'tost_equivalencia_holdout.csv'}\n-> {OUT/'tost_equivalencia_spatialblock.csv'}")
