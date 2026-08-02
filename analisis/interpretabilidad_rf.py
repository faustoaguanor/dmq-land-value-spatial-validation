"""
interpretabilidad_rf.py
========================
Permutation importance para Random Forest (control tabular y modelo recomendado
para interpolacion, Tabla 7).

Motivacion: RF es el modelo de menor error en interpolacion y el recomendado para
zonas con soporte muestral, pero la Tabla 4 de la tesis solo reportaba importancia
para los modelos neurales. Sin este analisis, el modelo que la tesis recomienda
operativamente era el unico sin evidencia de interpretabilidad.

Metodo (identico al de los modelos neurales, para que las columnas sean comparables):
  1. Re-entrena RF seed=42 sobre el train del holdout (mismo protocolo que
     baselines_tabulares.py: 27 covariables one-hot, sin coordenadas)
  2. RMSE_base sobre el holdout 20% en USD/m2, con smearing de Duan
  3. Para cada variable k: baraja sus valores en test, mide degradacion del RMSE
  4. 5 repeticiones con semillas distintas -> media +/- sd

Salidas: analisis/output_log/
  interp_rf_permutation_importance.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix

OUT_DIR = ROOT / "analisis" / "output_log"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_REPEATS = 5

# --- datos (mismo pipeline que baselines_tabulares.py) ---
gdf = gpd.read_file(ROOT / "datos" / "dataset.gpkg", layer="puntos_mercado").to_crs(epsg=32717)
sp = pd.read_csv(ROOT / "data_split" / "split.csv")
for df in (gdf, sp):
    df["predio_join"] = df["predio_join"].astype(int)   # contrato de clave canonica
gdf = gdf.merge(sp[["predio_join", "split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)

y_ori = gdf["valor_m2"].values.astype(float)
y_log = np.log(y_ori)
X, feat_names = build_feature_matrix(gdf)

tr_m = (gdf["split"] == "train").values
te_m = (gdf["split"] == "test").values
Xtr, Xte = X[tr_m], X[te_m]
ytr_log = y_log[tr_m]
yte_ori = y_ori[te_m]

print(f"train={tr_m.sum()}  test={te_m.sum()}  features={X.shape[1]}")

# --- entrenamiento + smearing de Duan sobre el train ---
model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)
model.fit(Xtr, ytr_log)
s_M = float(np.mean(np.exp(ytr_log - model.predict(Xtr))))


def rmse_usd(Xm):
    pred = np.exp(model.predict(Xm)) * s_M
    return float(np.sqrt(np.mean((yte_ori - pred) ** 2)))


rmse_base = rmse_usd(Xte)
print(f"smearing s_M={s_M:.6f}   RMSE_base={rmse_base:.4f} USD/m2")

# --- permutation importance ---
recs = []
for j, name in enumerate(feat_names):
    deltas, rmses = [], []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(RANDOM_STATE + rep)
        Xp = Xte.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        r = rmse_usd(Xp)
        rmses.append(r)
        deltas.append(r - rmse_base)
    recs.append({
        "variable": name,
        "importance_mean": round(float(np.mean(deltas)), 4),
        "importance_std": round(float(np.std(deltas)), 4),
        "rmse_perm_mean": round(float(np.mean(rmses)), 4),
    })

out = pd.DataFrame(recs).sort_values("importance_mean", ascending=False).reset_index(drop=True)
out.to_csv(OUT_DIR / "interp_rf_permutation_importance.csv", index=False)

print(f"\nRMSE base = {rmse_base:.2f} USD/m2  (smearing {s_M:.4f})")
print("\nTop-10 permutation importance (RF, holdout seed=42):")
print(out.head(10).to_string(index=False))
print(f"\n-> {OUT_DIR/'interp_rf_permutation_importance.csv'}")
