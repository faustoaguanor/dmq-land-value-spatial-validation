"""
baselines_tabulares_replicas.py — Replicas multi-seed de RF/HGB (refuerzo de auditoria 2026-07-11).
=====================================================================================================
baselines_tabulares.py evalua RF/HGB con una sola semilla (random_state=42). Dos auditorias
independientes (aplicada y de codigo/resultados) senalaron que esto es una asimetria frente a
los modelos neurales, que reportan media +/- sigma sobre 10 semillas. Este script cierra esa
brecha SIN reentrenar redes: repite exactamente el mismo protocolo de baselines_tabulares.py
(mismas 27 covariables one-hot, mismos splits/folds, smearing fold-especifico, 4 esquemas)
variando unicamente random_state sobre las 10 semillas canonicas del proyecto (ver CLAUDE.md).

No sobrescribe outputs de baselines_tabulares.py (run base seed=42 se mantiene como referencia
principal); escribe a output_log_replicas/.

Salidas:
  - baseline_replicas_fold.csv       : un registro por modelo x estrategia x seed x fold (+holdout)
  - baseline_replicas_summary_fold.csv: media +/- sd ENTRE SEEDS, por modelo x estrategia x fold
  - baseline_replicas_summary.csv     : media +/- sd ENTRE SEEDS de la media-entre-folds por seed
                                        (mismo estimando que la Tabla 2 de la tesis para redes)
  - baseline_replicas_holdout_summary.csv: media +/- sd entre 10 seeds en holdout
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "spatial_cv"))
sys.path.insert(0, str(ROOT / "modelos"))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV, SpatialBlockBufferedCV
from features import build_feature_matrix

SEEDS = [42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7]  # canonicas, ver CLAUDE.md
BUFFER_M = 2530
OUT = Path(__file__).parent / "output_log_replicas"; OUT.mkdir(parents=True, exist_ok=True)

def metrics(y_ori, pred_log, y_log, s_M):
    yp = np.exp(pred_log) * s_M; e = y_ori - yp
    mae = float(np.mean(np.abs(e))); rmse = float(np.sqrt(np.mean(e**2)))
    mape = float(np.mean(np.abs(e/y_ori))*100)
    r2 = round(1 - np.sum(e**2)/np.sum((y_ori-y_ori.mean())**2), 4)
    el = y_log - pred_log; r2l = round(1 - np.sum(el**2)/np.sum((y_log-y_log.mean())**2), 4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l,"smearing_factor":round(s_M,6)}

def make(name, seed):
    if name == "RF":
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                         max_depth=None, random_state=seed)

# datos (identico a baselines_tabulares.py)
gdf = gpd.read_file(ROOT/"datos"/"dataset.gpkg", layer="puntos_mercado").to_crs(epsg=32717)
sp = pd.read_csv(ROOT/"data_split"/"split.csv"); fo = pd.read_csv(ROOT/"spatial_cv"/"output"/"fold_assignments.csv")
for df in (gdf, sp, fo): df["predio_join"] = df["predio_join"].astype(int)
gdf = gdf.merge(fo[["predio_join","fold"]], on="predio_join", how="left").merge(sp[["predio_join","split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_ori = gdf["valor_m2"].values.astype(float); y_log = np.log(y_ori)
X, _ = build_feature_matrix(gdf)
tm = (gdf["split"]=="train").values; te_m = (gdf["split"]=="test").values
sp_folds = gdf["fold"].values.astype(int)

cv_r = list(RandomKFoldCV(n_splits=5, random_state=42).split(X[tm]))
cv_b = list(SpatialBlockCV(folds=sp_folds[tm]).split(X[tm]))
cv_bb = list(SpatialBlockBufferedCV(folds=sp_folds[tm], coords=coords[tm], buffer=BUFFER_M).split(X[tm]))
# folds identicos a baselines_tabulares.py (no dependen de random_state del modelo)

fold_recs = []
holdout_recs = []
t0 = time.time()
for name in ["RF", "HGB"]:
    for seed in SEEDS:
        for strat, splits in [("RandomKFold",cv_r),("SpatialBlock",cv_b),("SpatialBlock_buf",cv_bb)]:
            for fid,(tr,te) in enumerate(splits):
                Xtr, Xte = X[tm][tr], X[tm][te]
                ytr, ytel, yteo = y_log[tm][tr], y_log[tm][te], y_ori[tm][te]
                m = make(name, seed).fit(Xtr, ytr)
                s_M = float(np.mean(np.exp(ytr - m.predict(Xtr))))
                fold_recs.append({"modelo":name,"estrategia":strat,"seed":seed,"fold":fid,
                                   **metrics(yteo, m.predict(Xte), ytel, s_M)})
        # holdout
        mh = make(name, seed).fit(X[tm], y_log[tm])
        s_M = float(np.mean(np.exp(y_log[tm] - mh.predict(X[tm]))))
        pred_te = mh.predict(X[te_m])
        holdout_recs.append({"modelo":name,"estrategia":"Holdout20%","seed":seed,"n_test":int(te_m.sum()),
                              **metrics(y_ori[te_m], pred_te, y_log[te_m], s_M)})
        print(f"[{time.time()-t0:6.1f}s] {name} seed={seed} listo")

df_fold = pd.DataFrame(fold_recs)
df_hold = pd.DataFrame(holdout_recs)
df_fold.to_csv(OUT/"baseline_replicas_fold.csv", index=False)
df_hold.to_csv(OUT/"baseline_replicas_holdout_fold.csv", index=False)

# resumen por fold (SD ENTRE SEEDS, misma geometria)
summary_fold = (df_fold.groupby(["modelo","estrategia","fold"])
                 .agg(MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
                      RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
                      R2_mean=("R2","mean"), R2_std=("R2","std"),
                      n_seeds=("MAE","count"))
                 .reset_index())
summary_fold.to_csv(OUT/"baseline_replicas_summary_fold.csv", index=False)

# resumen estilo Tabla 2 de la tesis: media-entre-folds por seed, luego media+-sd ENTRE SEEDS
per_seed = (df_fold.groupby(["modelo","estrategia","seed"])
            .agg(MAE=("MAE","mean"), RMSE=("RMSE","mean"), R2=("R2","mean"))
            .reset_index())
summary = (per_seed.groupby(["modelo","estrategia"])
           .agg(MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
                RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
                R2_mean=("R2","mean"), R2_std=("R2","std"),
                n_seeds=("MAE","count"))
           .reset_index())
summary.to_csv(OUT/"baseline_replicas_summary.csv", index=False)

holdout_summary = (df_hold.groupby("modelo")
                    .agg(MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
                         RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
                         R2_mean=("R2","mean"), R2_std=("R2","std"),
                         n_seeds=("MAE","count"))
                    .reset_index())
holdout_summary.to_csv(OUT/"baseline_replicas_holdout_summary.csv", index=False)

print("\n=== Resumen holdout (10 seeds) ===")
print(holdout_summary.to_string(index=False))
print("\n=== Resumen por esquema (media+-sd entre seeds, media-entre-folds por seed) ===")
print(summary.to_string(index=False))
print(f"\n[CSV] {OUT}")
