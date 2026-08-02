"""
sensibilidad_buffer.py — Sensibilidad del ranking al radio del buffer.
======================================================================
Repite el CV buffered (OLS y GWR-27) con una grilla de buffers, usando EXACTAMENTE el
mismo estimador GWR que el modelo principal: implementacion canonica importada de
`modelos/gwr/gwr_core.py` (intercepto NO penalizado, lambda por CV espacial anidado
dentro de cada training fold). Corrige el hallazgo #2 de la 3a auditoria (antes este
script usaba lambda=0.1 con intercepto penalizado, distinto del modelo defendido).

Los modelos neurales no se incluyen por costo de GPU (ver §8.5).
Salida: analisis/output_log/sensibilidad_buffer.csv
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "spatial_cv"))
sys.path.insert(0, str(BASE / "modelos"))
sys.path.insert(0, str(BASE / "modelos" / "gwr"))
from estrategias_cv import SpatialBlockBufferedCV
from features import build_feature_matrix
from gwr_core import select_bw, select_lambda, predict_gwr, add_intercept   # GWR canonico unico
warnings.filterwarnings("ignore")

BUFFERS_M = [1500, 2000, 2530, 3000, 4000]
DATA  = BASE / "datos" / "dataset.gpkg"
SPLIT = BASE / "data_split" / "split.csv"
FOLDS = BASE / "spatial_cv" / "output" / "fold_assignments.csv"
OUT   = BASE / "analisis" / "output_log" / "sensibilidad_buffer.csv"

gdf = gpd.read_file(DATA, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT); folds_df = pd.read_csv(FOLDS)
for df in (gdf, split_df, folds_df):
    df["predio_join"] = df["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join", "fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join", "split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_orig = gdf["valor_m2"].values.astype(float); y_log = np.log(y_orig)
X_oh,  _ = build_feature_matrix(gdf)
X_cont, _ = build_feature_matrix(gdf, one_hot=False)
train_mask = (gdf["split"] == "train").values
sp_folds = gdf["fold"].values.astype(int)

rows = []
for buf in BUFFERS_M:
    t0 = time.time()
    cv = SpatialBlockBufferedCV(folds=sp_folds[train_mask], coords=coords[train_mask], buffer=buf)
    splits = list(cv.split(X_oh[train_mask]))
    gwr_mae, ols_mae, n_trains, lams = [], [], [], []
    for fid, (tr, te) in enumerate(splits):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_oh[train_mask][tr]); Xte = sc.transform(X_oh[train_mask][te])
        ytr_log = y_log[train_mask][tr]; yte_ori = y_orig[train_mask][te]
        ctr = coords[train_mask][tr]; cte = coords[train_mask][te]
        n_trains.append(len(tr))
        # OLS global
        ols = LinearRegression().fit(Xtr, ytr_log)
        ols_mae.append(float(np.mean(np.abs(yte_ori - np.exp(ols.predict(Xte))))))
        # GWR canonico: bw (continuo) + lambda nested (one-hot), intercepto no penalizado
        Xtr_i = add_intercept(Xtr); Xte_i = add_intercept(Xte)
        bw = select_bw(ctr, add_intercept(StandardScaler().fit_transform(X_cont[train_mask][tr])), ytr_log)
        lam, _ = select_lambda(ctr, Xtr_i, ytr_log, bw)
        pred_log, _ = predict_gwr(ctr, Xtr_i, ytr_log, cte, Xte_i, bw, lam=lam)
        gwr_mae.append(float(np.mean(np.abs(yte_ori - np.exp(pred_log)))))
        lams.append(lam)
    rows.append({
        "buffer_m": buf, "n_folds": len(splits), "n_train_mediana": float(np.median(n_trains)),
        "OLS_MAE_mediana_folds": round(float(np.median(ols_mae)), 2),
        "GWR27_MAE_mediana_folds": round(float(np.median(gwr_mae)), 2),
        "GWR27_MAE_media_folds": round(float(np.mean(gwr_mae)), 2),
        "lambdas_por_fold": str(lams),
    })
    print(f"buffer={buf:5d}m  n_tr_med={np.median(n_trains):.0f}  OLS={rows[-1]['OLS_MAE_mediana_folds']:.1f}  "
          f"GWR(med)={rows[-1]['GWR27_MAE_mediana_folds']:.1f}  lambdas={lams}  ({time.time()-t0:.0f}s)", flush=True)

out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
print(f"\n[OK] {OUT}"); print(out.to_string(index=False))
