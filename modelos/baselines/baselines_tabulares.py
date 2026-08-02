"""
baselines_tabulares.py — Baselines tabulares modernos (hallazgo A10 de la 4a auditoria).
=========================================================================================
Random Forest y Gradient Boosting (HistGradientBoostingRegressor, boosting moderno tipo
LightGBM en sklearn) bajo EXACTAMENTE el mismo protocolo que el resto de modelos: mismas
27 covariables one-hot (30 features), mismos split/folds, smearing de Duan FOLD-ESPECIFICO,
y los 4 esquemas (Holdout, RandomKFold, SpatialBlock, SpatialBlock_buf).

Motivacion: el MLP no es un baseline tabular fuerte. Sin arboles/boosting no puede decidirse
si la (in)utilidad de las redes espaciales se debe a la estructura espacial o a la ausencia
de un competidor no-lineal adecuado. Estos baselines NO usan coordenadas (son no-espaciales),
igual que el MLP: sirven para acotar cuanto del desempeno es no-linealidad tabular pura.

Hiperparametros: defaults razonables (sin tuning exhaustivo), coherente con que el resto de
modelos usa la configuracion de sus papers; no se optimizan mirando el test.

Salidas: modelos/baselines/output_log/{rf,hgb}_log_{results,holdout,predictions}.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "spatial_cv"))
sys.path.insert(0, str(ROOT / "modelos"))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV, SpatialBlockBufferedCV
from features import build_feature_matrix

RANDOM_STATE = 42
BUFFER_M = 2530
OUT = Path(__file__).parent / "output_log"; OUT.mkdir(parents=True, exist_ok=True)

def metrics(y_ori, pred_log, y_log, s_M):
    yp = np.exp(pred_log) * s_M; e = y_ori - yp
    mae = float(np.mean(np.abs(e))); rmse = float(np.sqrt(np.mean(e**2)))
    mape = float(np.mean(np.abs(e/y_ori))*100)
    r2 = round(1 - np.sum(e**2)/np.sum((y_ori-y_ori.mean())**2), 4)
    el = y_log - pred_log; r2l = round(1 - np.sum(el**2)/np.sum((y_log-y_log.mean())**2), 4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l,"smearing_factor":round(s_M,6)}

def make(name):
    if name == "RF":
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                         max_depth=None, random_state=RANDOM_STATE)

# datos
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

cv_r = list(RandomKFoldCV(n_splits=5, random_state=RANDOM_STATE).split(X[tm]))
cv_b = list(SpatialBlockCV(folds=sp_folds[tm]).split(X[tm]))
cv_bb = list(SpatialBlockBufferedCV(folds=sp_folds[tm], coords=coords[tm], buffer=BUFFER_M).split(X[tm]))

for name in ["RF", "HGB"]:
    recs = []
    for strat, splits in [("RandomKFold",cv_r),("SpatialBlock",cv_b),("SpatialBlock_buf",cv_bb)]:
        for fid,(tr,te) in enumerate(splits):
            Xtr, Xte = X[tm][tr], X[tm][te]
            ytr, ytel, yteo = y_log[tm][tr], y_log[tm][te], y_ori[tm][te]
            m = make(name).fit(Xtr, ytr)
            s_M = float(np.mean(np.exp(ytr - m.predict(Xtr))))   # smearing fold-especifico
            recs.append({"modelo":name,"estrategia":strat,"fold":fid, **metrics(yteo, m.predict(Xte), ytel, s_M)})
    # holdout final
    mh = make(name).fit(X[tm], y_log[tm])
    s_M = float(np.mean(np.exp(y_log[tm] - mh.predict(X[tm]))))
    pred_te = mh.predict(X[te_m])
    hold = {"modelo":name,"estrategia":"Holdout20%","n_test":int(te_m.sum()), **metrics(y_ori[te_m], pred_te, y_log[te_m], s_M)}
    pd.DataFrame(recs).to_csv(OUT/f"{name.lower()}_log_results.csv", index=False)
    pd.DataFrame([hold]).to_csv(OUT/f"{name.lower()}_log_holdout.csv", index=False)
    # predictions (test+train) por si se necesitan
    pred_tr = mh.predict(X[tm])
    pd.concat([
        pd.DataFrame({"predio_join":gdf.loc[te_m,"predio_join"].astype(int).values,"split":"test","y_obs_log":y_log[te_m],"y_pred_log":pred_te}),
        pd.DataFrame({"predio_join":gdf.loc[tm,"predio_join"].astype(int).values,"split":"train","y_obs_log":y_log[tm],"y_pred_log":pred_tr}),
    ], ignore_index=True).assign(modelo=name).to_csv(OUT/f"{name.lower()}_log_predictions.csv", index=False)
    print(f"=== {name} ===  holdout: MAE={hold['MAE']:.1f} RMSE={hold['RMSE']:.1f} R2={hold['R2']:.3f}")
    r = pd.DataFrame(recs)
    for est in ["RandomKFold","SpatialBlock","SpatialBlock_buf"]:
        s = r[r.estrategia==est]
        print(f"   {est}: MAE media={s.MAE.mean():.1f} mediana={np.median(s.MAE):.1f}")
print(f"\n[CSV] {OUT}")
