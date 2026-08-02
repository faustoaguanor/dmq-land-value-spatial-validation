"""
ols_log.py
==========
OLS con log(valor_m2) como target, holdout 80/20 espacialmente estratificado,
y Moran I sobre residuos OOS del CV (no in-sample).

Salidas
-------
  output_log/ols_log_results.csv   — metricas por fold (escala original + log)
  output_log/ols_log_summary.csv   — media/std por estrategia
  output_log/ols_log_holdout.csv   — metricas finales sobre test 20%
  output_log/ols_log_moran.csv     — Moran I OOS (CV + holdout)
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

try:
    from libpysal.weights import KNN as KNNWeights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

_CV_DIR = Path(__file__).parent.parent.parent / "spatial_cv"
sys.path.insert(0, str(_CV_DIR))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV, SpatialBlockBufferedCV

# one-hot de uso_suelo_cod via punto unico de verdad (modelos/features.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import build_feature_matrix

# ── Config ────────────────────────────────────────────────────────────────────
COVARIABLES = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]
MORAN_K      = 8
RANDOM_STATE = 42
BASE_DIR     = Path(__file__).parent
DATA_PATH    = BASE_DIR.parent.parent / "datos" / "dataset.gpkg"
SPLIT_PATH   = BASE_DIR.parent.parent / "data_split" / "split.csv"
FOLDS_PATH   = BASE_DIR.parent.parent / "spatial_cv" / "output" / "fold_assignments.csv"
OUT_DIR      = BASE_DIR / "output_log"
OUT_DIR.mkdir(exist_ok=True)

# ── Metricas en escala ORIGINAL (USD/m²) ─────────────────────────────────────
def compute_metrics(y_true_orig: np.ndarray, y_pred_orig: np.ndarray,
                    y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    e_orig = y_true_orig - y_pred_orig
    e_log  = y_true_log  - y_pred_log
    mae    = float(np.mean(np.abs(e_orig)))
    rmse   = float(np.sqrt(np.mean(e_orig**2)))
    mape   = float(np.mean(np.abs(e_orig / y_true_orig)) * 100)
    ss_res = float(np.sum(e_orig**2))
    ss_tot = float(np.sum((y_true_orig - y_true_orig.mean())**2))
    r2_orig = round(1 - ss_res/ss_tot, 4) if ss_tot > 0 else float("nan")
    ss_r_log = float(np.sum(e_log**2))
    ss_t_log = float(np.sum((y_true_log - y_true_log.mean())**2))
    r2_log  = round(1 - ss_r_log/ss_t_log, 4) if ss_t_log > 0 else float("nan")
    return {"MAE": round(mae,4), "RMSE": round(rmse,4),
            "MAPE": round(mape,4), "R2": r2_orig, "R2_log": r2_log}

def compute_moran_i(values, coords, k=MORAN_K):
    w = KNNWeights.from_array(coords, k=k); w.transform = "r"
    z = values - values.mean(); Wz = w.sparse @ z
    return float((z@Wz)/(z@z)), -1/(len(values)-1)

# ── Cargar datos ──────────────────────────────────────────────────────────────
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT_PATH)
folds_df = pd.read_csv(FOLDS_PATH)
folds_df["predio_join"] = folds_df["predio_join"].astype(int)
split_df["predio_join"] = split_df["predio_join"].astype(int)
gdf["predio_join"]      = gdf["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join","fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")

gdf = gdf.sort_values("predio_join").reset_index(drop=True)  # orden determinista + incluye predio con cero a la izquierda
coords    = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_orig    = gdf["valor_m2"].values.astype(float)
y_log     = np.log(y_orig)
X_raw, FEAT_NAMES = build_feature_matrix(gdf)   # one-hot uso_suelo_cod (27->31)
sp_folds  = gdf["fold"].values.astype(int)

train_mask = (gdf["split"] == "train").values
test_mask  = (gdf["split"] == "test").values
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]

print(f"n_total={len(gdf)}  train={len(train_idx)}  test={len(test_idx)}")

# ── CV dentro del 80% train ───────────────────────────────────────────────────
cv_rand  = RandomKFoldCV(n_splits=5, random_state=RANDOM_STATE)
cv_block = SpatialBlockCV(folds=sp_folds[train_mask])
splits_r = list(cv_rand.split(X_raw[train_mask], y_log[train_mask]))
splits_b = list(cv_block.split(X_raw[train_mask], y_log[train_mask]))
BUFFER_M = 2530  # rango residual exponencial (~2.5 km): SpatialBlock con separacion train-test garantizada (Roberts et al. 2017)
cv_block_buf = SpatialBlockBufferedCV(folds=sp_folds[train_mask], coords=coords[train_mask], buffer=BUFFER_M)
splits_bb = list(cv_block_buf.split(X_raw[train_mask]))

all_records = []
moran_records = []

for strat_name, splits in [("RandomKFold", splits_r), ("SpatialBlock", splits_b), ("SpatialBlock_buf", splits_bb)]:
    print(f"\n[OLS / {strat_name}]")
    y_pred_oos = np.full(len(train_idx), np.nan)

    for fold_id, (tr, te) in enumerate(splits):
        scaler  = StandardScaler()
        Xtr = scaler.fit_transform(X_raw[train_mask][tr])
        Xte = scaler.transform(X_raw[train_mask][te])
        ytr_log = y_log[train_mask][tr]
        yte_log = y_log[train_mask][te]
        yte_ori = y_orig[train_mask][te]

        model = LinearRegression().fit(Xtr, ytr_log)
        pred_log = model.predict(Xte)
        # smearing de Duan FOLD-ESPECIFICO (#7): residuales in-sample del train del fold
        s_M = float(np.mean(np.exp(ytr_log - model.predict(Xtr))))
        pred_ori = np.exp(pred_log) * s_M

        y_pred_oos[te] = pred_log
        m = compute_metrics(yte_ori, pred_ori, yte_log, pred_log)
        print(f"  fold {fold_id+1}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}  R2_log={m['R2_log']:.4f}")
        all_records.append({"modelo":"OLS","estrategia":strat_name,
                            "fold":fold_id, **m})

    # Moran I sobre residuos OOS del CV
    if LIBPYSAL_AVAILABLE:
        mask_oos = ~np.isnan(y_pred_oos)
        resid_oos = y_log[train_mask][mask_oos] - y_pred_oos[mask_oos]
        I, EI = compute_moran_i(resid_oos, coords[train_mask][mask_oos])
        print(f"  Moran I OOS {strat_name}: I={I:.4f}  E[I]={EI:.6f}")
        moran_records.append({"modelo":"OLS","estrategia":strat_name,
                              "I":round(I,6),"EI":round(EI,6),"tipo":"CV_OOS"})

# ── Modelo final: entrenar en 100% train, evaluar en test ─────────────────────
print("\n[OLS / Holdout final — entrena en 80%, evalua en 20%]")
scaler_final = StandardScaler()
X_train_sc   = scaler_final.fit_transform(X_raw[train_mask])
X_test_sc    = scaler_final.transform(X_raw[test_mask])

model_final  = LinearRegression().fit(X_train_sc, y_log[train_mask])
pred_log_test = model_final.predict(X_test_sc)
pred_ori_test = np.exp(pred_log_test)

m_test = compute_metrics(y_orig[test_mask], pred_ori_test,
                         y_log[test_mask], pred_log_test)
print(f"  HOLDOUT  MAE={m_test['MAE']:.2f}  RMSE={m_test['RMSE']:.2f}  "
      f"MAPE={m_test['MAPE']:.2f}%  R2={m_test['R2']:.4f}  R2_log={m_test['R2_log']:.4f}")

holdout_row = {"modelo":"OLS","estrategia":"Holdout20%",
               "n_test":len(test_idx), **m_test}

# Moran I sobre residuos del holdout
if LIBPYSAL_AVAILABLE:
    resid_test = y_log[test_mask] - pred_log_test
    I_h, EI_h = compute_moran_i(resid_test, coords[test_mask])
    print(f"  Moran I Holdout: I={I_h:.4f}  E[I]={EI_h:.6f}")
    moran_records.append({"modelo":"OLS","estrategia":"Holdout20%",
                          "I":round(I_h,6),"EI":round(EI_h,6),"tipo":"Holdout"})

# ── Guardar ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_records)
summary_df = results_df.groupby(["modelo","estrategia"]).agg(
    MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
    RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
    MAPE_mean=("MAPE","mean"), MAPE_std=("MAPE","std"),
    R2_mean=("R2","mean"), R2_std=("R2","std"),
).reset_index().round(4)

results_df.to_csv(OUT_DIR / "ols_log_results.csv", index=False)
summary_df.to_csv(OUT_DIR / "ols_log_summary.csv", index=False)
pd.DataFrame([holdout_row]).to_csv(OUT_DIR / "ols_log_holdout.csv", index=False)

# Predicciones por predio (test + train) para smearing / bootstrap / CDF / mapa
pred_log_train = model_final.predict(X_train_sc)
preds = pd.concat([
    pd.DataFrame({"predio_join": gdf.loc[test_mask,"predio_join"].astype(int).values,
                  "split":"test", "y_obs_log":y_log[test_mask], "y_pred_log":pred_log_test}),
    pd.DataFrame({"predio_join": gdf.loc[train_mask,"predio_join"].astype(int).values,
                  "split":"train","y_obs_log":y_log[train_mask],"y_pred_log":pred_log_train}),
], ignore_index=True)
preds["modelo"] = "OLS"
preds.to_csv(OUT_DIR / "ols_log_predictions.csv", index=False)
if moran_records:
    pd.DataFrame(moran_records).to_csv(OUT_DIR / "ols_log_moran.csv", index=False)

print("\n" + "="*60)
print("RESUMEN OLS log(y)")
print("="*60)
print(summary_df.to_string(index=False))
print(f"\nHOLDOUT: {holdout_row}")
print(f"\n[CSV] {OUT_DIR}")

if __name__ == "__main__":
    pass
