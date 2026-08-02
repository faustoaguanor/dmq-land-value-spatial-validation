"""
gwr_log_27vars.py
=================
GWR con las 27 variables originales completas + Ridge lambda=0.1.
Compara con gwr_log.py (17 vars VIF-clean) para validar que la comparacion
con modelos neurales (que usan 27 vars) sea justa.

Ridge lambda=0.1 (vs lambda=1e-5 en gwr_log.py) compensa la multicolinealidad
sin eliminar variables — la regularizacion estabiliza (X'WX + lambdaI)^-1.

Prediccion GWR estandar (Fotheringham, Yang & Kang 2017, Eq. 5; Harris et al.
2010): en cada punto de test se resuelve beta(x0)=(X'W(x0)X+lamI)^-1 X'W(x0)y
con el kernel centrado en x0 sobre los datos de entrenamiento, y se predice
y_hat(x0)=x0'beta(x0). (Version previa interpolaba coeficientes por IDW-K15;
sustituida por el estimador GWR canonico.)

Moran I de residuos con p-valor por permutacion (999, Anselin 1995).

Target: log(valor_m2). Metricas en escala original USD/m2 via exp().
Holdout: 80/20 espacialmente estratificado.

Salidas: output_log_27vars/gwr27_log_*.csv
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_CV_DIR = Path(__file__).parent.parent.parent / "spatial_cv"
sys.path.insert(0, str(_CV_DIR))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV, SpatialBlockBufferedCV

# one-hot de uso_suelo_cod via punto unico de verdad (modelos/features.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import build_feature_matrix

try:
    from mgwr.sel_bw import Sel_BW
    MGWR_AVAILABLE = True
except ImportError:
    MGWR_AVAILABLE = False

try:
    from libpysal.weights import KNN as KNNWeights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# 27 vars completas — mismas que usan los modelos neurales
COVARIABLES = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]

# Ridge para compensar multicolinealidad con 27 vars.
# CORRECCION (2026-06-19): el intercepto NO se penaliza (estandar de Ridge; sklearn no lo
# penaliza) y lambda se SELECCIONA por fold mediante validacion interna (nested), en vez de
# fijarlo en 0.1. La version previa penalizaba el intercepto con lambda=0.1 fijo, lo que
# inflaba artificialmente el error de GWR bajo extrapolacion (penalizar el intercepto encoge
# el nivel local hacia 0 en escala log) y producia un "colapso" que era un artefacto de
# especificacion, no estructural. Ver analisis/verif_gwr_intercepto.py.
# GWR canonico: implementacion UNICA importada de gwr_core (corrige hallazgo #2 de la 3a
# auditoria: misma funcion GWR en el modelo principal y en las sensibilidades).
sys.path.insert(0, str(Path(__file__).parent))
from gwr_core import (add_intercept, select_bw, fit_gwr_ridge, predict_gwr, select_lambda,
                      local_condition_diagnostics, LAMBDA_GRID, LAMBDA_RIDGE,
                      N_INNER_LAMBDA, PENALIZE_INTERCEPT, BW_MIN, BW_MAX, BW_FALLBACK)
MORAN_K      = 8
MORAN_NPERM  = 999  # permutaciones para p-valor de Moran I (Anselin 1995)
RANDOM_STATE = 42

BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR.parent.parent / "datos" / "dataset.gpkg"
SPLIT_PATH = BASE_DIR.parent.parent / "data_split" / "split.csv"
FOLDS_PATH = BASE_DIR.parent.parent / "spatial_cv" / "output" / "fold_assignments.csv"
OUT_DIR    = BASE_DIR / "output_log_27vars"
OUT_DIR.mkdir(exist_ok=True)

def pr(*a, **k): print(*a, **k, flush=True)

def compute_metrics(y_true_orig, y_pred_orig, y_true_log, y_pred_log):
    e_orig = y_true_orig - y_pred_orig
    e_log  = y_true_log  - y_pred_log
    mae    = float(np.mean(np.abs(e_orig)))
    rmse   = float(np.sqrt(np.mean(e_orig**2)))
    mape   = float(np.mean(np.abs(e_orig/y_true_orig))*100)
    ss_r   = float(np.sum(e_orig**2)); ss_t = float(np.sum((y_true_orig-y_true_orig.mean())**2))
    r2_orig = round(1-ss_r/ss_t, 4) if ss_t>0 else float("nan")
    ss_rl  = float(np.sum(e_log**2)); ss_tl = float(np.sum((y_true_log-y_true_log.mean())**2))
    r2_log  = round(1-ss_rl/ss_tl, 4) if ss_tl>0 else float("nan")
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),
            "R2":r2_orig,"R2_log":r2_log}

def compute_moran_i(values, coords, k=MORAN_K, n_perm=MORAN_NPERM, seed=RANDOM_STATE):
    """Moran I con inferencia por permutacion (Cliff & Ord 1981; Anselin 1995).
    Devuelve (I, E[I], p_sim, z_sim):
      - I       estadistico observado (pesos KNN fila-estandarizados)
      - E[I]    = -1/(n-1) valor esperado bajo H0
      - p_sim   p-valor de permutacion unilateral (autocorrelacion positiva):
                (#{I_perm >= I_obs} + 1) / (n_perm + 1)
      - z_sim   pseudo z-score vs la distribucion nula permutada
    El denominador z'z es invariante a permutaciones, asi que solo se repermuta
    el numerador z'Wz."""
    w = KNNWeights.from_array(coords, k=k); w.transform="r"
    z = values - values.mean()
    denom = float(z @ z)
    I_obs = float((z @ (w.sparse @ z)) / denom)
    rng = np.random.default_rng(seed)
    I_perm = np.empty(n_perm)
    for b in range(n_perm):
        zp = rng.permutation(z)
        I_perm[b] = (zp @ (w.sparse @ zp)) / denom
    p_sim = (int(np.sum(I_perm >= I_obs)) + 1) / (n_perm + 1)
    sd = I_perm.std(ddof=1)
    z_sim = float((I_obs - I_perm.mean()) / sd) if sd > 0 else float("nan")
    return I_obs, -1/(len(values)-1), float(p_sim), z_sim

# ── Cargar datos ──────────────────────────────────────────────────────────────
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT_PATH)
folds_df = pd.read_csv(FOLDS_PATH)
gdf["predio_join"]       = gdf["predio_join"].astype(int)
split_df["predio_join"]  = split_df["predio_join"].astype(int)
folds_df["predio_join"]  = folds_df["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join","fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")

gdf = gdf.sort_values("predio_join").reset_index(drop=True)  # orden determinista + incluye predio con cero a la izquierda
coords    = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_orig    = gdf["valor_m2"].values.astype(float)
y_log     = np.log(y_orig)
X_raw, FEAT_NAMES = build_feature_matrix(gdf)              # one-hot (30 feat) -> PREDICCION
X_raw_cont, _     = build_feature_matrix(gdf, one_hot=False)  # continuo (27) -> SELECCION DE BW
sp_folds  = gdf["fold"].values.astype(int)
train_mask = (gdf["split"]=="train").values
test_mask  = (gdf["split"]=="test").values
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]

pr(f"n={len(gdf)}  train={len(train_idx)}  test={len(test_idx)}  vars={len(COVARIABLES)}")

# ── CV dentro del 80% ─────────────────────────────────────────────────────────
cv_rand  = RandomKFoldCV(n_splits=5, random_state=RANDOM_STATE)
cv_block = SpatialBlockCV(folds=sp_folds[train_mask])
splits_r = list(cv_rand.split(X_raw[train_mask]))
splits_b = list(cv_block.split(X_raw[train_mask]))
BUFFER_M = 2530  # rango residual exponencial (~2.5 km): SpatialBlock con separacion train-test garantizada (Roberts et al. 2017)
cv_block_buf = SpatialBlockBufferedCV(folds=sp_folds[train_mask], coords=coords[train_mask], buffer=BUFFER_M)
splits_bb = list(cv_block_buf.split(X_raw[train_mask]))
all_records = []; moran_records = []; diag_records = []

# ── CV sin leakage: BW se selecciona DENTRO de cada fold sobre su train only ──
pr(f"\n[GWR-27] CV con BW seleccionado por fold (sin leakage)")

for strat_name, splits in [("RandomKFold",splits_r),("SpatialBlock",splits_b),("SpatialBlock_buf",splits_bb)]:
    pr(f"\n{'='*60}\n[GWR-log / {strat_name}]\n{'='*60}")
    y_pred_oos = np.full(len(train_idx), np.nan)

    for fold_id, (tr, te) in enumerate(splits):
        pr(f"\n  fold {fold_id+1}/{len(splits)}  train={len(tr):,}  test={len(te):,}")
        t0 = time.time()

        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(X_raw[train_mask][tr])
        Xte_sc = scaler.transform(X_raw[train_mask][te])
        ytr_log = y_log[train_mask][tr]
        yte_log = y_log[train_mask][te]
        yte_ori = y_orig[train_mask][te]
        ctr = coords[train_mask][tr]
        cte = coords[train_mask][te]

        Xtr_int = add_intercept(Xtr_sc)
        Xte_int = add_intercept(Xte_sc)

        # BW con encoding CONTINUO: mgwr/Sel_BW da matriz singular con las dummies
        # ralas (uso_otros, 3 obs) en vecindarios locales. El bandwidth es un parametro
        # de escala espacial; no depende de como se codifique 1 de 27 vars. La PREDICCION
        # usa one-hot + ridge (abajo). Asi se restaura la seleccion por fold sin leakage.
        Xtr_int_bw = add_intercept(StandardScaler().fit_transform(X_raw_cont[train_mask][tr]))
        pr(f"    Seleccionando BW (encoding continuo p={Xtr_int_bw.shape[1]}) sobre fold (n={len(tr)}) ...")
        bw = select_bw(ctr, Xtr_int_bw, ytr_log)
        pr(f"    BW_fold={bw} vecinos")

        # lambda por CV ESPACIAL ANIDADO dentro del training de este fold (sin mirar el
        # test externo; particion interna espacial -> refleja extrapolacion). Intercepto
        # NO penalizado. El lambda es consecuencia del procedimiento, no eleccion post hoc.
        lam, lam_maes = select_lambda(ctr, Xtr_int, ytr_log, bw)
        pr(f"    lambda={lam} (CV espacial anidado; MAE_inner por lambda={ {k:round(v,1) for k,v in lam_maes.items()} })")
        pr(f"    Prediccion GWR estandar (solve en cada punto test)  n_te={len(te)}  p={Xtr_int.shape[1]} ...")
        pred_log, n_rc = predict_gwr(ctr, Xtr_int, ytr_log, cte, Xte_int, bw, lam=lam)
        # smearing de Duan FOLD-ESPECIFICO (#7): residuales in-sample del train de este fold.
        params_tr, _ = fit_gwr_ridge(ctr, Xtr_int, ytr_log, bw, lam=lam)
        pred_tr_log = np.einsum("ij,ij->i", Xtr_int, params_tr)
        s_M = float(np.mean(np.exp(ytr_log - pred_tr_log)))
        # Diagnostico de condicion numerica por fold (#4): cond, eigenvalor min, |z|max test.
        cond = local_condition_diagnostics(ctr, Xtr_int, cte, Xte_int, bw, lam=lam)
        pr(f"    n_ridge_corrections={n_rc}  s_M={s_M:.4f}  cond_med={cond['cond_median']:.0f}  ({time.time()-t0:.1f}s)")
        diag_records.append({"modelo":"GWR-27","estrategia":strat_name,"fold":fold_id,
                             "n_train":len(tr),"n_test":len(te),"bw":bw,"lambda":lam,
                             "penaliza_intercepto":PENALIZE_INTERCEPT, "s_M":round(s_M,4),
                             "n_ridge_corrections":n_rc,
                             "pct_ridge_corrections":round(100*n_rc/len(te),2),
                             **{k:round(v,2) for k,v in cond.items()}})

        pred_ori = np.exp(pred_log) * s_M   # metricas USD con smearing fold-especifico
        y_pred_oos[te] = pred_log

        m = compute_metrics(yte_ori, pred_ori, yte_log, pred_log)
        pr(f"    MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
           f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}  R2_log={m['R2_log']:.4f}")
        all_records.append({"modelo":"GWR-27","estrategia":strat_name,"fold":fold_id,**m})

    if LIBPYSAL_AVAILABLE:
        mask = ~np.isnan(y_pred_oos)
        resid = y_log[train_mask][mask] - y_pred_oos[mask]
        I, EI, p_sim, z_sim = compute_moran_i(resid, coords[train_mask][mask])
        pr(f"  Moran I OOS {strat_name}: I={I:.4f}  E[I]={EI:.6f}  p_perm={p_sim:.4f}  z={z_sim:.3f}")
        moran_records.append({"modelo":"GWR-27","estrategia":strat_name,
                              "I":round(I,6),"EI":round(EI,6),
                              "p_sim":round(p_sim,4),"z_sim":round(z_sim,4),"tipo":"CV_OOS"})

# ── Modelo final sobre 100% train → evaluar test 20% ─────────────────────────
pr(f"\n{'='*60}\n[GWR-log / Holdout final]\n{'='*60}")
scaler_f = StandardScaler()
Xtr_sc_f = scaler_f.fit_transform(X_raw[train_mask])
Xte_sc_f = scaler_f.transform(X_raw[test_mask])
Xtr_int_f = add_intercept(Xtr_sc_f)
Xte_int_f = add_intercept(Xte_sc_f)
ctr_f = coords[train_mask]; cte_f = coords[test_mask]

pr(f"  Seleccionando BW final (encoding continuo) ...")
Xtr_int_f_bw = add_intercept(StandardScaler().fit_transform(X_raw_cont[train_mask]))
bw_f = select_bw(ctr_f, Xtr_int_f_bw, y_log[train_mask])
pr(f"  BW_final={bw_f}")
lam_f, lam_f_maes = select_lambda(ctr_f, Xtr_int_f, y_log[train_mask], bw_f)  # nested spatial CV
pr(f"  lambda_final={lam_f} (CV espacial anidado; MAE_inner={ {k:round(v,1) for k,v in lam_f_maes.items()} })")
pr(f"  Prediccion GWR estandar sobre holdout n_te={len(test_idx)} p={Xtr_int_f.shape[1]} ...")
pred_log_test, n_rc_f = predict_gwr(ctr_f, Xtr_int_f, y_log[train_mask], cte_f, Xte_int_f, bw_f, lam=lam_f)
diag_records.append({"modelo":"GWR-27","estrategia":"Holdout20%","fold":"all",
                     "n_train":len(train_idx),"n_test":len(test_idx),"bw":bw_f,"lambda":lam_f,
                     "penaliza_intercepto":PENALIZE_INTERCEPT,
                     "n_ridge_corrections":n_rc_f,
                     "pct_ridge_corrections":round(100*n_rc_f/len(test_idx),2)})
pred_ori_test = np.exp(pred_log_test)
m_test = compute_metrics(y_orig[test_mask], pred_ori_test,
                         y_log[test_mask], pred_log_test)
pr(f"  HOLDOUT: MAE={m_test['MAE']:.2f}  RMSE={m_test['RMSE']:.2f}  "
   f"MAPE={m_test['MAPE']:.2f}%  R2={m_test['R2']:.4f}  R2_log={m_test['R2_log']:.4f}")

if LIBPYSAL_AVAILABLE:
    resid_test = y_log[test_mask] - pred_log_test
    I_h, EI_h, p_h, z_h = compute_moran_i(resid_test, cte_f)
    pr(f"  Moran I Holdout: I={I_h:.4f}  E[I]={EI_h:.6f}  p_perm={p_h:.4f}  z={z_h:.3f}")
    moran_records.append({"modelo":"GWR-27","estrategia":"Holdout20%",
                          "I":round(I_h,6),"EI":round(EI_h,6),
                          "p_sim":round(p_h,4),"z_sim":round(z_h,4),"tipo":"Holdout"})

# ── Guardar ───────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(all_records)
sum_df = res_df.groupby(["modelo","estrategia"]).agg(
    MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
    RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
    MAPE_mean=("MAPE","mean"), MAPE_std=("MAPE","std"),
    R2_mean=("R2","mean"), R2_std=("R2","std"),
).reset_index().round(4)
holdout_row = {"modelo":"GWR-27","estrategia":"Holdout20%",
               "n_test":len(test_idx),**m_test}

res_df.to_csv(OUT_DIR/"gwr27_log_results.csv", index=False)
sum_df.to_csv(OUT_DIR/"gwr27_log_summary.csv", index=False)
pd.DataFrame([holdout_row]).to_csv(OUT_DIR/"gwr27_log_holdout.csv", index=False)

# Diagnostico de estabilidad numerica por esquema/fold: bandwidth seleccionado
# y conteo de correcciones Ridge (evidencia directa de mal condicionamiento
# local). Permite verificar empiricamente la hipotesis de inestabilidad bajo
# SpatialBlock en lugar de asumirla.
diag_df = pd.DataFrame(diag_records)
diag_df.to_csv(OUT_DIR/"gwr27_log_diagnostics.csv", index=False)
pr(f"\n[Diagnostico] guardado en {OUT_DIR/'gwr27_log_diagnostics.csv'}")
pr(diag_df.to_string(index=False))

# Predicciones por predio (test + train) para smearing / bootstrap / CDF / mapa.
# Train in-sample: fit GWR estandar (beta(u_i,v_i) en cada punto train) → y_hat_i=x_i'beta_i.
params_f, n_rc_train = fit_gwr_ridge(ctr_f, Xtr_int_f, y_log[train_mask], bw_f, lam=lam_f)
if n_rc_train > 0:
    pr(f"[WARN] GWR fit (train): {n_rc_train} puntos con Ridge correction — s_M puede estar afectado")
pred_log_train_all = np.einsum("ij,ij->i", Xtr_int_f, params_f)
preds = pd.concat([
    pd.DataFrame({"predio_join": gdf.loc[test_mask,"predio_join"].astype(int).values,
                  "split":"test", "y_obs_log":y_log[test_mask], "y_pred_log":pred_log_test}),
    pd.DataFrame({"predio_join": gdf.loc[train_mask,"predio_join"].astype(int).values,
                  "split":"train","y_obs_log":y_log[train_mask],"y_pred_log":pred_log_train_all}),
], ignore_index=True)
preds["modelo"] = "GWR-27"
preds.to_csv(OUT_DIR/"gwr27_log_predictions.csv", index=False)
if moran_records:
    pd.DataFrame(moran_records).to_csv(OUT_DIR/"gwr27_log_moran.csv", index=False)

pr("\n" + "="*60)
pr("RESUMEN GWR-27vars")
pr("="*60)
pr(sum_df.to_string(index=False))
pr(f"\nHOLDOUT: {holdout_row}")
pr(f"\n[CSV] {OUT_DIR}")
