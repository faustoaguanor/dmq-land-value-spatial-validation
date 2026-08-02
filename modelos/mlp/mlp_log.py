"""
mlp_log.py
==========
MLP (Multi-Layer Perceptron) NO-ESPACIAL con log(valor_m2) como target.
27 features, sin ningun componente espacial — baseline no-lineal puro.

Rol en la tesis
---------------
  Ocupa el hueco entre OLS (lineal no-espacial) y GWR/GNNWR (espacial).
  Si MLP < modelos espaciales, se confirma que la ganancia viene del
  componente espacial, no solo de la no-linealidad.

  OLS (lineal, no-esp.) → MLP (no-lineal, no-esp.) → GWR (lineal, esp.)
                        → GNNWR/SANNWR/GSAWR (no-lineal, esp.)

Arquitectura
------------
  27 → Linear(256) → BN → ReLU → Linear(128) → BN → ReLU
     → Linear(64)  → BN → ReLU → Linear(1)
  Adam lr=0.001, wd=1e-4, clip_grad=5.0, PATIENCE=40, BATCH_SIZE=64

Salidas
-------
  output_log/mlp_log_results.csv   — metricas por fold
  output_log/mlp_log_summary.csv   — media/std por estrategia
  output_log/mlp_log_holdout.csv   — metricas test 20%
  output_log/mlp_log_moran.csv     — Moran I OOS
"""
from __future__ import annotations
import sys, time
import random as _random
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
HIDDEN_LAYERS = [256, 128, 64]
N_EPOCHS     = 300
PATIENCE     = 40
BATCH_SIZE   = 64
LR           = 0.001
WD           = 1e-4
CLIP_GRAD    = 5.0
VAL_FRAC     = 0.10
MORAN_K      = 8
RANDOM_STATE = 42   # unificado con el resto de modelos (antes 123; eliminada asimetría de semilla)

BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR.parent.parent / "datos" / "dataset.gpkg"
SPLIT_PATH = BASE_DIR.parent.parent / "data_split" / "split.csv"
FOLDS_PATH = BASE_DIR.parent.parent / "spatial_cv" / "output" / "fold_assignments.csv"
OUT_DIR    = BASE_DIR / "output_log"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Semillas fijas para reproducibilidad
_random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

def pr(*a, **k): print(*a, **k, flush=True)

# ── Arquitectura MLP ──────────────────────────────────────────────────────────
class MLPRegressor(nn.Module):
    def __init__(self, n_feat: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_feat
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ── Metricas ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true_orig, y_pred_orig, y_true_log, y_pred_log):
    e_orig = y_true_orig - y_pred_orig
    e_log  = y_true_log  - y_pred_log
    mae    = float(np.mean(np.abs(e_orig)))
    rmse   = float(np.sqrt(np.mean(e_orig**2)))
    mape   = float(np.mean(np.abs(e_orig / y_true_orig)) * 100)
    ss_res = float(np.sum(e_orig**2))
    ss_tot = float(np.sum((y_true_orig - y_true_orig.mean())**2))
    r2_orig = round(1 - ss_res/ss_tot, 4) if ss_tot > 0 else float("nan")
    ss_rl  = float(np.sum(e_log**2))
    ss_tl  = float(np.sum((y_true_log - y_true_log.mean())**2))
    r2_log  = round(1 - ss_rl/ss_tl, 4) if ss_tl > 0 else float("nan")
    return {"MAE": round(mae,4), "RMSE": round(rmse,4),
            "MAPE": round(mape,4), "R2": r2_orig, "R2_log": r2_log}

def compute_moran_i(values, coords, k=MORAN_K):
    w = KNNWeights.from_array(coords, k=k); w.transform = "r"
    z = values - values.mean(); Wz = w.sparse @ z
    return float((z@Wz)/(z@z)), -1/(len(values)-1)

# ── Entrenamiento con early stopping ─────────────────────────────────────────
def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              n_feat: int) -> MLPRegressor:
    model = MLPRegressor(n_feat, HIDDEN_LAYERS).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.HuberLoss(delta=1.0)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_mlp(model: MLPRegressor, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return out.cpu().numpy()

# ── Cargar datos ──────────────────────────────────────────────────────────────
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT_PATH)
folds_df = pd.read_csv(FOLDS_PATH)
folds_df["predio_join"] = folds_df["predio_join"].astype(int)
split_df["predio_join"] = split_df["predio_join"].astype(int)
gdf["predio_join"]      = gdf["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join","fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")

coords    = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_orig    = gdf["valor_m2"].values.astype(float)
y_log     = np.log(y_orig)
X_raw, FEAT_NAMES = build_feature_matrix(gdf)   # one-hot uso_suelo_cod (27->31)
sp_folds  = gdf["fold"].values.astype(int)

train_mask = (gdf["split"] == "train").values
test_mask  = (gdf["split"] == "test").values
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]
n_feat     = X_raw.shape[1]

pr(f"MLP  device={DEVICE}  n={len(gdf)}  train={len(train_idx)}  test={len(test_idx)}  features={n_feat}")
pr(f"Arch: {n_feat} → {HIDDEN_LAYERS} → 1  |  PATIENCE={PATIENCE}  BATCH={BATCH_SIZE}")

# ── CV dentro del 80% train ───────────────────────────────────────────────────
cv_rand  = RandomKFoldCV(n_splits=5, random_state=RANDOM_STATE)
cv_block = SpatialBlockCV(folds=sp_folds[train_mask])
splits_r = list(cv_rand.split(X_raw[train_mask], y_log[train_mask]))
splits_b = list(cv_block.split(X_raw[train_mask], y_log[train_mask]))
BUFFER_M = 2530  # rango residual exponencial (~2.5 km): SpatialBlock con separacion train-test garantizada (Roberts et al. 2017)
cv_block_buf = SpatialBlockBufferedCV(folds=sp_folds[train_mask], coords=coords[train_mask], buffer=BUFFER_M)
splits_bb = list(cv_block_buf.split(X_raw[train_mask]))

all_records   = []
moran_records = []
rng_val = np.random.default_rng(RANDOM_STATE)

for strat_name, splits in [("RandomKFold", splits_r), ("SpatialBlock", splits_b), ("SpatialBlock_buf", splits_bb)]:
    pr(f"\n{'='*60}\n[MLP / {strat_name}]\n{'='*60}")
    y_pred_oos = np.full(len(train_idx), np.nan)

    for fold_id, (tr, te) in enumerate(splits):
        t0 = time.time()
        # ── Per-fold seed reset (Fix 2) ──────────────────────────────────────
        _fold_seed = RANDOM_STATE + fold_id * 100
        _random.seed(_fold_seed)
        np.random.seed(_fold_seed)
        torch.manual_seed(_fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_fold_seed)

        scaler  = StandardScaler()
        Xtr_sc  = scaler.fit_transform(X_raw[train_mask][tr])
        Xte_sc  = scaler.transform(X_raw[train_mask][te])
        ytr_log = y_log[train_mask][tr]
        yte_log = y_log[train_mask][te]
        yte_ori = y_orig[train_mask][te]

        # Split interno 90/10 para early stopping
        vm = rng_val.random(len(tr)) < VAL_FRAC
        tm = ~vm
        X_tr_fit  = Xtr_sc[tm]; y_tr_fit  = ytr_log[tm].astype(np.float32)
        X_val_fit = Xtr_sc[vm]; y_val_fit = ytr_log[vm].astype(np.float32)

        # ── Target standardization (Fix 1) ───────────────────────────────────
        y_mean = float(y_tr_fit.mean())
        y_std  = float(y_tr_fit.std()) + 1e-8
        y_tr_norm  = ((y_tr_fit  - y_mean) / y_std).astype(np.float32)
        y_val_norm = ((y_val_fit - y_mean) / y_std).astype(np.float32)

        model = train_mlp(X_tr_fit, y_tr_norm, X_val_fit, y_val_norm, n_feat)

        # Invert normalization after prediction
        pred_log = predict_mlp(model, Xte_sc) * y_std + y_mean
        pred_ori = np.exp(pred_log)
        y_pred_oos[te] = pred_log

        m = compute_metrics(yte_ori, pred_ori, yte_log, pred_log)
        pr(f"  fold {fold_id+1}/5  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
           f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}  ({time.time()-t0:.1f}s)")
        all_records.append({"modelo":"MLP","estrategia":strat_name,
                            "fold":fold_id, **m})

    if LIBPYSAL_AVAILABLE:
        mask_oos = ~np.isnan(y_pred_oos)
        resid_oos = y_log[train_mask][mask_oos] - y_pred_oos[mask_oos]
        I, EI = compute_moran_i(resid_oos, coords[train_mask][mask_oos])
        pr(f"  Moran I OOS {strat_name}: I={I:.4f}  E[I]={EI:.6f}")
        moran_records.append({"modelo":"MLP","estrategia":strat_name,
                              "I":round(I,6),"EI":round(EI,6),"tipo":"CV_OOS"})

# ── Modelo final: 80% train → evaluar 20% test ───────────────────────────────
pr(f"\n{'='*60}\n[MLP / Holdout final]\n{'='*60}")
# Re-seed para holdout CONTROLADO (init independiente del RNG post-CV).
_random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_STATE)
scaler_f = StandardScaler()
X_tr_f   = scaler_f.fit_transform(X_raw[train_mask])
X_te_f   = scaler_f.transform(X_raw[test_mask])
y_tr_log_f = y_log[train_mask].astype(np.float32)

vm_f = np.random.default_rng(RANDOM_STATE).random(len(train_idx)) < VAL_FRAC
tm_f = ~vm_f
y_f_mean = float(y_tr_log_f[tm_f].mean())
y_f_std  = float(y_tr_log_f[tm_f].std()) + 1e-8
y_tr_f_norm  = ((y_tr_log_f[tm_f] - y_f_mean) / y_f_std).astype(np.float32)
y_val_f_norm = ((y_tr_log_f[vm_f] - y_f_mean) / y_f_std).astype(np.float32)
model_f = train_mlp(X_tr_f[tm_f], y_tr_f_norm,
                    X_tr_f[vm_f], y_val_f_norm, n_feat)

pred_log_test = predict_mlp(model_f, X_te_f) * y_f_std + y_f_mean
pred_ori_test = np.exp(pred_log_test)
m_test = compute_metrics(y_orig[test_mask], pred_ori_test,
                         y_log[test_mask], pred_log_test)
pr(f"  HOLDOUT  MAE={m_test['MAE']:.2f}  RMSE={m_test['RMSE']:.2f}  "
   f"MAPE={m_test['MAPE']:.2f}%  R2={m_test['R2']:.4f}  R2_log={m_test['R2_log']:.4f}")

holdout_row = {"modelo":"MLP","estrategia":"Holdout20%",
               "n_test":len(test_idx), **m_test}

if LIBPYSAL_AVAILABLE:
    resid_test = y_log[test_mask] - pred_log_test
    I_h, EI_h  = compute_moran_i(resid_test, coords[test_mask])
    pr(f"  Moran I Holdout: I={I_h:.4f}  E[I]={EI_h:.6f}")
    moran_records.append({"modelo":"MLP","estrategia":"Holdout20%",
                          "I":round(I_h,6),"EI":round(EI_h,6),"tipo":"Holdout"})

# ── Guardar ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_records)
summary_df = results_df.groupby(["modelo","estrategia"]).agg(
    MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
    RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
    MAPE_mean=("MAPE","mean"), MAPE_std=("MAPE","std"),
    R2_mean=("R2","mean"), R2_std=("R2","std"),
).reset_index().round(4)

results_df.to_csv(OUT_DIR / "mlp_log_results.csv", index=False)
summary_df.to_csv(OUT_DIR / "mlp_log_summary.csv", index=False)
pd.DataFrame([holdout_row]).to_csv(OUT_DIR / "mlp_log_holdout.csv", index=False)

# Predicciones por predio (test + train) para smearing / bootstrap / CDF / mapa
pred_log_train_all = predict_mlp(model_f, X_tr_f) * y_f_std + y_f_mean
preds = pd.concat([
    pd.DataFrame({"predio_join": gdf.loc[test_mask,"predio_join"].astype(int).values,
                  "split":"test", "y_obs_log":y_log[test_mask], "y_pred_log":pred_log_test}),
    pd.DataFrame({"predio_join": gdf.loc[train_mask,"predio_join"].astype(int).values,
                  "split":"train","y_obs_log":y_log[train_mask],"y_pred_log":pred_log_train_all}),
], ignore_index=True)
preds["modelo"] = "MLP"
preds.to_csv(OUT_DIR / "mlp_log_predictions.csv", index=False)
if moran_records:
    pd.DataFrame(moran_records).to_csv(OUT_DIR / "mlp_log_moran.csv", index=False)

pr("\n" + "="*60)
pr("RESUMEN MLP log(y)")
pr("="*60)
pr(summary_df.to_string(index=False))
pr(f"\nHOLDOUT: {holdout_row}")
pr(f"\n[CSV] {OUT_DIR}")
