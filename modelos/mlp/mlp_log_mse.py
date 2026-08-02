"""
mlp_log_mse.py  — Control: MLP con MSE en lugar de Huber loss
==============================================================
Proposito: aislar si la diferencia MLP vs modelos espaciales se debe a
la funcion de perdida (Huber) o al componente espacial en si.

Arquitectura IDENTICA a mlp_log.py; solo cambia la loss:
  mlp_log.py    : nn.HuberLoss(delta=1.0)
  mlp_log_mse.py: nn.MSELoss()

Si RMSE_mse ~ RMSE_huber, la funcion de perdida no explica la diferencia.
Si RMSE_mse >> RMSE_huber, Huber da ventaja real a MLP.

Salidas: modelos/mlp/output_log/mlp_log_mse_holdout.csv
"""
from __future__ import annotations
import sys, time, random as _random
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix

OUT_DIR = Path(__file__).parent / "output_log"
OUT_DIR.mkdir(exist_ok=True)

HIDDEN_LAYERS = [256, 128, 64]
N_EPOCHS   = 300
PATIENCE   = 40
BATCH_SIZE = 64
LR         = 0.001
WD         = 1e-4
CLIP_GRAD  = 5.0
VAL_FRAC   = 0.10
SEED       = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPRegressor(nn.Module):
    def __init__(self, n_feat, hidden):
        super().__init__()
        layers = []; in_dim = n_feat
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]; in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                nn.init.constant_(m.bias, 0.0)
    def forward(self, x): return self.net(x).squeeze(-1)

print(f"Device: {DEVICE}  Seed: {SEED}")
_random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda": torch.cuda.manual_seed(SEED)

# ── Cargar datos ──────────────────────────────────────────────────────────────
gdf = gpd.read_file(ROOT/"datos"/"dataset.gpkg", layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"] = gdf["predio_join"].astype(int)
split_df = pd.read_csv(ROOT/"data_split"/"split.csv")
split_df["predio_join"] = split_df["predio_join"].astype(int)
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)

y_orig = gdf["valor_m2"].values.astype(float)
y_log  = np.log(y_orig)
X_raw, FEAT_NAMES = build_feature_matrix(gdf)
train_mask = (gdf["split"] == "train").values
test_mask  = (gdf["split"] == "test").values
n_feat = X_raw.shape[1]
print(f"n_train={train_mask.sum()}  n_test={test_mask.sum()}  n_feat={n_feat}")

scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(X_raw[train_mask]).astype(np.float32)
Xte_sc = scaler.transform(X_raw[test_mask]).astype(np.float32)
ytr_log = y_log[train_mask].astype(np.float32)

rng_v = np.random.default_rng(SEED)
vm = rng_v.random(train_mask.sum()) < VAL_FRAC; tm = ~vm
y_mean = float(ytr_log[tm].mean()); y_std = float(ytr_log[tm].std()) + 1e-8
ytr_norm  = (ytr_log[tm] - y_mean) / y_std
yval_norm = (ytr_log[vm] - y_mean) / y_std

# ── Entrenar con MSELoss ───────────────────────────────────────────────────────
model   = MLPRegressor(n_feat, HIDDEN_LAYERS).to(DEVICE)
opt     = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
loss_fn = nn.MSELoss()   # <-- UNICA diferencia vs mlp_log.py

Xt = torch.tensor(Xtr_sc[tm], dtype=torch.float32)
yt = torch.tensor(ytr_norm,    dtype=torch.float32)
Xv = torch.tensor(Xtr_sc[vm], dtype=torch.float32).to(DEVICE)
yv = torch.tensor(yval_norm,   dtype=torch.float32).to(DEVICE)
loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

best_val = float("inf"); best_state = None; no_improve = 0
t0 = time.time()
for ep in range(N_EPOCHS):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); loss = loss_fn(model(xb), yb)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD); opt.step()
    model.eval()
    with torch.no_grad(): val_loss = loss_fn(model(Xv), yv).item()
    if val_loss < best_val - 1e-6:
        best_val = val_loss; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE: print(f"Early stop ep={ep+1}"); break
if best_state: model.load_state_dict(best_state)
print(f"Entrenado en {time.time()-t0:.0f}s")

# ── Evaluar en holdout ────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    pred_norm = model(torch.tensor(Xte_sc, dtype=torch.float32).to(DEVICE)).cpu().numpy()
pred_log = pred_norm * y_std + y_mean
pred_orig = np.exp(pred_log)
y_te = y_orig[test_mask]
y_te_log = y_log[test_mask]

rmse   = float(np.sqrt(np.mean((y_te - pred_orig)**2)))
mae    = float(np.mean(np.abs(y_te - pred_orig)))
r2     = float(1 - np.var(y_te - pred_orig) / np.var(y_te))
rmse_log = float(np.sqrt(np.mean((y_te_log - pred_log)**2)))

# Smearing
with torch.no_grad():
    _pred_tr = model(torch.tensor(Xtr_sc, dtype=torch.float32).to(DEVICE)).cpu().numpy()
s_m  = float(np.mean(np.exp(y_log[train_mask] - (_pred_tr * y_std + y_mean))))
pred_smeared = np.exp(pred_log) * s_m
rmse_sm = float(np.sqrt(np.mean((y_te - pred_smeared)**2)))
mae_sm  = float(np.mean(np.abs(y_te - pred_smeared)))

print(f"\n=== MLP con MSELoss (seed={SEED}) ===")
print(f"  RMSE raw    : {rmse:.4f}")
print(f"  RMSE smeared: {rmse_sm:.4f}")
print(f"  MAE raw     : {mae:.4f}")
print(f"  MAE smeared : {mae_sm:.4f}")
print(f"  R²          : {r2:.6f}")
print(f"  RMSE log    : {rmse_log:.4f}")
print(f"  s_M (smear) : {s_m:.5f}")
print(f"\n  [Referencia MLP Huber] RMSE smeared~99.58  MAE~49.96")
print(f"  [Diferencia MSE vs Huber] ΔRMSE={rmse_sm-99.58:+.2f}  ΔMAE={mae_sm-49.96:+.2f}")

pd.DataFrame([{
    "modelo": "MLP_MSE", "seed": SEED, "loss_fn": "MSELoss",
    "RMSE_raw": round(rmse,4), "RMSE_smeared": round(rmse_sm,4),
    "MAE_raw": round(mae,4), "MAE_smeared": round(mae_sm,4),
    "R2": round(r2,6), "RMSE_log": round(rmse_log,4), "smearing_factor": round(s_m,5),
    "ref_MLP_Huber_RMSE_smeared": 99.58, "ref_MLP_Huber_MAE_smeared": 49.96,
    "delta_RMSE_vs_Huber": round(rmse_sm-99.58,4),
    "delta_MAE_vs_Huber": round(mae_sm-49.96,4),
}]).to_csv(OUT_DIR / "mlp_log_mse_holdout.csv", index=False)
print(f"\n[OK] mlp_log_mse_holdout.csv guardado")
