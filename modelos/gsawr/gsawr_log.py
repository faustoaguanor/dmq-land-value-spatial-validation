"""
gsawr_log.py
============
GSAWR con log(valor_m2) como target + holdout 80/20 espacialmente estratificado.

Arquitectura identica a gsawr_cv.py (Xu et al., 2025, IJGIS):
  SANNet (AFCNN + SAJPNN) + SAWCNN + OLS-weighted output

Target
------
  log(valor_m2) — estandar en modelos hedonicos de precios.
  Metricas reportadas en escala original (USD/m2) via exp().

Holdout
-------
  80% train (CV) + 20% test final. Split espacialmente estratificado.

Salidas
-------
  output_log/gsawr_log_results.csv   — fold x estrategia
  output_log/gsawr_log_summary.csv   — media/std
  output_log/gsawr_log_holdout.csv   — metricas test 20%
  output_log/gsawr_log_moran.csv     — Moran I OOS
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_CV_DIR = Path(__file__).parent.parent.parent / "spatial_cv"
sys.path.insert(0, str(_CV_DIR))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV, SpatialBlockBufferedCV

# one-hot de uso_suelo_cod via punto unico de verdad (modelos/features.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import build_feature_matrix

try:
    from libpysal.weights import KNN as KNNWeights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

# =============================================================================
# Configuracion
# =============================================================================

LAYER     = "puntos_mercado"
CRS_UTM   = 32717

GRID_COLS = 20
GRID_ROWS = 20
IDW_POWER = 2.0

# SANNet
AFCNN_HIDDEN  = [16]
SAJPNN_HIDDEN = 4

# SAWCNN
CNN_CHANNELS  = [16, 16, 32]
CNN_KERNEL    = (5, 5)
DENSE_LAYERS  = [256, 64]

# Entrenamiento
N_EPOCHS   = 300
PATIENCE   = 60
BATCH_SIZE = 32
START_LR   = 0.001
OPTIMIZER  = "Adam"

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MORAN_K      = 8

# Semillas fijas para reproducibilidad
import random as _random
_random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

COVARIABLES: list[str] = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]

BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR.parent.parent / "datos" / "dataset.gpkg"
SPLIT_PATH = BASE_DIR.parent.parent / "data_split" / "split.csv"
FOLDS_PATH = BASE_DIR.parent.parent / "spatial_cv" / "output" / "fold_assignments.csv"
OUT_DIR    = BASE_DIR / "output_log"
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def pr(*a, **k): print(*a, **k, flush=True)


# =============================================================================
# Capas neuronales (identicas a gsawr_cv.py)
# =============================================================================

class AFCNN(nn.Module):
    def __init__(self, n_feat, hidden, bias=True, batch_norm=True,
                 activate_func=None):
        super().__init__()
        if activate_func is None:
            activate_func = nn.ReLU()
        layers = nn.Sequential()
        in_ch = n_feat
        for i, out_ch in enumerate(hidden):
            layers.add_module(f"conv{i+1}",
                nn.Conv2d(in_ch, out_ch, kernel_size=(1,1), bias=bias))
            if batch_norm:
                layers.add_module(f"bn{i+1}", nn.BatchNorm2d(out_ch))
            layers.add_module(f"act{i+1}", activate_func)
            in_ch = out_ch
        layers.add_module("conv_out", nn.Conv2d(in_ch, 1, kernel_size=(1,1), bias=bias))
        layers.add_module("act_out", nn.ReLU())
        self.afcnn = layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)

    def forward(self, a): return self.afcnn(a)


class SAJPNN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden, bias=True), nn.PReLU(init=0.1),
            nn.Linear(hidden, 1, bias=True), nn.PReLU(init=0.1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)

    def forward(self, x): return self.net(x)


class SANNet(nn.Module):
    def __init__(self, n_feat, afcnn_hidden, sajpnn_hidden, bias=True, batch_norm=True):
        super().__init__()
        self.afcnn  = AFCNN(n_feat, afcnn_hidden, bias, batch_norm)
        self.sajpnn = SAJPNN(sajpnn_hidden)

    def forward(self, dis):
        b, c, h, w = dis.size()
        s_mat   = dis[:, 0:1, :, :]
        a_mat   = dis[:, 1:,  :, :]
        a_fused = self.afcnn(a_mat)
        sa      = torch.cat([s_mat, a_fused], dim=1)
        sa_flat = sa.permute(0,2,3,1).reshape(-1, 2)
        out     = self.sajpnn(sa_flat)
        return out.view(b, 1, h, w)


class SAWCNN(nn.Module):
    def __init__(self, rows, cols, n_features, cnn_channels, kernel_size,
                 dense_layers, activate_cnn=None, activate_dense=None):
        super().__init__()
        if activate_cnn is None:   activate_cnn   = nn.PReLU(init=0.1)
        if activate_dense is None: activate_dense = nn.ReLU()

        conv_blocks = nn.Sequential()
        in_ch = 1
        pad   = (kernel_size[0]//2, kernel_size[1]//2)
        for i, out_ch in enumerate(cnn_channels):
            block = nn.Sequential()
            block.add_module("conv", nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                                padding=pad, bias=True))
            block.add_module("act",  nn.PReLU(init=0.1))
            block.add_module("pool", nn.MaxPool2d(kernel_size=2))
            conv_blocks.add_module(f"block{i+1}", block)
            in_ch = out_ch
        self.conv_blocks = conv_blocks

        last_size = self._compute_flat_size(rows, cols, cnn_channels, kernel_size, pad)
        pr(f"    [SAWCNN] grid={rows}x{cols}  flattened={last_size}")

        fc = nn.Sequential()
        prev = last_size
        for i, h in enumerate(dense_layers):
            fc.add_module(f"fc{i}",  nn.Linear(prev, h))
            fc.add_module(f"act{i}", nn.ReLU())
            prev = h
        fc.add_module("fc_out", nn.Linear(prev, n_features))
        self.fc = fc
        self._init_weights()

    @staticmethod
    def _compute_flat_size(rows, cols, channels, kernel_size, pad):
        h, w = rows, cols
        for _ in channels:
            h = h // 2
            w = w // 2
        return channels[-1] * h * w

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)

    def forward(self, x):
        b = x.size(0)
        for block in self.conv_blocks:
            x = block(x)
        return self.fc(x.view(b, -1))


class GSAWRModel(nn.Module):
    def __init__(self, n_feat, n_features, rows, cols, afcnn_hidden, sajpnn_hidden,
                 cnn_channels, kernel_size, dense_layers, ols_coeff,
                 bias=True, batch_norm=True):
        super().__init__()
        self.sannet = SANNet(n_feat, afcnn_hidden, sajpnn_hidden, bias, batch_norm)
        self.sawcnn = SAWCNN(rows, cols, n_features, cnn_channels, kernel_size, dense_layers)
        self.out    = nn.Linear(n_features, 1, bias=False)
        self.out.weight = nn.Parameter(
            torch.tensor(ols_coeff.reshape(1,-1), dtype=torch.float32),
            requires_grad=False)

    def forward(self, dis, x):
        sa_map = self.sannet(dis)
        feat_w = self.sawcnn(sa_map)
        return self.out(feat_w * x)


# =============================================================================
# Metricas (dual-scale: log + original)
# =============================================================================

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


def compute_moran_i(values, coords, k=MORAN_K):
    w = KNNWeights.from_array(coords, k=k); w.transform = "r"
    z = values - values.mean(); Wz = w.sparse @ z
    return float((z@Wz)/(z@z)), -1/(len(values)-1)


# =============================================================================
# Funciones auxiliares
# =============================================================================

def add_intercept(X): return np.column_stack([np.ones(len(X)), X])

def compute_ols_coeff(X_int, y):
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_int, y)
    return ols.coef_.flatten().astype(np.float32)

def build_reference_grid(coords_train, cols, rows):
    x_min, y_min = coords_train.min(axis=0)
    x_max, y_max = coords_train.max(axis=0)
    dx = (x_max-x_min)*0.05; dy = (y_max-y_min)*0.05
    x_lin = np.linspace(x_min-dx, x_max+dx, cols)
    y_lin = np.linspace(y_min-dy, y_max+dy, rows)
    gx, gy = np.meshgrid(x_lin, y_lin)
    return np.column_stack([gx.ravel(), gy.ravel()])

def compute_grid_attrs_idw(coords_train, X_train_scaled, grid_xy, power=IDW_POWER):
    dists = cdist(grid_xy, coords_train, "euclidean")
    w = 1.0/(dists**power+1e-10); w /= w.sum(axis=1, keepdims=True)
    return w @ X_train_scaled

def compute_grid_distances_image(coords_query, X_query_scaled, grid_xy, grid_attrs, rows, cols):
    n      = len(coords_query)
    n_feat = X_query_scaled.shape[1]
    d_spatial = cdist(coords_query, grid_xy, "euclidean").reshape(n, 1, rows, cols)
    delta = np.abs(X_query_scaled[:,np.newaxis,:] - grid_attrs[np.newaxis,:,:])
    delta = delta.transpose(0,2,1).reshape(n, n_feat, rows, cols)
    return np.concatenate([d_spatial, delta], axis=1)

def make_loader(dis_img, X_int, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(dis_img, dtype=torch.float32),
        torch.tensor(X_int,   dtype=torch.float32),
        torch.tensor(y.reshape(-1,1), dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# =============================================================================
# Entrenamiento
# =============================================================================

def _one_epoch(model, loader, optimizer):
    training = optimizer is not None
    model.train(training)
    total_loss = total_n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for dis_img, x, y in loader:
            dis_img, x, y = dis_img.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
            y_hat = model(dis_img, x)
            loss  = F.mse_loss(y_hat, y)
            if training:
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); optimizer.step()
            total_loss += loss.item()*len(y); total_n += len(y)
    return total_loss/total_n

def train_model(model, train_loader, val_loader, model_path):
    opt = torch.optim.Adam(model.parameters(), lr=START_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=1e-5)
    best_val = float("inf"); patience = 0
    for epoch in range(1, N_EPOCHS+1):
        tr_loss  = _one_epoch(model, train_loader, opt)
        val_loss = _one_epoch(model, val_loader,   None)
        scheduler.step()
        if epoch%50==0 or epoch==1:
            pr(f"    epoch {epoch:4d}/{N_EPOCHS}  train={tr_loss:.4f}  val={val_loss:.4f}  "
               f"lr={opt.param_groups[0]['lr']:.6f}")
        if val_loss < best_val:
            best_val = val_loss; patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                pr(f"    Early stop en epoch {epoch}"); break
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))

def predict_model(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dis_img, x, _ in loader:
            preds.append(model(dis_img.to(DEVICE), x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(preds)


# =============================================================================
# Preprocesar tensor imagen (fit on train, transform train+test)
# =============================================================================

def build_image_tensors(coords_tr, X_tr_sc, coords_te, X_te_sc):
    grid_xy    = build_reference_grid(coords_tr, GRID_COLS, GRID_ROWS)
    grid_attrs = compute_grid_attrs_idw(coords_tr, X_tr_sc, grid_xy)
    dis_tr_raw = compute_grid_distances_image(
        coords_tr, X_tr_sc, grid_xy, grid_attrs, GRID_ROWS, GRID_COLS)
    dis_te_raw = compute_grid_distances_image(
        coords_te, X_te_sc, grid_xy, grid_attrs, GRID_ROWS, GRID_COLS)

    n_channels = dis_tr_raw.shape[1]
    dis_tr = np.empty_like(dis_tr_raw, dtype=np.float32)
    dis_te = np.empty_like(dis_te_raw, dtype=np.float32)
    for ch in range(n_channels):
        sc = StandardScaler()
        flat_tr = dis_tr_raw[:, ch, :, :].reshape(-1, 1)
        flat_te = dis_te_raw[:, ch, :, :].reshape(-1, 1)
        dis_tr[:, ch, :, :] = sc.fit_transform(flat_tr).reshape(
            len(coords_tr), GRID_ROWS, GRID_COLS)
        dis_te[:, ch, :, :] = sc.transform(flat_te).reshape(
            len(coords_te), GRID_ROWS, GRID_COLS)
    pr(f"    Tensor: {dis_tr.shape}  ({dis_tr.nbytes/1e6:.0f} MB train)")
    return dis_tr, dis_te


# =============================================================================
# Cargar datos
# =============================================================================

pr("=" * 70)
pr("  GSAWR-log — GSAWR con log(valor_m2) + holdout 80/20")
pr("=" * 70)

gdf      = gpd.read_file(DATA_PATH, layer=LAYER).to_crs(epsg=CRS_UTM)
split_df = pd.read_csv(SPLIT_PATH)
folds_df = pd.read_csv(FOLDS_PATH)
gdf["predio_join"]      = gdf["predio_join"].astype(int)
split_df["predio_join"] = split_df["predio_join"].astype(int)
folds_df["predio_join"] = folds_df["predio_join"].astype(int)
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

pr(f"n={len(gdf)}  train={len(train_idx)}  test={len(test_idx)}  vars={len(COVARIABLES)}")
if DEVICE.type == "cuda":
    pr(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# CV dentro del 80% train
# =============================================================================

cv_rand  = RandomKFoldCV(n_splits=5, random_state=RANDOM_STATE)
cv_block = SpatialBlockCV(folds=sp_folds[train_mask])
splits_r = list(cv_rand.split(X_raw[train_mask]))
splits_b = list(cv_block.split(X_raw[train_mask]))
BUFFER_M = 2530  # rango residual exponencial (~2.5 km): SpatialBlock con separacion train-test garantizada (Roberts et al. 2017)
cv_block_buf = SpatialBlockBufferedCV(folds=sp_folds[train_mask], coords=coords[train_mask], buffer=BUFFER_M)
splits_bb = list(cv_block_buf.split(X_raw[train_mask]))
all_records = []; moran_records = []

for strat_name, splits in [("RandomKFold", splits_r), ("SpatialBlock", splits_b), ("SpatialBlock_buf", splits_bb)]:
    pr(f"\n{'='*60}\n[GSAWR-log / {strat_name}]\n{'='*60}")
    n_splits = len(splits)
    y_pred_oos = np.full(len(train_idx), np.nan)

    for fold_id, (tr, te) in enumerate(splits):
        pr(f"\n  fold {fold_id+1}/{n_splits}  train={len(tr):,}  test={len(te):,}")
        t0 = time.time()

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_raw[train_mask][tr])
        X_te_sc  = scaler.transform(X_raw[train_mask][te])
        ytr_log  = y_log[train_mask][tr]
        yte_log  = y_log[train_mask][te]
        yte_orig = y_orig[train_mask][te]
        coords_tr = coords[train_mask][tr]
        coords_te = coords[train_mask][te]

        X_tr_int = add_intercept(X_tr_sc)
        X_te_int = add_intercept(X_te_sc)
        n_feat    = X_tr_sc.shape[1]
        n_features = X_tr_int.shape[1]

        ols_coeff = compute_ols_coeff(X_tr_int, ytr_log)

        # Tensores imagen
        dis_tr, dis_te = build_image_tensors(coords_tr, X_tr_sc, coords_te, X_te_sc)

        # Split interno 90/10 para early stopping
        rng      = np.random.default_rng(RANDOM_STATE + fold_id)
        val_mask = rng.random(len(tr)) < 0.10
        tr_mask  = ~val_mask

        train_loader = make_loader(dis_tr[tr_mask],  X_tr_int[tr_mask],  ytr_log[tr_mask],  BATCH_SIZE, True)
        val_loader   = make_loader(dis_tr[val_mask], X_tr_int[val_mask], ytr_log[val_mask], BATCH_SIZE, False)
        test_loader  = make_loader(dis_te, X_te_int, yte_log, BATCH_SIZE, False)

        model = GSAWRModel(
            n_feat=n_feat, n_features=n_features, rows=GRID_ROWS, cols=GRID_COLS,
            afcnn_hidden=AFCNN_HIDDEN, sajpnn_hidden=SAJPNN_HIDDEN,
            cnn_channels=CNN_CHANNELS, kernel_size=CNN_KERNEL, dense_layers=DENSE_LAYERS,
            ols_coeff=ols_coeff,
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pr(f"    Params entrenables: {n_params:,}")

        model_path = MODELS_DIR / f"gsawr_log_{strat_name}_fold{fold_id}.pt"
        train_model(model, train_loader, val_loader, model_path)

        pred_log = predict_model(model, test_loader)
        pred_orig = np.exp(pred_log)
        y_pred_oos[te] = pred_log

        m = compute_metrics(yte_orig, pred_orig, yte_log, pred_log)
        pr(f"    MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
           f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}  R2_log={m['R2_log']:.4f}  "
           f"({time.time()-t0:.1f}s)")
        all_records.append({"modelo":"GSAWR","estrategia":strat_name,"fold":fold_id,**m})

        del model, dis_tr, dis_te
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    # Moran I OOS
    if LIBPYSAL_AVAILABLE:
        mask = ~np.isnan(y_pred_oos)
        resid = y_log[train_mask][mask] - y_pred_oos[mask]
        I, EI = compute_moran_i(resid, coords[train_mask][mask])
        pr(f"  Moran I OOS {strat_name}: I={I:.4f}  E[I]={EI:.6f}")
        moran_records.append({"modelo":"GSAWR","estrategia":strat_name,
                               "I":round(I,6),"EI":round(EI,6),"tipo":"CV_OOS"})


# =============================================================================
# Modelo final sobre 100% train → test 20%
# =============================================================================

pr(f"\n{'='*60}\n[GSAWR-log / Holdout final]\n{'='*60}")

# Re-seed para holdout CONTROLADO: la inicialización del modelo final no debe
# depender del estado del RNG consumido por la CV previa (reproducible desde seed fijo).
import random as _rnd_hold
_rnd_hold.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_STATE)

scaler_f = StandardScaler()
X_tr_sc_f = scaler_f.fit_transform(X_raw[train_mask])
X_te_sc_f = scaler_f.transform(X_raw[test_mask])
X_tr_int_f = add_intercept(X_tr_sc_f)
X_te_int_f = add_intercept(X_te_sc_f)
coords_tr_f = coords[train_mask]
coords_te_f = coords[test_mask]
n_feat_f    = X_tr_sc_f.shape[1]
n_features_f = X_tr_int_f.shape[1]

ols_coeff_f = compute_ols_coeff(X_tr_int_f, y_log[train_mask])

pr("  Construyendo tensores finales ...")
dis_tr_f, dis_te_f = build_image_tensors(coords_tr_f, X_tr_sc_f, coords_te_f, X_te_sc_f)

# Split interno 10% validation
rng_f    = np.random.default_rng(RANDOM_STATE)
val_m    = rng_f.random(len(train_idx)) < 0.10
tr_m     = ~val_m

train_loader_f = make_loader(dis_tr_f[tr_m],  X_tr_int_f[tr_m],  y_log[train_mask][tr_m],  BATCH_SIZE, True)
val_loader_f   = make_loader(dis_tr_f[val_m], X_tr_int_f[val_m], y_log[train_mask][val_m], BATCH_SIZE, False)
test_loader_f  = make_loader(dis_te_f, X_te_int_f, y_log[test_mask], BATCH_SIZE, False)

model_f = GSAWRModel(
    n_feat=n_feat_f, n_features=n_features_f, rows=GRID_ROWS, cols=GRID_COLS,
    afcnn_hidden=AFCNN_HIDDEN, sajpnn_hidden=SAJPNN_HIDDEN,
    cnn_channels=CNN_CHANNELS, kernel_size=CNN_KERNEL, dense_layers=DENSE_LAYERS,
    ols_coeff=ols_coeff_f,
).to(DEVICE)
pr(f"  Params: {sum(p.numel() for p in model_f.parameters() if p.requires_grad):,}")

model_path_f = MODELS_DIR / "gsawr_log_final.pt"
pr("  Entrenando modelo final ...")
train_model(model_f, train_loader_f, val_loader_f, model_path_f)

pred_log_test  = predict_model(model_f, test_loader_f)
pred_orig_test = np.exp(pred_log_test)
m_test = compute_metrics(y_orig[test_mask], pred_orig_test, y_log[test_mask], pred_log_test)
pr(f"  HOLDOUT: MAE={m_test['MAE']:.2f}  RMSE={m_test['RMSE']:.2f}  "
   f"MAPE={m_test['MAPE']:.2f}%  R2={m_test['R2']:.4f}  R2_log={m_test['R2_log']:.4f}")

if LIBPYSAL_AVAILABLE:
    resid_test = y_log[test_mask] - pred_log_test
    I_h, EI_h  = compute_moran_i(resid_test, coords_te_f)
    pr(f"  Moran I Holdout: I={I_h:.4f}  E[I]={EI_h:.6f}")
    moran_records.append({"modelo":"GSAWR","estrategia":"Holdout20%",
                           "I":round(I_h,6),"EI":round(EI_h,6),"tipo":"Holdout"})


# =============================================================================
# Guardar
# =============================================================================

res_df = pd.DataFrame(all_records)
sum_df = res_df.groupby(["modelo","estrategia"]).agg(
    MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
    RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
    MAPE_mean=("MAPE","mean"), MAPE_std=("MAPE","std"),
    R2_mean=("R2","mean"), R2_std=("R2","std"),
).reset_index().round(4)

holdout_row = {"modelo":"GSAWR","estrategia":"Holdout20%","n_test":len(test_idx),**m_test}

res_df.to_csv(OUT_DIR/"gsawr_log_results.csv", index=False)
sum_df.to_csv(OUT_DIR/"gsawr_log_summary.csv", index=False)
pd.DataFrame([holdout_row]).to_csv(OUT_DIR/"gsawr_log_holdout.csv", index=False)

# Predicciones por predio (test + train) para smearing / bootstrap / CDF / mapa
train_loader_full = make_loader(dis_tr_f, X_tr_int_f, y_log[train_mask], BATCH_SIZE, False)
pred_log_train_all = predict_model(model_f, train_loader_full)
preds = pd.concat([
    pd.DataFrame({"predio_join": gdf.loc[test_mask,"predio_join"].astype(int).values,
                  "split":"test","y_obs_log":y_log[test_mask],"y_pred_log":pred_log_test}),
    pd.DataFrame({"predio_join": gdf.loc[train_mask,"predio_join"].astype(int).values,
                  "split":"train","y_obs_log":y_log[train_mask],"y_pred_log":pred_log_train_all}),
], ignore_index=True)
preds["modelo"] = "GSAWR"
preds.to_csv(OUT_DIR/"gsawr_log_predictions.csv", index=False)
if moran_records:
    pd.DataFrame(moran_records).to_csv(OUT_DIR/"gsawr_log_moran.csv", index=False)

pr("\n" + "="*60)
pr("RESUMEN GSAWR-log")
pr("="*60)
pr(sum_df.to_string(index=False))
pr(f"\nHOLDOUT: {holdout_row}")
pr(f"\n[CSV] {OUT_DIR}")
