"""
gsawr_log_replicas.py
=====================
Replicas GSAWR con seeds [42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7] — solo holdout 20%.
Evalua estabilidad del modelo CNN multi-atributo frente a inicializacion aleatoria.

Salida: output_log/gsawr_log_replicas.csv
"""
from __future__ import annotations
import math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from libpysal.weights import KNN as KNNWeights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────
# Config IDÉNTICA al script principal gsawr_log.py (solo varía la semilla).
# Antes usaba AFCNN_HIDDEN=[8] y PATIENCE=40 — distinto al principal, invalidando la medición.
GRID_COLS = 20; GRID_ROWS = 20; IDW_POWER = 2.0
AFCNN_HIDDEN  = [16]; SAJPNN_HIDDEN = 4
CNN_CHANNELS  = [16, 16, 32]; CNN_KERNEL = (5, 5)
DENSE_LAYERS  = [256, 64]
N_EPOCHS = 300; PATIENCE = 60; BATCH_SIZE = 32; START_LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MORAN_K = 8; SEEDS = [42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7]

COVARIABLES = [
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
OUT_DIR    = BASE_DIR / "output_log"
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)

def pr(*a, **k): print(*a, **k, flush=True)

# ── Capas neuronales ─────────────────────────────────────────────────────────
class AFCNN(nn.Module):
    def __init__(self, n_feat, hidden, bias=True, batch_norm=True, activate_func=None):
        super().__init__()
        if activate_func is None: activate_func = nn.ReLU()
        layers = nn.Sequential(); in_ch = n_feat
        for i, out_ch in enumerate(hidden):
            layers.add_module(f"conv{i+1}", nn.Conv2d(in_ch, out_ch, kernel_size=(1,1), bias=bias))
            if batch_norm: layers.add_module(f"bn{i+1}", nn.BatchNorm2d(out_ch))
            layers.add_module(f"act{i+1}", activate_func); in_ch = out_ch
        layers.add_module("conv_out", nn.Conv2d(in_ch, 1, kernel_size=(1,1), bias=bias))
        layers.add_module("act_out", nn.ReLU())
        self.afcnn = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in"); (m.bias.data.fill_(0.) if m.bias is not None else None)
    def forward(self, a): return self.afcnn(a)

class SAJPNN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden, bias=True), nn.PReLU(init=0.1),
                                 nn.Linear(hidden, 1, bias=True), nn.PReLU(init=0.1))
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in"); (m.bias.data.fill_(0.) if m.bias is not None else None)
    def forward(self, x): return self.net(x)

class SANNet(nn.Module):
    def __init__(self, n_feat, afcnn_hidden, sajpnn_hidden, bias=True, batch_norm=True):
        super().__init__()
        self.afcnn = AFCNN(n_feat, afcnn_hidden, bias, batch_norm)
        self.sajpnn = SAJPNN(sajpnn_hidden)
    def forward(self, dis):
        b, c, h, w = dis.size()
        s_mat = dis[:, 0:1, :, :]; a_mat = dis[:, 1:, :, :]
        a_fused = self.afcnn(a_mat)
        sa = torch.cat([s_mat, a_fused], dim=1)
        sa_flat = sa.permute(0,2,3,1).reshape(-1, 2)
        return self.sajpnn(sa_flat).view(b, 1, h, w)

class SAWCNN(nn.Module):
    def __init__(self, rows, cols, n_features, cnn_channels, kernel_size, dense_layers):
        super().__init__()
        conv_blocks = nn.Sequential(); in_ch = 1
        pad = (kernel_size[0]//2, kernel_size[1]//2)
        for i, out_ch in enumerate(cnn_channels):
            block = nn.Sequential()
            block.add_module("conv", nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=True))
            block.add_module("act", nn.PReLU(init=0.1))
            block.add_module("pool", nn.MaxPool2d(kernel_size=2))
            conv_blocks.add_module(f"block{i+1}", block); in_ch = out_ch
        self.conv_blocks = conv_blocks
        h, w = rows, cols
        for _ in cnn_channels: h //= 2; w //= 2
        last_size = cnn_channels[-1] * h * w
        fc = nn.Sequential(); prev = last_size
        for i, hh in enumerate(dense_layers):
            fc.add_module(f"fc{i}", nn.Linear(prev, hh)); fc.add_module(f"act{i}", nn.ReLU()); prev = hh
        fc.add_module("fc_out", nn.Linear(prev, n_features))
        self.fc = fc
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)): nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in"); (m.bias.data.fill_(0.) if m.bias is not None else None)
    def forward(self, x):
        b = x.size(0)
        for block in self.conv_blocks: x = block(x)
        return self.fc(x.view(b, -1))

class GSAWRModel(nn.Module):
    def __init__(self, n_feat, n_features, rows, cols, afcnn_hidden, sajpnn_hidden,
                 cnn_channels, kernel_size, dense_layers, ols_coeff, bias=True, batch_norm=True):
        super().__init__()
        self.sannet = SANNet(n_feat, afcnn_hidden, sajpnn_hidden, bias, batch_norm)
        self.sawcnn = SAWCNN(rows, cols, n_features, cnn_channels, kernel_size, dense_layers)
        self.out = nn.Linear(n_features, 1, bias=False)
        self.out.weight = nn.Parameter(torch.tensor(ols_coeff.reshape(1,-1), dtype=torch.float32), requires_grad=False)
    def forward(self, dis, x):
        sa_map = self.sannet(dis); feat_w = self.sawcnn(sa_map)
        return self.out(feat_w * x)

# ── Funciones ─────────────────────────────────────────────────────────────────
def add_intercept(X): return np.column_stack([np.ones(len(X)), X])

def compute_ols_coeff(X_int, y):
    return LinearRegression(fit_intercept=False).fit(X_int, y).coef_.flatten().astype(np.float32)

def compute_metrics(y_true_orig, y_pred_orig, y_true_log, y_pred_log):
    e_orig = y_true_orig - y_pred_orig; e_log = y_true_log - y_pred_log
    mae = float(np.mean(np.abs(e_orig))); rmse = float(np.sqrt(np.mean(e_orig**2)))
    mape = float(np.mean(np.abs(e_orig/y_true_orig))*100)
    ss_r = np.sum(e_orig**2); ss_t = np.sum((y_true_orig-y_true_orig.mean())**2)
    r2 = round(1-ss_r/ss_t, 4) if ss_t>0 else float("nan")
    ss_rl = np.sum(e_log**2); ss_tl = np.sum((y_true_log-y_true_log.mean())**2)
    r2l = round(1-ss_rl/ss_tl, 4) if ss_tl>0 else float("nan")
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l}

def compute_moran_i(values, coords, k=MORAN_K):
    w = KNNWeights.from_array(coords, k=k); w.transform="r"
    z = values-values.mean(); Wz = w.sparse@z
    return float((z@Wz)/(z@z)), -1/(len(values)-1)

def build_reference_grid(coords_train, cols, rows):
    x_min, y_min = coords_train.min(axis=0); x_max, y_max = coords_train.max(axis=0)
    dx = (x_max-x_min)*0.05; dy = (y_max-y_min)*0.05
    return np.column_stack([
        np.meshgrid(np.linspace(x_min-dx, x_max+dx, cols),
                    np.linspace(y_min-dy, y_max+dy, rows))[0].ravel(),
        np.meshgrid(np.linspace(x_min-dx, x_max+dx, cols),
                    np.linspace(y_min-dy, y_max+dy, rows))[1].ravel(),
    ])

def compute_grid_attrs_idw(coords_train, X_train_scaled, grid_xy):
    dists = cdist(grid_xy, coords_train); w = 1.0/(dists**IDW_POWER+1e-10)
    w /= w.sum(axis=1, keepdims=True); return w @ X_train_scaled

def compute_grid_distances_image(coords_query, X_query_scaled, grid_xy, grid_attrs, rows, cols):
    n = len(coords_query); n_feat = X_query_scaled.shape[1]
    d_spatial = cdist(coords_query, grid_xy).reshape(n, 1, rows, cols)
    delta = np.abs(X_query_scaled[:,np.newaxis,:] - grid_attrs[np.newaxis,:,:])
    delta = delta.transpose(0,2,1).reshape(n, n_feat, rows, cols)
    return np.concatenate([d_spatial, delta], axis=1)

def build_image_tensors(coords_tr, X_tr_sc, coords_te, X_te_sc):
    grid_xy = build_reference_grid(coords_tr, GRID_COLS, GRID_ROWS)
    ga = compute_grid_attrs_idw(coords_tr, X_tr_sc, grid_xy)
    dis_tr_raw = compute_grid_distances_image(coords_tr, X_tr_sc, grid_xy, ga, GRID_ROWS, GRID_COLS)
    dis_te_raw = compute_grid_distances_image(coords_te, X_te_sc, grid_xy, ga, GRID_ROWS, GRID_COLS)
    n_ch = dis_tr_raw.shape[1]
    dis_tr = np.empty_like(dis_tr_raw, dtype=np.float32)
    dis_te = np.empty_like(dis_te_raw, dtype=np.float32)
    for ch in range(n_ch):
        sc = StandardScaler()
        dis_tr[:, ch, :, :] = sc.fit_transform(dis_tr_raw[:, ch, :, :].reshape(-1, 1)).reshape(len(coords_tr), GRID_ROWS, GRID_COLS)
        dis_te[:, ch, :, :] = sc.transform(dis_te_raw[:, ch, :, :].reshape(-1, 1)).reshape(len(coords_te), GRID_ROWS, GRID_COLS)
    pr(f"    Tensor: {dis_tr.shape}  ({dis_tr.nbytes/1e6:.0f} MB train)")
    return dis_tr, dis_te

def make_loader(dis_img, X_int, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(dis_img, dtype=torch.float32),
                       torch.tensor(X_int,   dtype=torch.float32),
                       torch.tensor(y.reshape(-1,1), dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

def _one_epoch(model, loader, optimizer):
    training = optimizer is not None; model.train(training)
    total_loss = total_n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for dis_img, x, y in loader:
            dis_img, x, y = dis_img.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
            y_hat = model(dis_img, x); loss = F.mse_loss(y_hat, y)
            if training: optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); optimizer.step()
            total_loss += loss.item()*len(y); total_n += len(y)
    return total_loss/total_n

def train_model(model, train_loader, val_loader, model_path):
    opt = torch.optim.Adam(model.parameters(), lr=START_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-5)
    best_val = float("inf"); patience = 0
    for epoch in range(1, N_EPOCHS+1):
        tr_loss = _one_epoch(model, train_loader, opt)
        val_loss = _one_epoch(model, val_loader, None)
        scheduler.step()
        if epoch%50==0 or epoch==1:
            pr(f"    epoch {epoch:4d}/{N_EPOCHS}  train={tr_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss; patience = 0; torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= PATIENCE: pr(f"    Early stop en epoch {epoch}"); break
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))

def predict_model(model, loader):
    model.eval(); preds = []
    with torch.no_grad():
        for dis_img, x, _ in loader:
            preds.append(model(dis_img.to(DEVICE), x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(preds)

# ── Cargar datos ──────────────────────────────────────────────────────────────
pr("Cargando datos ...")
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"] = gdf["predio_join"].astype(int)
split_df = pd.read_csv(SPLIT_PATH); split_df["predio_join"] = split_df["predio_join"].astype(int)
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_orig = gdf["valor_m2"].values.astype(float); y_log = np.log(y_orig)
import sys as _s; from pathlib import Path as _P
_s.path.insert(0, str(_P(__file__).resolve().parent.parent))
from features import build_feature_matrix
X_raw, _FEAT = build_feature_matrix(gdf)   # one-hot uso_suelo_cod (27->31)
train_mask = (gdf["split"]=="train").values; test_mask = (gdf["split"]=="test").values
train_idx = np.where(train_mask)[0]; test_idx = np.where(test_mask)[0]
pr(f"GSAWR replicas  device={DEVICE}  train={len(train_idx)}  test={len(test_idx)}")

# Pre-calcular tensores imagen (son fijos, no dependen de seed)
pr("Construyendo tensores imagen finales ...")
scaler_f = StandardScaler()
X_tr_sc_f = scaler_f.fit_transform(X_raw[train_mask])
X_te_sc_f = scaler_f.transform(X_raw[test_mask])
X_tr_int_f = add_intercept(X_tr_sc_f); X_te_int_f = add_intercept(X_te_sc_f)
coords_tr_f = coords[train_mask]; coords_te_f = coords[test_mask]
n_feat_f = X_tr_sc_f.shape[1]; n_features_f = X_tr_int_f.shape[1]
ols_coeff_f = compute_ols_coeff(X_tr_int_f, y_log[train_mask])
dis_tr_f, dis_te_f = build_image_tensors(coords_tr_f, X_tr_sc_f, coords_te_f, X_te_sc_f)
pr("Tensores listos.")

# ── Loop sobre seeds ──────────────────────────────────────────────────────────
records = []
for seed in SEEDS:
    pr(f"\n{'='*60}\n[GSAWR Replica seed={seed}]\n{'='*60}")
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda": torch.cuda.manual_seed(seed)

    rng_f = np.random.default_rng(seed)
    val_m = rng_f.random(len(train_idx)) < 0.10; tr_m = ~val_m

    train_loader_f = make_loader(dis_tr_f[tr_m],  X_tr_int_f[tr_m],  y_log[train_mask][tr_m],  BATCH_SIZE, True)
    val_loader_f   = make_loader(dis_tr_f[val_m], X_tr_int_f[val_m], y_log[train_mask][val_m], BATCH_SIZE, False)
    test_loader_f  = make_loader(dis_te_f, X_te_int_f, y_log[test_mask], BATCH_SIZE, False)

    model_f = GSAWRModel(
        n_feat=n_feat_f, n_features=n_features_f, rows=GRID_ROWS, cols=GRID_COLS,
        afcnn_hidden=AFCNN_HIDDEN, sajpnn_hidden=SAJPNN_HIDDEN,
        cnn_channels=CNN_CHANNELS, kernel_size=CNN_KERNEL, dense_layers=DENSE_LAYERS,
        ols_coeff=ols_coeff_f,
    ).to(DEVICE)
    pr(f"  Params: {sum(p.numel() for p in model_f.parameters() if p.requires_grad):,}  Entrenando...")

    mp_f = MODELS_DIR / f"_replica_gsawr_seed{seed}.pt"
    train_model(model_f, train_loader_f, val_loader_f, mp_f)

    pred_log_test  = predict_model(model_f, test_loader_f)
    pred_orig_test = np.exp(pred_log_test)
    m = compute_metrics(y_orig[test_mask], pred_orig_test, y_log[test_mask], pred_log_test)
    pr(f"  seed={seed}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
       f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}  ({time.time()-t0:.1f}s)")

    moran_I = float("nan")
    if LIBPYSAL_AVAILABLE:
        resid_test = y_log[test_mask] - pred_log_test
        moran_I, _ = compute_moran_i(resid_test, coords_te_f)
        pr(f"  Moran I Holdout: I={moran_I:.4f}")

    records.append({"modelo":"GSAWR","seed":seed,"n_test":len(test_idx),
                    **m, "Moran_I_holdout": round(moran_I, 6)})

    mp_f.unlink(missing_ok=True)
    del model_f, train_loader_f, val_loader_f, test_loader_f
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

# ── Resumen ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "gsawr_log_replicas.csv", index=False)

pr(f"\n{'='*60}\nRESUMEN REPLICAS GSAWR\n{'='*60}")
pr(df[["seed","RMSE","R2","Moran_I_holdout"]].to_string(index=False))
for col in ["RMSE","MAE","MAPE","R2"]:
    pr(f"  {col}: mean={df[col].mean():.4f}  std={df[col].std():.4f}")

summary = {
    "modelo": "GSAWR",
    "seeds": str(SEEDS),
    "n_replicas": len(SEEDS),
    "RMSE_mean": round(df["RMSE"].mean(), 4),
    "RMSE_std":  round(df["RMSE"].std(),  4),
    "MAE_mean":  round(df["MAE"].mean(),  4),
    "MAE_std":   round(df["MAE"].std(),   4),
    "MAPE_mean": round(df["MAPE"].mean(), 4),
    "MAPE_std":  round(df["MAPE"].std(),  4),
    "R2_mean":   round(df["R2"].mean(),   4),
    "R2_std":    round(df["R2"].std(),    4),
}
pd.DataFrame([summary]).to_csv(OUT_DIR / "gsawr_log_replicas_summary.csv", index=False)
pr(f"\nRMSE: {summary['RMSE_mean']:.2f} ± {summary['RMSE_std']:.2f}")
pr(f"R2  : {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}")
pr(f"\n[CSV] {OUT_DIR}/gsawr_log_replicas.csv")
