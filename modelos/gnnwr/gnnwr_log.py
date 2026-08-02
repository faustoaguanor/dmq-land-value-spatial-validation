"""
gnnwr_log.py — GNNWR con log(valor_m2) + holdout 80/20
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
# Config OFICIAL Du et al. (2020), verificada vs runs/HIV/GNNWR_run.py de Xu et al. (2025):
# dense_layers stack completo + Adadelta — corrige la versión previa con capas truncadas.
DENSE_LAYERS = [2048, 1024, 512, 256, 64]; DROP_OUT = 0.2; BATCH_NORM = True
N_EPOCHS = 1000; PATIENCE = 200; BATCH_SIZE = 64
START_LR = 0.2; OPTIMIZER = "Adadelta"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42; MORAN_K = 8

BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR.parent.parent / "datos" / "dataset.gpkg"
SPLIT_PATH = BASE_DIR.parent.parent / "data_split" / "split.csv"
FOLDS_PATH = BASE_DIR.parent.parent / "spatial_cv" / "output" / "fold_assignments.csv"
OUT_DIR    = BASE_DIR / "output_log"
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)

# ── Clases (identicas a gnnwr_cv.py) ─────────────────────────────────────────
class SWNN(nn.Module):
    def __init__(self, insize, outsize, dense_layers=None, drop_out=0.2,
                 activate_func=None, batch_norm=True):
        super().__init__()
        if not dense_layers:
            dense_layers = []
            s = int(2**math.floor(math.log2(insize)))
            while s > outsize: dense_layers.append(s); s //= 2
        if activate_func is None: activate_func = nn.PReLU(init=0.1)
        layers, last = [], insize
        for h in dense_layers:
            layers += [nn.Linear(last, h)]
            if batch_norm: layers += [nn.BatchNorm1d(h)]
            layers += [activate_func, nn.Dropout(drop_out)]
            last = h
        layers += [nn.Linear(last, outsize)]
        self.fc = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self, x): return self.fc(x)

class GNNWRModel(nn.Module):
    def __init__(self, n_train, n_features, dense_layers, drop_out,
                 batch_norm, ols_coeff):
        super().__init__()
        self.swnn = SWNN(n_train, n_features, dense_layers, drop_out, batch_norm=batch_norm)
        self.out  = nn.Linear(n_features, 1, bias=False)
        self.out.weight = nn.Parameter(
            torch.tensor(ols_coeff.reshape(1,-1), dtype=torch.float32), requires_grad=False)
    def forward(self, dis, x):
        return self.out(self.swnn(dis) * x)

# ── Funciones ─────────────────────────────────────────────────────────────────
def add_intercept(X): return np.column_stack([np.ones(len(X)), X])

def compute_ols(X_int, y):
    m = LinearRegression(fit_intercept=False).fit(X_int, y)
    return m.coef_.flatten().astype(np.float32)

def metrics_log(y_true_orig, y_pred_log, y_true_log):
    y_pred_orig = np.exp(y_pred_log)
    e = y_true_orig - y_pred_orig
    mae = float(np.mean(np.abs(e))); rmse = float(np.sqrt(np.mean(e**2)))
    mape = float(np.mean(np.abs(e/y_true_orig))*100)
    ss_r = np.sum(e**2); ss_t = np.sum((y_true_orig-y_true_orig.mean())**2)
    r2 = round(1-ss_r/ss_t,4) if ss_t>0 else float("nan")
    el = y_true_log - y_pred_log
    ss_rl = np.sum(el**2); ss_tl = np.sum((y_true_log-y_true_log.mean())**2)
    r2l = round(1-ss_rl/ss_tl,4) if ss_tl>0 else float("nan")
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l}

def compute_moran(v, c, k=MORAN_K):
    w = KNNWeights.from_array(c, k=k); w.transform="r"
    z = v-v.mean(); Wz = w.sparse@z
    return float((z@Wz)/(z@z)), -1/(len(v)-1)

def make_loader(dis, X, y, bs, shuf):
    ds = TensorDataset(torch.tensor(dis,dtype=torch.float32),
                       torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y.reshape(-1,1),dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=shuf)

def one_epoch(model, loader, opt):
    training = opt is not None; model.train(training)
    tot, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for dis, x, y in loader:
            dis,x,y = dis.to(DEVICE),x.to(DEVICE),y.to(DEVICE)
            yh = model(dis,x); loss = F.mse_loss(yh,y)
            if training: opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()
            tot += loss.item()*len(y); n += len(y)
    return tot/n

def train_model(model, tr_loader, val_loader, path):
    opt = torch.optim.Adadelta(model.parameters(), lr=START_LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=100,T_mult=3,eta_min=0.01)
    best, pat = float("inf"), 0
    for ep in range(1, N_EPOCHS+1):
        tl = one_epoch(model, tr_loader, opt)
        vl = one_epoch(model, val_loader, None)
        sched.step()
        if ep%50==0 or ep==1:
            print(f"    ep {ep:4d}  train={tl:.4f}  val={vl:.4f}", flush=True)
        if vl < best: best=vl; pat=0; torch.save(model.state_dict(), path)
        else:
            pat+=1
            if pat>=PATIENCE: print(f"    Early stop ep {ep}", flush=True); break
    model.load_state_dict(torch.load(path, map_location=DEVICE))

def predict(model, loader):
    model.eval(); ps=[]
    with torch.no_grad():
        for dis,x,_ in loader:
            ps.append(model(dis.to(DEVICE),x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(ps)

# ── Cargar datos ──────────────────────────────────────────────────────────────
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
for df in [gdf]: df["predio_join"] = df["predio_join"].astype(int)
split_df = pd.read_csv(SPLIT_PATH); split_df["predio_join"]=split_df["predio_join"].astype(int)
folds_df = pd.read_csv(FOLDS_PATH); folds_df["predio_join"]=folds_df["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join","fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_orig = gdf["valor_m2"].values.astype(float)
y_log  = np.log(y_orig)
X_raw, FEAT_NAMES = build_feature_matrix(gdf)   # one-hot uso_suelo_cod (27->31)
sp_folds = gdf["fold"].values.astype(int)
train_mask = (gdf["split"]=="train").values; test_mask = (gdf["split"]=="test").values
train_idx = np.where(train_mask)[0]; test_idx = np.where(test_mask)[0]
print(f"GNNWR-log  device={DEVICE}  train={len(train_idx)}  test={len(test_idx)}")

cv_rand  = RandomKFoldCV(n_splits=5, random_state=RANDOM_STATE)
cv_block = SpatialBlockCV(folds=sp_folds[train_mask])
splits_r = list(cv_rand.split(X_raw[train_mask]))
splits_b = list(cv_block.split(X_raw[train_mask]))
BUFFER_M = 2530  # rango residual exponencial (~2.5 km): SpatialBlock con separacion train-test garantizada (Roberts et al. 2017)
cv_block_buf = SpatialBlockBufferedCV(folds=sp_folds[train_mask], coords=coords[train_mask], buffer=BUFFER_M)
splits_bb = list(cv_block_buf.split(X_raw[train_mask]))
all_records=[]; moran_records=[]

# ── CV ────────────────────────────────────────────────────────────────────────
for strat, splits in [("RandomKFold",splits_r),("SpatialBlock",splits_b),("SpatialBlock_buf",splits_bb)]:
    print(f"\n{'='*55}\n[GNNWR-log / {strat}]\n{'='*55}")
    y_pred_oos = np.full(len(train_idx), np.nan)

    for fid, (tr, te) in enumerate(splits):
        print(f"\n  fold {fid+1}/{len(splits)}  train={len(tr)}  test={len(te)}", flush=True)
        t0 = time.time()
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_raw[train_mask][tr])
        Xte = sc.transform(X_raw[train_mask][te])
        ytr = y_log[train_mask][tr]; yte_l = y_log[train_mask][te]
        yte_o = y_orig[train_mask][te]
        ctr = coords[train_mask][tr]; cte = coords[train_mask][te]

        Xtr_i = add_intercept(Xtr); Xte_i = add_intercept(Xte)
        ols_c = compute_ols(Xtr_i, ytr); n_feat = Xtr_i.shape[1]

        print("    Calculando distancias ...", flush=True)
        dis_tr = cdist(ctr, ctr); dis_te = cdist(cte, ctr)
        ds = StandardScaler()
        dis_tr_sc = ds.fit_transform(dis_tr); dis_te_sc = ds.transform(dis_te)

        rng = np.random.default_rng(RANDOM_STATE+fid)
        vm = rng.random(len(tr)) < 0.10; tm = ~vm

        tr_ld  = make_loader(dis_tr_sc[tm], Xtr_i[tm], ytr[tm], BATCH_SIZE, True)
        val_ld = make_loader(dis_tr_sc[vm], Xtr_i[vm], ytr[vm], BATCH_SIZE, False)
        te_ld  = make_loader(dis_te_sc, Xte_i, yte_l, BATCH_SIZE, False)

        mp = MODELS_DIR/f"gnnwr_log_{strat}_fold{fid}.pt"
        model = GNNWRModel(len(tr), n_feat, DENSE_LAYERS, DROP_OUT, BATCH_NORM, ols_c).to(DEVICE)
        print("    Entrenando ...", flush=True)
        train_model(model, tr_ld, val_ld, mp)

        pred_log = predict(model, te_ld)
        y_pred_oos[te] = pred_log
        m = metrics_log(yte_o, pred_log, yte_l)
        print(f"    MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        all_records.append({"modelo":"GNNWR","estrategia":strat,"fold":fid,**m})

    if LIBPYSAL_AVAILABLE:
        mask = ~np.isnan(y_pred_oos)
        resid = y_log[train_mask][mask] - y_pred_oos[mask]
        I,EI = compute_moran(resid, coords[train_mask][mask])
        print(f"  Moran I OOS {strat}: I={I:.4f}", flush=True)
        moran_records.append({"modelo":"GNNWR","estrategia":strat,"I":round(I,6),"EI":round(EI,6)})

# ── Modelo final → test 20% ───────────────────────────────────────────────────
print(f"\n{'='*55}\n[GNNWR-log / Holdout final]\n{'='*55}")
# Re-seed para holdout CONTROLADO (init independiente del RNG post-CV).
import random as _rnd_hold
_rnd_hold.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_STATE)
sc_f = StandardScaler()
Xtr_f = sc_f.fit_transform(X_raw[train_mask]); Xte_f = sc_f.transform(X_raw[test_mask])
Xtr_if = add_intercept(Xtr_f); Xte_if = add_intercept(Xte_f)
ols_cf = compute_ols(Xtr_if, y_log[train_mask]); n_feat_f = Xtr_if.shape[1]
ctr_f = coords[train_mask]; cte_f = coords[test_mask]
print("  Calculando distancias finales ...", flush=True)
dis_tr_f = cdist(ctr_f, ctr_f); dis_te_f = cdist(cte_f, ctr_f)
ds_f = StandardScaler()
dis_tr_f_sc = ds_f.fit_transform(dis_tr_f); dis_te_f_sc = ds_f.transform(dis_te_f)
rng_f = np.random.default_rng(RANDOM_STATE)
vm_f = rng_f.random(len(train_idx)) < 0.10; tm_f = ~vm_f
tr_ld_f  = make_loader(dis_tr_f_sc[tm_f], Xtr_if[tm_f], y_log[train_mask][tm_f], BATCH_SIZE, True)
val_ld_f = make_loader(dis_tr_f_sc[vm_f], Xtr_if[vm_f], y_log[train_mask][vm_f], BATCH_SIZE, False)
te_ld_f  = make_loader(dis_te_f_sc, Xte_if, y_log[test_mask], BATCH_SIZE, False)
mp_f = MODELS_DIR/"gnnwr_log_final.pt"
model_f = GNNWRModel(len(train_idx), n_feat_f, DENSE_LAYERS, DROP_OUT, BATCH_NORM, ols_cf).to(DEVICE)
print("  Entrenando modelo final ...", flush=True)
train_model(model_f, tr_ld_f, val_ld_f, mp_f)
pred_log_test = predict(model_f, te_ld_f)
m_test = metrics_log(y_orig[test_mask], pred_log_test, y_log[test_mask])
print(f"  HOLDOUT: MAE={m_test['MAE']:.2f}  RMSE={m_test['RMSE']:.2f}  "
      f"MAPE={m_test['MAPE']:.2f}%  R2={m_test['R2']:.4f}  R2_log={m_test['R2_log']:.4f}", flush=True)
if LIBPYSAL_AVAILABLE:
    resid_t = y_log[test_mask] - pred_log_test
    Ih, EIh = compute_moran(resid_t, cte_f)
    print(f"  Moran I Holdout: I={Ih:.4f}", flush=True)
    moran_records.append({"modelo":"GNNWR","estrategia":"Holdout20%","I":round(Ih,6),"EI":round(EIh,6)})

# ── Guardar ───────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(all_records)
sum_df = res_df.groupby(["modelo","estrategia"]).agg(
    MAE_mean=("MAE","mean"),MAE_std=("MAE","std"),
    RMSE_mean=("RMSE","mean"),RMSE_std=("RMSE","std"),
    MAPE_mean=("MAPE","mean"),MAPE_std=("MAPE","std"),
    R2_mean=("R2","mean"),R2_std=("R2","std")).reset_index().round(4)
res_df.to_csv(OUT_DIR/"gnnwr_log_results.csv", index=False)
sum_df.to_csv(OUT_DIR/"gnnwr_log_summary.csv", index=False)
pd.DataFrame([{"modelo":"GNNWR","estrategia":"Holdout20%","n_test":len(test_idx),**m_test}]
             ).to_csv(OUT_DIR/"gnnwr_log_holdout.csv", index=False)

# Predicciones por predio (test + train) para smearing / bootstrap / CDF / mapa
tr_ld_full = make_loader(dis_tr_f_sc, Xtr_if, y_log[train_mask], BATCH_SIZE, False)
pred_log_train_all = predict(model_f, tr_ld_full)
preds = pd.concat([
    pd.DataFrame({"predio_join": gdf.loc[test_mask,"predio_join"].astype(int).values,
                  "split":"test", "y_obs_log":y_log[test_mask], "y_pred_log":pred_log_test}),
    pd.DataFrame({"predio_join": gdf.loc[train_mask,"predio_join"].astype(int).values,
                  "split":"train","y_obs_log":y_log[train_mask],"y_pred_log":pred_log_train_all}),
], ignore_index=True)
preds["modelo"] = "GNNWR"
preds.to_csv(OUT_DIR/"gnnwr_log_predictions.csv", index=False)
if moran_records:
    pd.DataFrame(moran_records).to_csv(OUT_DIR/"gnnwr_log_moran.csv", index=False)
print(f"\nRESUMEN:\n{sum_df.to_string(index=False)}")
print(f"HOLDOUT: {m_test}")
print(f"\n[CSV] {OUT_DIR}")
