"""
sannwr_real_log_replicas.py — Réplicas holdout del SANNWR CANÓNICO (Ni et al. 2022)
====================================================================================
Resuelve el hallazgo adversarial #8: el SANNWR canónico (sannwr_real_log.py) solo
corría con 1 seed, mientras la varianza reportada (Tabla 4) provenía de SANNWR* (la
variante de grilla). Este script entrena el canónico con 10 seeds sobre el holdout
20% fijo, para reportar media ± σ DEL MISMO modelo que se usa como "core".

Arquitectura IDÉNTICA a sannwr_real_log.py:
  d_hibrida(i,j) = alpha*d_espacial + (1-alpha)*d_atributiva   (alpha=0.5)
  SWNN(n_train -> n_features) modula coeficientes OLS fijos (como GNNWR).

NO aplica smearing (consistente con el resto de holdout-replicas / Tabla 4, que mide
ESTABILIDAD entre seeds; el factor de Duan es ~1.0 multiplicativo y no altera σ).

Salidas: modelos/sannwr/output_log_real/
  sannwr_real_log_replicas.csv          -- por seed
  sannwr_real_log_replicas_summary.csv  -- media ± σ sobre seeds
"""
from __future__ import annotations
import sys, time
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

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
if not (ROOT / "datos" / "dataset.gpkg").exists():
    ROOT = Path("/workspace/pod")

sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix

try:
    from libpysal.weights import KNN as KNNWeights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

# ── Config IDÉNTICA a sannwr_real_log.py ───────────────────────────────────────
ALPHA = 0.5
DENSE_LAYERS = [2048, 1024, 512, 256, 64]; DROP_OUT = 0.2; BATCH_NORM = True
N_EPOCHS = 1000; PATIENCE = 200; BATCH_SIZE = 64
START_LR = 0.2; RANDOM_STATE = 42; MORAN_K = 8
SEEDS = [42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH  = ROOT / "datos" / "dataset.gpkg"
SPLIT_PATH = ROOT / "data_split" / "split.csv"
OUT_DIR    = _HERE / "output_log_real"
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)

def pr(*a, **k): print(*a, **k, flush=True)

# ── Arquitectura (idéntica a sannwr_real_log.py) ───────────────────────────────
class SWNN(nn.Module):
    def __init__(self, insize, outsize, dense_layers, drop_out=0.2, batch_norm=True):
        super().__init__()
        act = nn.PReLU(init=0.1); layers, last = [], insize
        for h in dense_layers:
            layers += [nn.Linear(last, h)]
            if batch_norm: layers += [nn.BatchNorm1d(h)]
            layers += [act, nn.Dropout(drop_out)]; last = h
        layers += [nn.Linear(last, outsize)]
        self.fc = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self, x): return self.fc(x)

class SANNWRRealModel(nn.Module):
    def __init__(self, n_train, n_features, dense_layers, drop_out, batch_norm, ols_coeff):
        super().__init__()
        self.swnn = SWNN(n_train, n_features, dense_layers, drop_out, batch_norm=batch_norm)
        self.out  = nn.Linear(n_features, 1, bias=False)
        self.out.weight = nn.Parameter(
            torch.tensor(ols_coeff.reshape(1, -1), dtype=torch.float32), requires_grad=False)
    def forward(self, dis, x):
        return self.out(self.swnn(dis) * x)

# ── Funciones ───────────────────────────────────────────────────────────────────
def add_intercept(X): return np.column_stack([np.ones(len(X)), X])
def compute_ols(X_int, y):
    return LinearRegression(fit_intercept=False).fit(X_int, y).coef_.flatten().astype(np.float32)

def metrics_log(y_true_orig, y_pred_log, y_true_log):
    yp = np.exp(y_pred_log); e = y_true_orig - yp
    mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2)))
    mape=float(np.mean(np.abs(e/y_true_orig))*100)
    r2=round(1-np.sum(e**2)/np.sum((y_true_orig-y_true_orig.mean())**2),4)
    el=y_true_log-y_pred_log
    r2l=round(1-np.sum(el**2)/np.sum((y_true_log-y_true_log.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l}

def compute_moran(v, c, k=MORAN_K):
    w = KNNWeights.from_array(c, k=k); w.transform = "r"
    z = v - v.mean(); return float((z@(w.sparse@z))/(z@z))

def hybrid_dist(coords_q, X_q, coords_tr, X_tr, sc_sp, sc_at):
    d_sp = sc_sp.transform(cdist(coords_q, coords_tr))
    d_at = sc_at.transform(cdist(X_q, X_tr))
    return (ALPHA * d_sp + (1.0 - ALPHA) * d_at).astype(np.float32)

def fit_scalers(coords_tr, X_tr):
    sc_sp = StandardScaler().fit(cdist(coords_tr, coords_tr))
    sc_at = StandardScaler().fit(cdist(X_tr, X_tr))
    return sc_sp, sc_at

def make_loader(dis, X, y, bs, shuf):
    ds = TensorDataset(torch.tensor(dis, dtype=torch.float32),
                       torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y.reshape(-1,1), dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=shuf)

def one_epoch(model, loader, opt):
    training = opt is not None; model.train(training); tot, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for dis, x, y in loader:
            dis, x, y = dis.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
            yh = model(dis, x); loss = F.mse_loss(yh, y)
            if training:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()
            tot += loss.item()*len(y); n += len(y)
    return tot/n

def train_model(model, tr_ld, val_ld, path):
    opt = torch.optim.Adadelta(model.parameters(), lr=START_LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=3, eta_min=0.01)
    best, pat = float("inf"), 0
    for ep in range(1, N_EPOCHS+1):
        tl = one_epoch(model, tr_ld, opt); vl = one_epoch(model, val_ld, None); sch.step()
        if ep % 100 == 0 or ep == 1: pr(f"    ep {ep:4d}  train={tl:.4f}  val={vl:.4f}")
        if vl < best: best = vl; pat = 0; torch.save(model.state_dict(), path)
        else:
            pat += 1
            if pat >= PATIENCE: pr(f"    Early stop ep {ep}"); break
    model.load_state_dict(torch.load(path, map_location=DEVICE))

def predict(model, loader):
    model.eval(); ps = []
    with torch.no_grad():
        for dis, x, _ in loader:
            ps.append(model(dis.to(DEVICE), x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(ps)

# ── Cargar datos (one-hot, predio int, orden determinista) ────────────────────
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT_PATH)
for d in (gdf, split_df): d["predio_join"] = d["predio_join"].astype(int)
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y]).astype(float)
y_orig = gdf["valor_m2"].values.astype(float); y_log = np.log(y_orig)
X_raw, FEAT = build_feature_matrix(gdf)               # one-hot (30 feat)
train_mask = (gdf["split"]=="train").values; test_mask = (gdf["split"]=="test").values
train_idx = np.where(train_mask)[0]; test_idx = np.where(test_mask)[0]
pr(f"SANNWR-real REPLICAS (alpha={ALPHA})  device={DEVICE}  train={len(train_idx)}  test={len(test_idx)}  feat={X_raw.shape[1]}  seeds={SEEDS}")

# Preprocesamiento fijo (no depende de la seed del modelo)
scf = StandardScaler(); Xtr_f = scf.fit_transform(X_raw[train_mask]); Xte_f = scf.transform(X_raw[test_mask])
Xtr_if = add_intercept(Xtr_f); Xte_if = add_intercept(Xte_f); n_feat_f = Xtr_if.shape[1]
oc_f = compute_ols(Xtr_if, y_log[train_mask])
ctr_f = coords[train_mask]; cte_f = coords[test_mask]
sc_sp_f, sc_at_f = fit_scalers(ctr_f, Xtr_f)
dis_tr_f = hybrid_dist(ctr_f, Xtr_f, ctr_f, Xtr_f, sc_sp_f, sc_at_f)
dis_te_f = hybrid_dist(cte_f, Xte_f, ctr_f, Xtr_f, sc_sp_f, sc_at_f)

# ── Loop sobre seeds ────────────────────────────────────────────────────────────
records = []
for seed in SEEDS:
    pr(f"\n{'='*60}\n[SANNWR-real Replica seed={seed}]\n{'='*60}")
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == "cuda": torch.cuda.manual_seed_all(seed)
    rng_f = np.random.default_rng(seed); vm_f = rng_f.random(len(train_idx)) < 0.10; tm_f = ~vm_f
    tr_ld_f  = make_loader(dis_tr_f[tm_f], Xtr_if[tm_f], y_log[train_mask][tm_f], BATCH_SIZE, True)
    val_ld_f = make_loader(dis_tr_f[vm_f], Xtr_if[vm_f], y_log[train_mask][vm_f], BATCH_SIZE, False)
    te_ld_f  = make_loader(dis_te_f, Xte_if, y_log[test_mask], BATCH_SIZE, False)
    mp = MODELS_DIR / f"_replica_sannwr_real_seed{seed}.pt"
    t0 = time.time()
    model = SANNWRRealModel(len(train_idx), n_feat_f, DENSE_LAYERS, DROP_OUT, BATCH_NORM, oc_f).to(DEVICE)
    train_model(model, tr_ld_f, val_ld_f, mp)
    pred_log = predict(model, te_ld_f)
    m = metrics_log(y_orig[test_mask], pred_log, y_log[test_mask])
    moran = compute_moran(y_log[test_mask] - pred_log, cte_f) if LIBPYSAL_AVAILABLE else float("nan")
    pr(f"  seed={seed}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  Moran={moran:.4f}  ({time.time()-t0:.0f}s)")
    records.append({"modelo":"SANNWR-real","seed":seed,"n_test":len(test_idx),
                    **m, "Moran_I_holdout":round(moran,6)})
    mp.unlink(missing_ok=True)
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    # Guardado incremental (resume-safe si se corta el pod)
    pd.DataFrame(records).to_csv(OUT_DIR/"sannwr_real_log_replicas.csv", index=False)

df = pd.DataFrame(records)
df.to_csv(OUT_DIR/"sannwr_real_log_replicas.csv", index=False)
summ = pd.DataFrame([{
    "modelo":"SANNWR-real","seeds":str(SEEDS),"n_replicas":len(SEEDS),
    "RMSE_mean":round(df["RMSE"].mean(),4),"RMSE_std":round(df["RMSE"].std(),4),
    "MAE_mean":round(df["MAE"].mean(),4),"MAE_std":round(df["MAE"].std(),4),
    "MAPE_mean":round(df["MAPE"].mean(),4),"MAPE_std":round(df["MAPE"].std(),4),
    "R2_mean":round(df["R2"].mean(),4),"R2_std":round(df["R2"].std(),4),
}])
summ.to_csv(OUT_DIR/"sannwr_real_log_replicas_summary.csv", index=False)
pr(f"\n=== RESUMEN SANNWR-real réplicas holdout ===")
pr(df[["seed","RMSE","R2","Moran_I_holdout"]].to_string(index=False))
pr(summ[["RMSE_mean","RMSE_std","R2_mean"]].to_string(index=False))
pr(f"\n[CSV] {OUT_DIR}")
