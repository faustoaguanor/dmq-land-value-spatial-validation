"""
sannwr_real_log_cv_replicas.py — Réplicas CV del SANNWR CANÓNICO (Ni et al. 2022)
==================================================================================
Resuelve los hallazgos adversariales #8 (SANNWR canónico solo 1 seed) y #5 (smearing
ausente en CV) PARA EL MODELO CANÓNICO: réplicas CV (RandomKFold + SpatialBlock) con
5 seeds, métricas en USD/m² con corrección de smearing de Duan calculada DENTRO de
cada fold (residuales del train de ese fold, sin leakage de test).

Arquitectura IDÉNTICA a sannwr_real_log.py (distancia híbrida alpha=0.5, SWNN n_tr->n_feat).
Estrategias RandomKFold + SpatialBlock (mismas que los otros *_cv_replicas, para que la
Tabla 2 sea comparable; SpatialBlock_buf con 1 seed sigue en sannwr_real_log.py).

Salidas: modelos/sannwr/output_log_real/
  sannwr_real_log_cv_replicas.csv          -- por seed x estrategia x fold
  sannwr_real_log_cv_replicas_seed.csv     -- RMSE_mean ± std por seed x estrategia
  sannwr_real_log_cv_replicas_summary.csv  -- RMSE_mean ± std por estrategia sobre seeds
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

sys.path.insert(0, str(ROOT / "spatial_cv"))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV
sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix

# ── Config IDÉNTICA a sannwr_real_log.py ───────────────────────────────────────
ALPHA = 0.5
DENSE_LAYERS = [2048, 1024, 512, 256, 64]; DROP_OUT = 0.2; BATCH_NORM = True
N_EPOCHS = 1000; PATIENCE = 200; BATCH_SIZE = 64
START_LR = 0.2; RANDOM_STATE = 42
SEEDS = [42, 2011, 456, 777, 2026]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH  = ROOT / "datos" / "dataset.gpkg"
SPLIT_PATH = ROOT / "data_split" / "split.csv"
FOLDS_PATH = ROOT / "spatial_cv" / "output" / "fold_assignments.csv"
OUT_DIR    = _HERE / "output_log_real"
TMP_DIR    = OUT_DIR / "models"; OUT_DIR.mkdir(exist_ok=True); TMP_DIR.mkdir(exist_ok=True)

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

def smearing_factor(y_train_log, y_pred_train_log):
    """s_M = mean(exp(e_train)), e_train en escala log (Duan 1983).
    Calculado SOLO con residuales de training de este fold (sin leakage de test)."""
    e = y_train_log - y_pred_train_log
    return float(np.mean(np.exp(e)))

def metrics_log(y_orig, y_pred_log, y_log, s_M=1.0):
    yp = np.exp(y_pred_log) * s_M; e = y_orig - yp
    mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2)))
    mape=float(np.mean(np.abs(e/y_orig))*100)
    r2=round(1-np.sum(e**2)/np.sum((y_orig-y_orig.mean())**2),4)
    el=y_log-y_pred_log
    r2l=round(1-np.sum(el**2)/np.sum((y_log-y_log.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l,"smearing_factor":round(s_M,6)}

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
        if ep % 100 == 0 or ep == 1: pr(f"      ep {ep:4d}  train={tl:.4f}  val={vl:.4f}")
        if vl < best: best = vl; pat = 0; torch.save(model.state_dict(), path)
        else:
            pat += 1
            if pat >= PATIENCE: pr(f"      Early stop ep {ep}"); break
    model.load_state_dict(torch.load(path, map_location=DEVICE))

def predict(model, loader):
    model.eval(); ps = []
    with torch.no_grad():
        for dis, x, _ in loader:
            ps.append(model(dis.to(DEVICE), x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(ps)

# ── Cargar datos ───────────────────────────────────────────────────────────────
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT_PATH); folds_df = pd.read_csv(FOLDS_PATH)
for d in (gdf, split_df, folds_df): d["predio_join"] = d["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join","fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y]).astype(float)
y_orig = gdf["valor_m2"].values.astype(float); y_log = np.log(y_orig)
X_raw, FEAT = build_feature_matrix(gdf)
sp_folds = gdf["fold"].values.astype(int)
train_mask = (gdf["split"]=="train").values
train_idx = np.where(train_mask)[0]
pr(f"SANNWR-real CV replicas (alpha={ALPHA})  device={DEVICE}  train={len(train_idx)}  seeds={SEEDS}")

cv_rand = RandomKFoldCV(n_splits=5, random_state=RANDOM_STATE)
cv_block = SpatialBlockCV(folds=sp_folds[train_mask])
splits_r = list(cv_rand.split(X_raw[train_mask]))
splits_b = list(cv_block.split(X_raw[train_mask]))

# ── Pre-calcular datos de cada fold (fuera del loop de seeds) ──────────────────
pr("\nPre-calculando distancias híbridas por fold ...")
fold_cache = {}
for strat, splits in [("RandomKFold", splits_r), ("SpatialBlock", splits_b)]:
    fold_cache[strat] = []
    for fid, (tr, te) in enumerate(splits):
        t0 = time.time()
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_raw[train_mask][tr]); Xte = sc.transform(X_raw[train_mask][te])
        ytr = y_log[train_mask][tr]; ytel = y_log[train_mask][te]; yteo = y_orig[train_mask][te]
        ctr = coords[train_mask][tr]; cte = coords[train_mask][te]
        Xtr_i = add_intercept(Xtr); Xte_i = add_intercept(Xte); n_feat = Xtr_i.shape[1]
        ols_c = compute_ols(Xtr_i, ytr)
        sc_sp, sc_at = fit_scalers(ctr, Xtr)
        dis_tr = hybrid_dist(ctr, Xtr, ctr, Xtr, sc_sp, sc_at)
        dis_te = hybrid_dist(cte, Xte, ctr, Xtr, sc_sp, sc_at)
        rng = np.random.default_rng(RANDOM_STATE + fid); vm = rng.random(len(tr)) < 0.10; tm = ~vm
        fold_cache[strat].append({
            "fid":fid,"n_tr":len(tr),
            "Xtr_i":Xtr_i.astype(np.float32),"Xte_i":Xte_i.astype(np.float32),
            "ytr":ytr.astype(np.float32),"ytel":ytel.astype(np.float32),"yteo":yteo,
            "dis_tr":dis_tr,"dis_te":dis_te,"ols_c":ols_c,"n_feat":n_feat,"vm":vm,"tm":tm,
        })
        pr(f"  {strat} fold{fid+1}: {len(tr)}tr/{len(te)}te  ({time.time()-t0:.0f}s)")

# ── Loop seeds ─────────────────────────────────────────────────────────────────
records_fold = []; records_seed = []
for seed in SEEDS:
    pr(f"\n{'='*65}\n[SANNWR-real CV Replica seed={seed}]\n{'='*65}")
    for strat in ["RandomKFold", "SpatialBlock"]:
        fold_metrics = []
        for fd in fold_cache[strat]:
            fid = fd["fid"]
            pr(f"\n  [{strat}] fold{fid+1}  seed={seed}  n_tr={fd['n_tr']}")
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            if DEVICE.type == "cuda": torch.cuda.manual_seed(seed)
            tr_ld = make_loader(fd["dis_tr"][fd["tm"]], fd["Xtr_i"][fd["tm"]], fd["ytr"][fd["tm"]], BATCH_SIZE, True)
            val_ld = make_loader(fd["dis_tr"][fd["vm"]], fd["Xtr_i"][fd["vm"]], fd["ytr"][fd["vm"]], BATCH_SIZE, False)
            te_ld = make_loader(fd["dis_te"], fd["Xte_i"], fd["ytel"], BATCH_SIZE, False)
            fulltr_ld = make_loader(fd["dis_tr"], fd["Xtr_i"], fd["ytr"], BATCH_SIZE, False)
            mp = TMP_DIR / f"_sannwr_real_cv_{strat}_fold{fid}_seed{seed}.pt"
            model = SANNWRRealModel(fd["n_tr"], fd["n_feat"], DENSE_LAYERS, DROP_OUT, BATCH_NORM, fd["ols_c"]).to(DEVICE)
            train_model(model, tr_ld, val_ld, mp)
            pred_tr = predict(model, fulltr_ld)
            s_M = smearing_factor(fd["ytr"], pred_tr)
            pred = predict(model, te_ld)
            m = metrics_log(fd["yteo"], pred, fd["ytel"], s_M=s_M)
            mp.unlink(missing_ok=True)
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            pr(f"  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  s_M={s_M:.4f}  ({time.time()-t0:.0f}s)")
            records_fold.append({"seed":seed,"estrategia":strat,"fold":fid,**m})
            fold_metrics.append(m)
            pd.DataFrame(records_fold).to_csv(OUT_DIR/"sannwr_real_log_cv_replicas.csv", index=False)

        rmse_vals = [x["RMSE"] for x in fold_metrics]; r2_vals = [x["R2"] for x in fold_metrics]
        row = {"seed":seed,"estrategia":strat,
               "RMSE_mean":round(np.mean(rmse_vals),4),"RMSE_std":round(np.std(rmse_vals,ddof=1),4),
               "R2_mean":round(np.mean(r2_vals),4),"R2_std":round(np.std(r2_vals,ddof=1),4)}
        records_seed.append(row)
        pr(f"\n  [{strat}] seed={seed}  RMSE={row['RMSE_mean']:.2f}+-{row['RMSE_std']:.2f}")

df_fold = pd.DataFrame(records_fold); df_seed = pd.DataFrame(records_seed)
df_fold.to_csv(OUT_DIR/"sannwr_real_log_cv_replicas.csv", index=False)
df_seed.to_csv(OUT_DIR/"sannwr_real_log_cv_replicas_seed.csv", index=False)
meta = df_seed.groupby("estrategia").agg(
    n_seeds=("seed","count"),
    RMSE_mean=("RMSE_mean","mean"), RMSE_std=("RMSE_mean","std"),
    R2_mean=("R2_mean","mean"), R2_std=("R2_mean","std")).reset_index().round(4)
meta.to_csv(OUT_DIR/"sannwr_real_log_cv_replicas_summary.csv", index=False)
pr(f"\n{'='*65}\nRESUMEN FINAL SANNWR-real CV REPLICAS\n{'='*65}")
pr(meta.to_string(index=False))
pr(f"\n[CSV] {OUT_DIR}")
