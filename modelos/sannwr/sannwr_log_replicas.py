"""
sannwr_log_replicas.py — Réplicas SANNWR con seeds [42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7] — solo holdout 20%.
=======================================================================================
Evalúa estabilidad de SANNWR ante inicialización aleatoria, con la MISMA config que el
script principal (sannwr_log.py). Re-siembra torch.manual_seed(seed) antes de entrenar
el modelo final, para obtener un número controlado e independiente del estado del RNG
tras la CV (a diferencia del script principal, que entrena el holdout post-CV).
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

# ── Config IDÉNTICA a sannwr_log.py ────────────────────────────────────────────
COVARIABLES = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]
GRID_COLS=20; GRID_ROWS=20; N_GRID=400; IDW_POWER=2.0
SAPDNN_HIDDEN=64; DROP_OUT=0.2; BATCH_NORM=True
N_EPOCHS=300; PATIENCE=60; BATCH_SIZE=64; START_LR=0.001
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MORAN_K=8; SEEDS=[42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7]

BASE_DIR   = Path(__file__).parent
ROOT       = BASE_DIR.parent.parent
DATA_PATH  = ROOT/"datos"/"dataset.gpkg"
SPLIT_PATH = ROOT/"data_split"/"split.csv"
OUT_DIR    = BASE_DIR/"output_log"
MODELS_DIR = OUT_DIR/"models"
OUT_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)

def pr(*a, **k): print(*a, **k, flush=True)

# ── Clases (idénticas a sannwr_log.py) ─────────────────────────────────────────
class SAPDNN(nn.Module):
    def __init__(self, hidden, batch_norm=True):
        super().__init__()
        self.dnn = nn.Sequential()
        self.dnn.add_module("full1", nn.Linear(2, hidden, bias=False))
        if batch_norm: self.dnn.add_module("bn1", nn.BatchNorm1d(hidden))
        self.dnn.add_module("act1",  nn.PReLU(init=0.1))
        self.dnn.add_module("full2", nn.Linear(hidden, 1, bias=False))
        self.dnn.add_module("act2",  nn.PReLU(init=0.1))
        self.ln = nn.LayerNorm(N_GRID)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
    def forward(self, dis):
        b,n,c = dis.size()
        out = self.dnn(dis.reshape(-1,c)).reshape(b,n)
        return self.ln(out)

class SWNN(nn.Module):
    def __init__(self, insize, outsize, drop_out=0.2, batch_norm=True):
        super().__init__()
        layers, last = [], insize
        s = int(2**math.floor(math.log2(insize)))
        dense = []
        while s > outsize: dense.append(s); s //= 2
        act = nn.PReLU(init=0.1)
        for h in dense:
            layers += [nn.Linear(last,h)]
            if batch_norm: layers += [nn.BatchNorm1d(h)]
            layers += [act, nn.Dropout(drop_out)]
            last = h
        layers += [nn.Linear(last, outsize)]
        self.fc = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self, x): return self.fc(x)

class SANNWRModel(nn.Module):
    def __init__(self, n_grid, n_features, sapdnn_hidden, drop_out, batch_norm, ols_coeff):
        super().__init__()
        self.sapdnn = SAPDNN(sapdnn_hidden, batch_norm)
        self.swnn   = SWNN(n_grid, n_features, drop_out, batch_norm)
        self.out    = nn.Linear(n_features, 1, bias=False)
        self.out.weight = nn.Parameter(
            torch.tensor(ols_coeff.reshape(1,-1), dtype=torch.float32), requires_grad=False)
    def forward(self, dis, x):
        return self.out(self.swnn(self.sapdnn(dis)) * x)

# ── Funciones ───────────────────────────────────────────────────────────────────
def add_intercept(X): return np.column_stack([np.ones(len(X)), X])
def ols_coeff(X_int, y):
    return LinearRegression(fit_intercept=False).fit(X_int,y).coef_.flatten().astype(np.float32)

def metrics_log(y_true_orig, y_pred_log, y_true_log):
    yp = np.exp(y_pred_log); e = y_true_orig - yp
    mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2)))
    mape=float(np.mean(np.abs(e/y_true_orig))*100)
    r2=round(1-np.sum(e**2)/np.sum((y_true_orig-y_true_orig.mean())**2),4)
    el=y_true_log-y_pred_log
    r2l=round(1-np.sum(el**2)/np.sum((y_true_log-y_true_log.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l}

def compute_moran(v,c,k=MORAN_K):
    w=KNNWeights.from_array(c,k=k); w.transform="r"
    z=v-v.mean(); Wz=w.sparse@z
    return float((z@Wz)/(z@z))

def build_grid(coords_tr, cols, rows):
    xmn,ymn=coords_tr.min(0); xmx,ymx=coords_tr.max(0)
    dx=(xmx-xmn)*0.05; dy=(ymx-ymn)*0.05
    xl=np.linspace(xmn-dx,xmx+dx,cols); yl=np.linspace(ymn-dy,ymx+dy,rows)
    gx,gy=np.meshgrid(xl,yl)
    return np.column_stack([gx.ravel(),gy.ravel()])

def grid_attrs_idw(ctr,Xtr,gxy,p=IDW_POWER):
    d=cdist(gxy,ctr); w=1/(d**p+1e-10); w/=w.sum(1,keepdims=True); return w@Xtr

def grid_dists(cq,Xq,gxy,gattrs):
    return np.stack([cdist(cq,gxy), cdist(Xq,gattrs)], axis=-1)

def make_loader(dis,X,y,bs,shuf):
    ds=TensorDataset(torch.tensor(dis,dtype=torch.float32),
                     torch.tensor(X,dtype=torch.float32),
                     torch.tensor(y.reshape(-1,1),dtype=torch.float32))
    return DataLoader(ds,batch_size=bs,shuffle=shuf, drop_last=shuf)

def one_epoch(model,loader,opt):
    training=opt is not None; model.train(training)
    tot,n=0.0,0
    ctx=torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for dis,x,y in loader:
            dis,x,y=dis.to(DEVICE),x.to(DEVICE),y.to(DEVICE)
            yh=model(dis,x); loss=F.mse_loss(yh,y)
            if training: opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()
            tot+=loss.item()*len(y); n+=len(y)
    return tot/n

def train_model(model,tr_ld,val_ld,path):
    opt=torch.optim.Adam(model.parameters(),lr=START_LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2,eta_min=1e-5)
    best,pat=float("inf"),0
    for ep in range(1,N_EPOCHS+1):
        tl=one_epoch(model,tr_ld,opt); vl=one_epoch(model,val_ld,None); sch.step()
        if vl<best: best=vl; pat=0; torch.save(model.state_dict(),path)
        else:
            pat+=1
            if pat>=PATIENCE: break
    model.load_state_dict(torch.load(path,map_location=DEVICE))

def predict(model,loader):
    model.eval(); ps=[]
    with torch.no_grad():
        for dis,x,_ in loader:
            ps.append(model(dis.to(DEVICE),x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(ps)

# ── Cargar datos ──────────────────────────────────────────────────────────────
gdf=gpd.read_file(DATA_PATH,layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"]=gdf["predio_join"].astype(int)
split_df=pd.read_csv(SPLIT_PATH); split_df["predio_join"]=split_df["predio_join"].astype(int)
gdf=gdf.merge(split_df[["predio_join","split"]],on="predio_join",how="left")
coords=np.column_stack([gdf.geometry.x,gdf.geometry.y])
y_orig=gdf["valor_m2"].values.astype(float); y_log=np.log(y_orig)
import sys as _s; from pathlib import Path as _P
_s.path.insert(0, str(_P(__file__).resolve().parent.parent))
from features import build_feature_matrix
X_raw, _FEAT = build_feature_matrix(gdf)   # one-hot uso_suelo_cod (27->31)
train_mask=(gdf["split"]=="train").values; test_mask=(gdf["split"]=="test").values
train_idx=np.where(train_mask)[0]; test_idx=np.where(test_mask)[0]
pr(f"SANNWR replicas  device={DEVICE}  train={len(train_idx)}  test={len(test_idx)}")

# Preprocesamiento fijo (no depende de seed)
scf=StandardScaler()
Xtr_f=scf.fit_transform(X_raw[train_mask]); Xte_f=scf.transform(X_raw[test_mask])
Xtr_if=add_intercept(Xtr_f); Xte_if=add_intercept(Xte_f); n_feat_f=Xtr_if.shape[1]
oc_f=ols_coeff(Xtr_if,y_log[train_mask])
ctr_f=coords[train_mask]; cte_f=coords[test_mask]
gxy_f=build_grid(ctr_f,GRID_COLS,GRID_ROWS); gattrs_f=grid_attrs_idw(ctr_f,Xtr_f,gxy_f)
dis_tr_rf=grid_dists(ctr_f,Xtr_f,gxy_f,gattrs_f); dis_te_rf=grid_dists(cte_f,Xte_f,gxy_f,gattrs_f)
dsf=StandardScaler(); dsf.fit(dis_tr_rf.reshape(-1,2))
dis_tr_f=dsf.transform(dis_tr_rf.reshape(-1,2)).reshape(len(train_idx),N_GRID,2)
dis_te_f=dsf.transform(dis_te_rf.reshape(-1,2)).reshape(len(test_idx),N_GRID,2)

# ── Loop sobre seeds ────────────────────────────────────────────────────────────
records=[]
for seed in SEEDS:
    pr(f"\n{'='*60}\n[SANNWR Replica seed={seed}]\n{'='*60}")
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type=="cuda": torch.cuda.manual_seed_all(seed)
    rng_f=np.random.default_rng(seed); vm_f=rng_f.random(len(train_idx))<0.10; tm_f=~vm_f
    tr_ld_f=make_loader(dis_tr_f[tm_f],Xtr_if[tm_f],y_log[train_mask][tm_f],BATCH_SIZE,True)
    val_ld_f=make_loader(dis_tr_f[vm_f],Xtr_if[vm_f],y_log[train_mask][vm_f],BATCH_SIZE,False)
    te_ld_f=make_loader(dis_te_f,Xte_if,y_log[test_mask],BATCH_SIZE,False)
    mp=MODELS_DIR/f"_replica_sannwr_seed{seed}.pt"
    t0=time.time()
    model=SANNWRModel(N_GRID,n_feat_f,SAPDNN_HIDDEN,DROP_OUT,BATCH_NORM,oc_f).to(DEVICE)
    train_model(model,tr_ld_f,val_ld_f,mp)
    pred_log=predict(model,te_ld_f)
    m=metrics_log(y_orig[test_mask],pred_log,y_log[test_mask])
    moran=compute_moran(y_log[test_mask]-pred_log, cte_f) if LIBPYSAL_AVAILABLE else float("nan")
    pr(f"  seed={seed}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  Moran={moran:.4f}  ({time.time()-t0:.0f}s)")
    records.append({"modelo":"SANNWR","seed":seed,"n_test":len(test_idx),
                    **m,"Moran_I_holdout":round(moran,6)})

df=pd.DataFrame(records)
df.to_csv(OUT_DIR/"sannwr_log_replicas.csv",index=False)
summ=pd.DataFrame([{
    "modelo":"SANNWR","seeds":str(SEEDS),"n_replicas":len(SEEDS),
    "RMSE_mean":round(df["RMSE"].mean(),4),"RMSE_std":round(df["RMSE"].std(),4),
    "MAE_mean":round(df["MAE"].mean(),4),"MAE_std":round(df["MAE"].std(),4),
    "MAPE_mean":round(df["MAPE"].mean(),4),"MAPE_std":round(df["MAPE"].std(),4),
    "R2_mean":round(df["R2"].mean(),4),"R2_std":round(df["R2"].std(),4),
}])
summ.to_csv(OUT_DIR/"sannwr_log_replicas_summary.csv",index=False)
pr(f"\n=== RESUMEN SANNWR réplicas ===")
pr(df[["seed","RMSE","R2","Moran_I_holdout"]].to_string(index=False))
pr(summ[["RMSE_mean","RMSE_std","R2_mean"]].to_string(index=False))
