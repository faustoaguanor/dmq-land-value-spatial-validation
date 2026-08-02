"""
sannwr_log_cv_replicas.py
=========================
Replicas de CV (RandomKFold + SpatialBlock) para SANNWR con 5 seeds.
Para cada seed se corre CV completo (5 folds x 2 estrategias).

Salidas: modelos/sannwr/output_log/
  sannwr_log_cv_replicas.csv          -- por seed x estrategia x fold
  sannwr_log_cv_replicas_seed.csv     -- RMSE_mean+-std por seed x estrategia
  sannwr_log_cv_replicas_summary.csv  -- RMSE_mean+-std por estrategia sobre seeds
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

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
if not (ROOT / "datos" / "dataset.csv").exists():
    ROOT = Path("/workspace/pod")

sys.path.insert(0, str(ROOT / "spatial_cv"))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV

# ── Config IDENTICA a sannwr_log.py ───────────────────────────────────────────
COVARIABLES = [
    "suscept_codigo","pc_pnbi","dist_metro","dist_centr_metro",
    "dist_centr_zonal","dist_cc","dist_universidad","dist_hospital",
    "dist_parque_metro","dist_industrial","dist_via_principal",
    "uso_suelo_cod","cos_num","dist_quebrada","dist_mercado_mayorista",
    "dist_plataforma_gub","log_area","frente_m","area_const_m2",
    "tiene_const","num_pisos","antiguedad","topografia_factor",
    "conservacion_cod","acabados_cod","es_ph","pendiente_grados",
]
GRID_COLS=20; GRID_ROWS=20; N_GRID=400; IDW_POWER=2.0
SAPDNN_HIDDEN=64; DROP_OUT=0.2; BATCH_NORM=True
N_EPOCHS=300; PATIENCE=60; BATCH_SIZE=64; START_LR=0.001
RANDOM_STATE=42
SEEDS=[42,2011,456,777,2026]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH  = ROOT/"datos"/"dataset.gpkg"
SPLIT_PATH = ROOT/"data_split"/"split.csv"
FOLDS_PATH = ROOT/"spatial_cv"/"output"/"fold_assignments.csv"
OUT_DIR    = ROOT/"modelos"/"sannwr"/"output_log"
TMP_DIR    = OUT_DIR/"models"; OUT_DIR.mkdir(exist_ok=True); TMP_DIR.mkdir(exist_ok=True)

# ── Arquitectura IDENTICA a sannwr_log.py ─────────────────────────────────────
class SAPDNN(nn.Module):
    def __init__(self,hidden,batch_norm=True):
        super().__init__()
        self.dnn=nn.Sequential()
        self.dnn.add_module("full1",nn.Linear(2,hidden,bias=False))
        if batch_norm: self.dnn.add_module("bn1",nn.BatchNorm1d(hidden))
        self.dnn.add_module("act1",nn.PReLU(init=0.1))
        self.dnn.add_module("full2",nn.Linear(hidden,1,bias=False))
        self.dnn.add_module("act2",nn.PReLU(init=0.1))
        self.ln=nn.LayerNorm(N_GRID)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
    def forward(self,dis):
        b,n,c=dis.size(); out=self.dnn(dis.reshape(-1,c)).reshape(b,n); return self.ln(out)

class SWNN(nn.Module):
    def __init__(self,insize,outsize,drop_out=0.2,batch_norm=True):
        super().__init__()
        layers=[]; last=insize; s=int(2**math.floor(math.log2(insize))); dense=[]
        while s>outsize: dense.append(s); s//=2
        act=nn.PReLU(init=0.1)
        for h in dense:
            layers+=[nn.Linear(last,h)]
            if batch_norm: layers+=[nn.BatchNorm1d(h)]
            layers+=[act,nn.Dropout(drop_out)]; last=h
        layers+=[nn.Linear(last,outsize)]
        self.fc=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self,x): return self.fc(x)

class SANNWRModel(nn.Module):
    def __init__(self,n_grid,n_features,sapdnn_hidden,drop_out,batch_norm,ols_coeff):
        super().__init__()
        self.sapdnn=SAPDNN(sapdnn_hidden,batch_norm)
        self.swnn=SWNN(n_grid,n_features,drop_out,batch_norm)
        self.out=nn.Linear(n_features,1,bias=False)
        self.out.weight=nn.Parameter(
            torch.tensor(ols_coeff.reshape(1,-1),dtype=torch.float32),requires_grad=False)
    def forward(self,dis,x): return self.out(self.swnn(self.sapdnn(dis))*x)

# ── Helpers ────────────────────────────────────────────────────────────────────
def add_intercept(X): return np.column_stack([np.ones(len(X)),X])
def ols_coeff(Xi,y):
    return LinearRegression(fit_intercept=False).fit(Xi,y).coef_.flatten().astype(np.float32)
def smearing_factor(y_train_log, y_pred_train_log):
    """s_M = mean(exp(e_train)), e_train en escala log (Duan 1983).
    Calculado SOLO con residuales de training de este fold (sin leakage de test)."""
    e = y_train_log - y_pred_train_log
    return float(np.mean(np.exp(e)))

def metrics_log(y_orig,y_pred_log,y_log,s_M=1.0):
    yp=np.exp(y_pred_log)*s_M; e=y_orig-yp
    mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2)))
    mape=float(np.mean(np.abs(e/y_orig))*100)
    r2=round(1-np.sum(e**2)/np.sum((y_orig-y_orig.mean())**2),4)
    el=y_log-y_pred_log
    r2l=round(1-np.sum(el**2)/np.sum((y_log-y_log.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l,"smearing_factor":round(s_M,6)}
def build_grid(ctr,cols,rows):
    xmn,ymn=ctr.min(0); xmx,ymx=ctr.max(0)
    dx=(xmx-xmn)*0.05; dy=(ymx-ymn)*0.05
    xl=np.linspace(xmn-dx,xmx+dx,cols); yl=np.linspace(ymn-dy,ymx+dy,rows)
    gx,gy=np.meshgrid(xl,yl); return np.column_stack([gx.ravel(),gy.ravel()])
def grid_attrs_idw(ctr,Xtr,gxy,p=IDW_POWER):
    d=cdist(gxy,ctr); w=1/(d**p+1e-10); w/=w.sum(1,keepdims=True); return w@Xtr
def grid_dists(cq,Xq,gxy,gattrs):
    return np.stack([cdist(cq,gxy),cdist(Xq,gattrs)],axis=-1)
def make_loader(dis,X,y,bs,shuf):
    ds=TensorDataset(torch.tensor(dis,dtype=torch.float32),
                     torch.tensor(X,dtype=torch.float32),
                     torch.tensor(y.reshape(-1,1),dtype=torch.float32))
    return DataLoader(ds,batch_size=bs,shuffle=shuf, drop_last=shuf)
def one_epoch(model,loader,opt):
    training=opt is not None; model.train(training); tot,n=0.0,0
    ctx=torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for dis,x,y in loader:
            dis,x,y=dis.to(DEVICE),x.to(DEVICE),y.to(DEVICE)
            yh=model(dis,x); loss=F.mse_loss(yh,y)
            if training: opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0); opt.step()
            tot+=loss.item()*len(y); n+=len(y)
    return tot/n
def train_model(model,tr_ld,val_ld,path):
    opt=torch.optim.Adam(model.parameters(),lr=START_LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2,eta_min=1e-5)
    best,pat=float("inf"),0
    for ep in range(1,N_EPOCHS+1):
        tl=one_epoch(model,tr_ld,opt); vl=one_epoch(model,val_ld,None); sch.step()
        if ep%100==0 or ep==1: print(f"      ep{ep:4d} train={tl:.4f} val={vl:.4f}",flush=True)
        if vl<best: best=vl; pat=0; torch.save(model.state_dict(),path)
        else:
            pat+=1
            if pat>=PATIENCE: print(f"      Early stop ep{ep}",flush=True); break
    model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=False))
def predict(model,loader):
    model.eval(); ps=[]
    with torch.no_grad():
        for dis,x,_ in loader:
            ps.append(model(dis.to(DEVICE),x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(ps)

# ── Cargar datos ───────────────────────────────────────────────────────────────
print(f"ROOT={ROOT}",flush=True)
gdf=gpd.read_file(DATA_PATH,layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"]=gdf["predio_join"].astype(int)
split_df=pd.read_csv(SPLIT_PATH); split_df["predio_join"]=split_df["predio_join"].astype(int)
folds_df=pd.read_csv(FOLDS_PATH); folds_df["predio_join"]=folds_df["predio_join"].astype(int)
gdf=gdf.merge(folds_df[["predio_join","fold"]],on="predio_join",how="left")
gdf=gdf.merge(split_df[["predio_join","split"]],on="predio_join",how="left")
coords=np.column_stack([gdf.geometry.x,gdf.geometry.y])
y_orig=gdf["valor_m2"].values.astype(float); y_log=np.log(y_orig)
import sys as _s; from pathlib import Path as _P
_s.path.insert(0, str(_P(__file__).resolve().parent.parent))
from features import build_feature_matrix
X_raw, _FEAT = build_feature_matrix(gdf)   # one-hot uso_suelo_cod (27->31)
sp_folds=gdf["fold"].values.astype(int)
train_mask=(gdf["split"]=="train").values
train_idx=np.where(train_mask)[0]
print(f"SANNWR CV replicas  device={DEVICE}  train={len(train_idx)}  seeds={SEEDS}",flush=True)

cv_rand=RandomKFoldCV(n_splits=5,random_state=RANDOM_STATE)
cv_block=SpatialBlockCV(folds=sp_folds[train_mask])
splits_r=list(cv_rand.split(X_raw[train_mask]))
splits_b=list(cv_block.split(X_raw[train_mask]))

# ── Pre-calcular datos de cada fold ───────────────────────────────────────────
print("\nPre-calculando datos de folds ...",flush=True)
fold_cache={}
for strat,splits in [("RandomKFold",splits_r),("SpatialBlock",splits_b)]:
    fold_cache[strat]=[]
    for fid,(tr,te) in enumerate(splits):
        t0=time.time()
        sc=StandardScaler()
        Xtr=sc.fit_transform(X_raw[train_mask][tr])
        Xte=sc.transform(X_raw[train_mask][te])
        ytr=y_log[train_mask][tr]; ytel=y_log[train_mask][te]; yteo=y_orig[train_mask][te]
        ctr=coords[train_mask][tr]; cte=coords[train_mask][te]
        Xtr_i=add_intercept(Xtr); Xte_i=add_intercept(Xte); n_feat=Xtr_i.shape[1]
        oc=ols_coeff(Xtr_i,ytr)
        gxy=build_grid(ctr,GRID_COLS,GRID_ROWS)
        gattrs=grid_attrs_idw(ctr,Xtr,gxy)
        dis_tr_r=grid_dists(ctr,Xtr,gxy,gattrs); dis_te_r=grid_dists(cte,Xte,gxy,gattrs)
        ds=StandardScaler(); ds.fit(dis_tr_r.reshape(-1,2))
        dis_tr=ds.transform(dis_tr_r.reshape(-1,2)).reshape(len(tr),N_GRID,2).astype(np.float32)
        dis_te=ds.transform(dis_te_r.reshape(-1,2)).reshape(len(te),N_GRID,2).astype(np.float32)
        rng=np.random.default_rng(RANDOM_STATE+fid); vm=rng.random(len(tr))<0.10; tm=~vm
        fold_cache[strat].append({
            "fid":fid,"n_tr":len(tr),
            "Xtr_i":Xtr_i.astype(np.float32),"Xte_i":Xte_i.astype(np.float32),
            "ytr":ytr.astype(np.float32),"ytel":ytel.astype(np.float32),"yteo":yteo,
            "dis_tr":dis_tr,"dis_te":dis_te,"oc":oc,"n_feat":n_feat,"vm":vm,"tm":tm,
        })
        print(f"  {strat} fold{fid+1}: {len(tr)}tr/{len(te)}te  ({time.time()-t0:.0f}s)",flush=True)

# ── Loop seeds ─────────────────────────────────────────────────────────────────
records_fold=[]; records_seed=[]

for seed in SEEDS:
    print(f"\n{'='*65}\n[SANNWR CV Replica seed={seed}]\n{'='*65}",flush=True)
    for strat in ["RandomKFold","SpatialBlock"]:
        fold_metrics=[]
        for fd in fold_cache[strat]:
            fid=fd["fid"]
            print(f"\n  [{strat}] fold{fid+1}  seed={seed}  n_tr={fd['n_tr']}",flush=True)
            t0=time.time()

            torch.manual_seed(seed); np.random.seed(seed); import random as _r; _r.seed(seed)
            if DEVICE.type=="cuda": torch.cuda.manual_seed(seed)

            tr_ld=make_loader(fd["dis_tr"][fd["tm"]],fd["Xtr_i"][fd["tm"]],fd["ytr"][fd["tm"]],BATCH_SIZE,True)
            val_ld=make_loader(fd["dis_tr"][fd["vm"]],fd["Xtr_i"][fd["vm"]],fd["ytr"][fd["vm"]],BATCH_SIZE,False)
            te_ld=make_loader(fd["dis_te"],fd["Xte_i"],fd["ytel"],BATCH_SIZE,False)
            fulltr_ld=make_loader(fd["dis_tr"],fd["Xtr_i"],fd["ytr"],BATCH_SIZE,False)

            mp=TMP_DIR/f"_sannwr_cv_{strat}_fold{fid}_seed{seed}.pt"
            model=SANNWRModel(N_GRID,fd["n_feat"],SAPDNN_HIDDEN,DROP_OUT,BATCH_NORM,fd["oc"]).to(DEVICE)
            train_model(model,tr_ld,val_ld,mp)
            pred_tr=predict(model,fulltr_ld)
            s_M=smearing_factor(fd["ytr"],pred_tr)
            pred=predict(model,te_ld)
            m=metrics_log(fd["yteo"],pred,fd["ytel"],s_M=s_M)
            mp.unlink(missing_ok=True)
            if DEVICE.type=="cuda": torch.cuda.empty_cache()

            dt=time.time()-t0
            print(f"  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  ({dt:.0f}s)",flush=True)
            records_fold.append({"seed":seed,"estrategia":strat,"fold":fid,**m})
            fold_metrics.append(m)

        rmse_vals=[x["RMSE"] for x in fold_metrics]
        r2_vals=[x["R2"] for x in fold_metrics]
        row={"seed":seed,"estrategia":strat,
             "RMSE_mean":round(np.mean(rmse_vals),4),"RMSE_std":round(np.std(rmse_vals,ddof=1),4),
             "R2_mean":round(np.mean(r2_vals),4),"R2_std":round(np.std(r2_vals,ddof=1),4)}
        records_seed.append(row)
        print(f"\n  [{strat}] seed={seed}  RMSE={row['RMSE_mean']:.2f}+-{row['RMSE_std']:.2f}  R2={row['R2_mean']:.4f}",flush=True)

# ── Guardar ────────────────────────────────────────────────────────────────────
df_fold=pd.DataFrame(records_fold)
df_seed=pd.DataFrame(records_seed)
df_fold.to_csv(OUT_DIR/"sannwr_log_cv_replicas.csv",index=False)
df_seed.to_csv(OUT_DIR/"sannwr_log_cv_replicas_seed.csv",index=False)

meta=df_seed.groupby("estrategia").agg(
    n_seeds=("seed","count"),
    RMSE_mean=("RMSE_mean","mean"),RMSE_std=("RMSE_mean","std"),
    R2_mean=("R2_mean","mean"),R2_std=("R2_mean","std")).reset_index().round(4)
meta.to_csv(OUT_DIR/"sannwr_log_cv_replicas_summary.csv",index=False)

print(f"\n{'='*65}\nRESUMEN FINAL SANNWR CV REPLICAS\n{'='*65}")
print(meta.to_string(index=False))
print(f"\n[CSV] {OUT_DIR}/sannwr_log_cv_replicas_summary.csv")
