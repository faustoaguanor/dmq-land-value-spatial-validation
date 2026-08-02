"""
gsawr_log_cv_replicas.py
========================
Replicas de CV (RandomKFold + SpatialBlock) para GSAWR con 5 seeds.
Para cada seed se corre CV completo (5 folds x 2 estrategias).
Tensores imagen pre-calculados por fold (fuera del loop de seeds) para eficiencia.

Salidas: modelos/gsawr/output_log/
  gsawr_log_cv_replicas.csv          -- por seed x estrategia x fold
  gsawr_log_cv_replicas_seed.csv     -- RMSE_mean+-std por seed x estrategia
  gsawr_log_cv_replicas_summary.csv  -- RMSE_mean+-std por estrategia sobre seeds
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

# ── Config IDENTICA a gsawr_log.py ────────────────────────────────────────────
COVARIABLES = [
    "suscept_codigo","pc_pnbi","dist_metro","dist_centr_metro",
    "dist_centr_zonal","dist_cc","dist_universidad","dist_hospital",
    "dist_parque_metro","dist_industrial","dist_via_principal",
    "uso_suelo_cod","cos_num","dist_quebrada","dist_mercado_mayorista",
    "dist_plataforma_gub","log_area","frente_m","area_const_m2",
    "tiene_const","num_pisos","antiguedad","topografia_factor",
    "conservacion_cod","acabados_cod","es_ph","pendiente_grados",
]
GRID_COLS=20; GRID_ROWS=20; IDW_POWER=2.0
AFCNN_HIDDEN=[16]; SAJPNN_HIDDEN=4
CNN_CHANNELS=[16,16,32]; CNN_KERNEL=(5,5); DENSE_LAYERS=[256,64]
N_EPOCHS=300; PATIENCE=60; BATCH_SIZE=32; START_LR=0.001
RANDOM_STATE=42; SEEDS=[42,2011,456,777,2026]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH  = ROOT/"datos"/"dataset.gpkg"
SPLIT_PATH = ROOT/"data_split"/"split.csv"
FOLDS_PATH = ROOT/"spatial_cv"/"output"/"fold_assignments.csv"
OUT_DIR    = ROOT/"modelos"/"gsawr"/"output_log"
TMP_DIR    = OUT_DIR/"models"; OUT_DIR.mkdir(exist_ok=True); TMP_DIR.mkdir(exist_ok=True)

# ── Arquitectura IDENTICA a gsawr_log.py ──────────────────────────────────────
class AFCNN(nn.Module):
    def __init__(self,n_feat,hidden,bias=True,batch_norm=True):
        super().__init__()
        layers=nn.Sequential(); in_ch=n_feat
        for i,out_ch in enumerate(hidden):
            layers.add_module(f"conv{i+1}",nn.Conv2d(in_ch,out_ch,kernel_size=(1,1),bias=bias))
            if batch_norm: layers.add_module(f"bn{i+1}",nn.BatchNorm2d(out_ch))
            layers.add_module(f"act{i+1}",nn.ReLU()); in_ch=out_ch
        layers.add_module("conv_out",nn.Conv2d(in_ch,1,kernel_size=(1,1),bias=bias))
        layers.add_module("act_out",nn.ReLU()); self.afcnn=layers
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self,a): return self.afcnn(a)

class SAJPNN(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,hidden,bias=True),nn.PReLU(init=0.1),
                               nn.Linear(hidden,1,bias=True),nn.PReLU(init=0.1))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self,x): return self.net(x)

class SANNet(nn.Module):
    def __init__(self,n_feat,afcnn_hidden,sajpnn_hidden,bias=True,batch_norm=True):
        super().__init__()
        self.afcnn=AFCNN(n_feat,afcnn_hidden,bias,batch_norm)
        self.sajpnn=SAJPNN(sajpnn_hidden)
    def forward(self,dis):
        b,c,h,w=dis.size(); s_mat=dis[:,0:1,:,:]; a_mat=dis[:,1:,:,:]
        a_fused=self.afcnn(a_mat); sa=torch.cat([s_mat,a_fused],dim=1)
        out=self.sajpnn(sa.permute(0,2,3,1).reshape(-1,2))
        return out.view(b,1,h,w)

class SAWCNN(nn.Module):
    def __init__(self,rows,cols,n_features,cnn_channels,kernel_size,dense_layers):
        super().__init__()
        conv_blocks=nn.Sequential(); in_ch=1
        pad=(kernel_size[0]//2,kernel_size[1]//2)
        for i,out_ch in enumerate(cnn_channels):
            block=nn.Sequential()
            block.add_module("conv",nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,padding=pad,bias=True))
            block.add_module("act",nn.PReLU(init=0.1))
            block.add_module("pool",nn.MaxPool2d(kernel_size=2))
            conv_blocks.add_module(f"block{i+1}",block); in_ch=out_ch
        self.conv_blocks=conv_blocks
        h,w=rows,cols
        for _ in cnn_channels: h//=2; w//=2
        last_size=cnn_channels[-1]*h*w
        fc=nn.Sequential(); prev=last_size
        for i,hh in enumerate(dense_layers):
            fc.add_module(f"fc{i}",nn.Linear(prev,hh)); fc.add_module(f"act{i}",nn.ReLU()); prev=hh
        fc.add_module("fc_out",nn.Linear(prev,n_features)); self.fc=fc
        for m in self.modules():
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self,x):
        b=x.size(0)
        for block in self.conv_blocks: x=block(x)
        return self.fc(x.view(b,-1))

class GSAWRModel(nn.Module):
    def __init__(self,n_feat,n_features,rows,cols,afcnn_hidden,sajpnn_hidden,cnn_channels,kernel_size,dense_layers,ols_coeff):
        super().__init__()
        self.sannet=SANNet(n_feat,afcnn_hidden,sajpnn_hidden)
        self.sawcnn=SAWCNN(rows,cols,n_features,cnn_channels,kernel_size,dense_layers)
        self.out=nn.Linear(n_features,1,bias=False)
        self.out.weight=nn.Parameter(torch.tensor(ols_coeff.reshape(1,-1),dtype=torch.float32),requires_grad=False)
    def forward(self,dis,x): return self.out(self.sawcnn(self.sannet(dis))*x)

def add_intercept(X): return np.column_stack([np.ones(len(X)),X])
def compute_ols(Xi,y): return LinearRegression(fit_intercept=False).fit(Xi,y).coef_.flatten().astype(np.float32)
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
    el=y_log-y_pred_log; r2l=round(1-np.sum(el**2)/np.sum((y_log-y_log.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l,"smearing_factor":round(s_M,6)}

def build_grid(ctr,cols,rows):
    xmn,ymn=ctr.min(0); xmx,ymx=ctr.max(0)
    dx=(xmx-xmn)*0.05; dy=(ymx-ymn)*0.05
    xl=np.linspace(xmn-dx,xmx+dx,cols); yl=np.linspace(ymn-dy,ymx+dy,rows)
    gx,gy=np.meshgrid(xl,yl); return np.column_stack([gx.ravel(),gy.ravel()])

def compute_idw(ctr,Xtr,gxy,p=IDW_POWER):
    d=cdist(gxy,ctr); w=1/(d**p+1e-10); w/=w.sum(1,keepdims=True); return w@Xtr

def build_image(cq,Xq,gxy,gattrs,rows,cols):
    n=len(cq); nf=Xq.shape[1]
    ds=cdist(cq,gxy).reshape(n,1,rows,cols)
    delta=np.abs(Xq[:,np.newaxis,:]-gattrs[np.newaxis,:,:]).transpose(0,2,1).reshape(n,nf,rows,cols)
    return np.concatenate([ds,delta],axis=1)

def make_loader(dis,X,y,bs,shuf):
    ds=TensorDataset(torch.tensor(dis,dtype=torch.float32),torch.tensor(X,dtype=torch.float32),
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
        for dis,x,_ in loader: ps.append(model(dis.to(DEVICE),x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(ps)

# ── Cargar datos ───────────────────────────────────────────────────────────────
print(f"ROOT={ROOT}  device={DEVICE}",flush=True)
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
print(f"GSAWR CV replicas  device={DEVICE}  train={len(train_idx)}  seeds={SEEDS}",flush=True)

cv_rand=RandomKFoldCV(n_splits=5,random_state=RANDOM_STATE)
cv_block=SpatialBlockCV(folds=sp_folds[train_mask])
splits_r=list(cv_rand.split(X_raw[train_mask]))
splits_b=list(cv_block.split(X_raw[train_mask]))

# ── Pre-calcular tensores imagen por fold (fuera del loop de seeds) ────────────
print("\nPre-calculando tensores imagen por fold ...",flush=True)
fold_cache={}
for strat,splits in [("RandomKFold",splits_r),("SpatialBlock",splits_b)]:
    fold_cache[strat]=[]
    for fid,(tr,te) in enumerate(splits):
        t0=time.time()
        sc=StandardScaler()
        Xtr=sc.fit_transform(X_raw[train_mask][tr])
        Xte=sc.transform(X_raw[train_mask][te])
        ytr=y_log[train_mask][tr].astype(np.float32)
        ytel=y_log[train_mask][te].astype(np.float32)
        yteo=y_orig[train_mask][te]
        ctr=coords[train_mask][tr]; cte=coords[train_mask][te]
        Xtr_i=add_intercept(Xtr).astype(np.float32)
        Xte_i=add_intercept(Xte).astype(np.float32)
        ols_c=compute_ols(Xtr_i,ytr); n_feat=Xtr_i.shape[1]; n_img_ch=Xtr.shape[1]
        gxy=build_grid(ctr,GRID_COLS,GRID_ROWS)
        gattrs=compute_idw(ctr,Xtr,gxy)
        raw_tr=build_image(ctr,Xtr,gxy,gattrs,GRID_ROWS,GRID_COLS)
        raw_te=build_image(cte,Xte,gxy,gattrs,GRID_ROWS,GRID_COLS)
        # Per-channel scalers (fit en train)
        ch_sc=[]
        for ch in range(raw_tr.shape[1]):
            s=StandardScaler(); s.fit(raw_tr[:,ch,:,:].reshape(-1,1)); ch_sc.append(s)
        def scale_img(raw):
            out=np.empty_like(raw,dtype=np.float32)
            for ch in range(raw.shape[1]):
                out[:,ch,:,:]=ch_sc[ch].transform(raw[:,ch,:,:].reshape(-1,1)).reshape(len(raw),GRID_ROWS,GRID_COLS)
            return out
        dis_tr=scale_img(raw_tr); dis_te=scale_img(raw_te)
        rng=np.random.default_rng(RANDOM_STATE+fid); vm=rng.random(len(tr))<0.10; tm=~vm
        fold_cache[strat].append({
            "fid":fid,"n_tr":len(tr),"n_feat":n_feat,"n_img_ch":n_img_ch,
            "Xtr_i":Xtr_i,"Xte_i":Xte_i,"ytr":ytr,"ytel":ytel,"yteo":yteo,
            "dis_tr":dis_tr,"dis_te":dis_te,"ols_c":ols_c,"vm":vm,"tm":tm,
        })
        mb=(dis_tr.nbytes+dis_te.nbytes)/(1024**2)
        print(f"  {strat} fold{fid+1}: {len(tr)}tr/{len(te)}te  tensores={mb:.0f}MB  ({time.time()-t0:.0f}s)",flush=True)

# ── Loop seeds ─────────────────────────────────────────────────────────────────
records_fold=[]; records_seed=[]

for seed in SEEDS:
    print(f"\n{'='*65}\n[GSAWR CV Replica seed={seed}]\n{'='*65}",flush=True)
    for strat in ["RandomKFold","SpatialBlock"]:
        fold_metrics=[]
        for fd in fold_cache[strat]:
            fid=fd["fid"]
            print(f"\n  [{strat}] fold{fid+1}  seed={seed}  n_tr={fd['n_tr']}",flush=True)
            t0=time.time()
            torch.manual_seed(seed); import random as _r; _r.seed(seed); np.random.seed(seed)
            if DEVICE.type=="cuda": torch.cuda.manual_seed(seed)
            tr_ld=make_loader(fd["dis_tr"][fd["tm"]],fd["Xtr_i"][fd["tm"]],fd["ytr"][fd["tm"]],BATCH_SIZE,True)
            val_ld=make_loader(fd["dis_tr"][fd["vm"]],fd["Xtr_i"][fd["vm"]],fd["ytr"][fd["vm"]],BATCH_SIZE,False)
            te_ld=make_loader(fd["dis_te"],fd["Xte_i"],fd["ytel"],BATCH_SIZE,False)
            fulltr_ld=make_loader(fd["dis_tr"],fd["Xtr_i"],fd["ytr"],BATCH_SIZE,False)
            mp=TMP_DIR/f"_gsawr_cv_{strat}_fold{fid}_seed{seed}.pt"
            model=GSAWRModel(fd["n_img_ch"],fd["n_feat"],GRID_ROWS,GRID_COLS,
                             AFCNN_HIDDEN,SAJPNN_HIDDEN,CNN_CHANNELS,CNN_KERNEL,DENSE_LAYERS,fd["ols_c"]).to(DEVICE)
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
            # Guardar incrementalmente por si se interrumpe
            pd.DataFrame(records_fold).to_csv(OUT_DIR/"gsawr_log_cv_replicas.csv",index=False)

        rmse_vals=[x["RMSE"] for x in fold_metrics]
        r2_vals=[x["R2"] for x in fold_metrics]
        row={"seed":seed,"estrategia":strat,
             "RMSE_mean":round(np.mean(rmse_vals),4),"RMSE_std":round(np.std(rmse_vals,ddof=1),4),
             "R2_mean":round(np.mean(r2_vals),4),"R2_std":round(np.std(r2_vals,ddof=1),4)}
        records_seed.append(row)
        print(f"\n  [{strat}] seed={seed}  RMSE={row['RMSE_mean']:.2f}+-{row['RMSE_std']:.2f}",flush=True)

df_fold=pd.DataFrame(records_fold); df_seed=pd.DataFrame(records_seed)
df_fold.to_csv(OUT_DIR/"gsawr_log_cv_replicas.csv",index=False)
df_seed.to_csv(OUT_DIR/"gsawr_log_cv_replicas_seed.csv",index=False)
meta=df_seed.groupby("estrategia").agg(
    n_seeds=("seed","count"),
    RMSE_mean=("RMSE_mean","mean"),RMSE_std=("RMSE_mean","std"),
    R2_mean=("R2_mean","mean"),R2_std=("R2_mean","std")).reset_index().round(4)
meta.to_csv(OUT_DIR/"gsawr_log_cv_replicas_summary.csv",index=False)
print(f"\n{'='*65}\nRESUMEN FINAL GSAWR CV REPLICAS\n{'='*65}")
print(meta.to_string(index=False))
