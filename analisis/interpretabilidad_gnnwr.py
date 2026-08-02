"""
interpretabilidad_gnnwr.py
==========================
Permutation importance para GNNWR (Du et al. 2020).

La matriz de distancia espacial (dis) NO cambia al permutar atributos —
es geometrica pura. Solo cambia X_int[:,k+1] (el vector de features).
Esto hace la permutacion mas rapida que GSAWR/SANNWR.

Metodo:
  1. Re-entrena GNNWR seed=42 sobre el train completo (identico a gnnwr_log.py)
  2. RMSE_base sobre holdout 20%
  3. Para cada variable k: baraja columna k en X_test, mide degradacion RMSE
  4. 5 repeticiones, reporta mean+-std

Salidas: analisis/output_log/
  interp_gnnwr_permutation_importance.csv
  interp_gnnwr_perm_importance.png/.pdf
"""
from __future__ import annotations
import math, sys, time
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

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if not (ROOT / "datos" / "dataset.gpkg").exists():
    ROOT = Path("/workspace/pod")

sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix

OUT_DIR = ROOT / "analisis" / "output_log"
OUT_DIR.mkdir(exist_ok=True)
TMP_DIR = ROOT / "modelos" / "gnnwr" / "output_log" / "models"
TMP_DIR.mkdir(exist_ok=True)
DENSE_LAYERS=[2048,1024,512,256,64]; DROP_OUT=0.2; BATCH_NORM=True
N_EPOCHS=1000; PATIENCE=200; BATCH_SIZE=64; START_LR=0.2
RANDOM_STATE=42; N_REPEATS=5
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Arquitectura ───────────────────────────────────────────────────────────────
class SWNN(nn.Module):
    def __init__(self,insize,outsize,dense_layers,drop_out,batch_norm):
        super().__init__()
        act=nn.PReLU(init=0.1); layers=[]; last=insize
        for h in dense_layers:
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

class GNNWRModel(nn.Module):
    def __init__(self,n_train,n_features,dense_layers,drop_out,batch_norm,ols_coeff):
        super().__init__()
        self.swnn=SWNN(n_train,n_features,dense_layers,drop_out,batch_norm)
        self.out=nn.Linear(n_features,1,bias=False)
        self.out.weight=nn.Parameter(
            torch.tensor(ols_coeff.reshape(1,-1),dtype=torch.float32),requires_grad=False)
    def forward(self,dis,x): return self.out(self.swnn(dis)*x)

def add_intercept(X): return np.column_stack([np.ones(len(X)),X])

def make_loader(dis,X,y,bs,shuf):
    ds=TensorDataset(torch.tensor(dis,dtype=torch.float32),
                     torch.tensor(X,dtype=torch.float32),
                     torch.tensor(y.reshape(-1,1),dtype=torch.float32))
    return DataLoader(ds,batch_size=bs,shuffle=shuf,drop_last=False)

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
    opt=torch.optim.Adadelta(model.parameters(),lr=START_LR,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=100,T_mult=3,eta_min=0.01)
    best,pat=float("inf"),0
    for ep in range(1,N_EPOCHS+1):
        tl=one_epoch(model,tr_ld,opt); vl=one_epoch(model,val_ld,None); sch.step()
        if ep%100==0 or ep==1: print(f"    ep{ep:4d} train={tl:.4f} val={vl:.4f}",flush=True)
        if vl<best: best=vl; pat=0; torch.save(model.state_dict(),path)
        else:
            pat+=1
            if pat>=PATIENCE: print(f"    Early stop ep{ep}",flush=True); break
    model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))

def predict_gnnwr(model,dis_sc,X_int,bs=BATCH_SIZE):
    model.eval(); preds=[]
    dis_t=torch.tensor(dis_sc,dtype=torch.float32)
    x_t=torch.tensor(X_int,dtype=torch.float32)
    with torch.no_grad():
        for i in range(0,len(X_int),bs):
            preds.append(model(dis_t[i:i+bs].to(DEVICE),x_t[i:i+bs].to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(preds)

def rmse_orig(y_true,pred_log): return float(np.sqrt(np.mean((y_true-np.exp(pred_log))**2)))

# ── Cargar datos ───────────────────────────────────────────────────────────────
print(f"ROOT={ROOT}  device={DEVICE}",flush=True)
gdf=gpd.read_file(ROOT/"datos"/"dataset.gpkg",layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"]=gdf["predio_join"].astype(int)
split_df=pd.read_csv(ROOT/"data_split"/"split.csv"); split_df["predio_join"]=split_df["predio_join"].astype(int)
gdf=gdf.merge(split_df[["predio_join","split"]],on="predio_join",how="left")
gdf=gdf.sort_values("predio_join").reset_index(drop=True)
coords=np.column_stack([gdf.geometry.x,gdf.geometry.y])
y_orig=gdf["valor_m2"].values.astype(float); y_log=np.log(y_orig)
X_raw, FEAT_NAMES=build_feature_matrix(gdf)
train_mask=(gdf["split"]=="train").values; test_mask=(gdf["split"]=="test").values
train_idx=np.where(train_mask)[0]; test_idx=np.where(test_mask)[0]
print(f"n_train={len(train_idx)}  n_test={len(test_idx)}",flush=True)

# ── Preprocesar ────────────────────────────────────────────────────────────────
scaler=StandardScaler()
Xtr_sc=scaler.fit_transform(X_raw[train_mask]); Xte_sc=scaler.transform(X_raw[test_mask])
Xtr_int=add_intercept(Xtr_sc).astype(np.float32)
Xte_int=add_intercept(Xte_sc).astype(np.float32)
n_feat=Xtr_int.shape[1]
ctr=coords[train_mask]; cte=coords[test_mask]

print("Calculando distancias ...",flush=True)
dis_tr=cdist(ctr,ctr); dis_te=cdist(cte,ctr)
ds=StandardScaler()
dis_tr_sc=ds.fit_transform(dis_tr).astype(np.float32)
dis_te_sc=ds.transform(dis_te).astype(np.float32)

ols_c=LinearRegression(fit_intercept=False).fit(Xtr_int,y_log[train_mask]).coef_.flatten().astype(np.float32)

# ── Cargar modelo final (mismo que genero las metricas holdout reportadas) ─────
MODEL_PATH = ROOT/"modelos"/"gnnwr"/"output_log"/"models"/"gnnwr_log_final.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontro {MODEL_PATH}. Corre gnnwr_log.py primero.")
model=GNNWRModel(len(train_idx),n_feat,DENSE_LAYERS,DROP_OUT,BATCH_NORM,ols_c).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE,weights_only=True))
print(f"Modelo cargado: {MODEL_PATH.name}  (mismo que genero RMSE holdout reportado)",flush=True)
model.eval()

# ── Prediccion base ────────────────────────────────────────────────────────────
pred_base=predict_gnnwr(model,dis_te_sc,Xte_int)
rmse_base=rmse_orig(y_orig[test_mask],pred_base)
print(f"RMSE base: {rmse_base:.4f}",flush=True)

# ── Permutation importance ────────────────────────────────────────────────────
print(f"\nPermutation Importance ({N_REPEATS} reps x {len(FEAT_NAMES)} vars) ...",flush=True)
rng2=np.random.default_rng(RANDOM_STATE+1)
rows=[]
for k,var in enumerate(FEAT_NAMES):
    deltas=[]
    for _ in range(N_REPEATS):
        Xp=X_raw[test_mask].copy()
        Xp[:,k]=rng2.permutation(Xp[:,k])
        Xp_sc=scaler.transform(Xp).astype(np.float32)
        Xp_int=add_intercept(Xp_sc).astype(np.float32)
        # dis NO cambia (distancia geometrica pura)
        pred_p=predict_gnnwr(model,dis_te_sc,Xp_int)
        deltas.append(rmse_orig(y_orig[test_mask],pred_p)-rmse_base)
    md=float(np.mean(deltas)); sd=float(np.std(deltas))
    rows.append({"variable":var,"importance_mean":round(md,4),"importance_std":round(sd,4),
                 "rmse_perm_mean":round(rmse_base+md,4)})
    print(f"  [{k+1:2d}/{len(FEAT_NAMES)}] {var:30s}  dRMSE={md:+.3f} +- {sd:.3f}",flush=True)

imp_df=pd.DataFrame(rows).sort_values("importance_mean",ascending=False)
imp_df.to_csv(OUT_DIR/"interp_gnnwr_permutation_importance.csv",index=False)
print("\nTop-10:")
print(imp_df.head(10)[["variable","importance_mean","importance_std"]].to_string(index=False))

# ── Figura ──────────────────────────────────────────────────────────────────────
n_show=min(20,len(imp_df)); top=imp_df.head(n_show).iloc[::-1]
fig,ax=plt.subplots(figsize=(10,7))
colors=["#E53935" if v>0 else "#43A047" for v in top["importance_mean"]]
ax.barh(range(n_show),top["importance_mean"],xerr=top["importance_std"],capsize=3,
        color=colors,alpha=0.82,edgecolor="black",lw=0.5,error_kw={"elinewidth":1.0})
ax.axvline(0,color="black",lw=1.0)
ax.set_yticks(range(n_show)); ax.set_yticklabels(top["variable"],fontsize=9)
ax.set_xlabel("dRMSE (USD/m2) al permutar la variable",fontsize=11)
ax.set_title(f"Permutation Importance - GNNWR (Du et al. 2020)\n"
             f"Holdout 20% (n={len(test_idx)})  |  5 reps  |  RMSE base={rmse_base:.1f} USD/m2",
             fontsize=11,fontweight="bold")
ax.grid(axis="x",alpha=0.3,linestyle="--"); ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(OUT_DIR/"interp_gnnwr_perm_importance.png",dpi=180,bbox_inches="tight")
fig.savefig(OUT_DIR/"interp_gnnwr_perm_importance.pdf",bbox_inches="tight")
plt.close(fig)
print(f"[OK] {OUT_DIR}/interp_gnnwr_perm_importance.png/pdf")
print(f"[OK] {OUT_DIR}/interp_gnnwr_permutation_importance.csv")
