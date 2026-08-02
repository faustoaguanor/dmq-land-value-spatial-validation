"""
interpretabilidad_mlp.py
=========================
Permutation importance para MLP (baseline no-espacial).

El MLP no tiene componente espacial: solo X_test cambia al permutar.
Sin necesidad de reconstruir tensores ni matrices de distancia.

Metodo:
  1. Re-entrena MLP seed=42 sobre el train completo (identico a mlp_log.py)
  2. RMSE_base sobre holdout 20%
  3. Para cada variable k: baraja, mide degradacion RMSE
  4. 5 repeticiones

Salidas: analisis/output_log/
  interp_mlp_permutation_importance.csv
  interp_mlp_perm_importance.png/.pdf
"""
from __future__ import annotations
import sys, time, random as _random
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if not (ROOT / "datos" / "dataset.gpkg").exists():
    ROOT = Path("/workspace/pod")

sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix

OUT_DIR = ROOT / "analisis" / "output_log"
OUT_DIR.mkdir(exist_ok=True)
HIDDEN_LAYERS=[256,128,64]; N_EPOCHS=300; PATIENCE=40; BATCH_SIZE=64
LR=0.001; WD=1e-4; CLIP_GRAD=5.0; VAL_FRAC=0.10
RANDOM_STATE=42; N_REPEATS=5
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPRegressor(nn.Module):
    def __init__(self,n_feat,hidden):
        super().__init__()
        layers=[]; in_dim=n_feat
        for h in hidden:
            layers+=[nn.Linear(in_dim,h),nn.BatchNorm1d(h),nn.ReLU()]; in_dim=h
        layers.append(nn.Linear(in_dim,1))
        self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
                nn.init.constant_(m.bias,0.0)
    def forward(self,x): return self.net(x).squeeze(-1)

def train_mlp(Xtr,ytr,Xval,yval,n_feat):
    model=MLPRegressor(n_feat,HIDDEN_LAYERS).to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    loss_fn=nn.HuberLoss(delta=1.0)
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.float32)
    Xv=torch.tensor(Xval,dtype=torch.float32).to(DEVICE); yv=torch.tensor(yval,dtype=torch.float32).to(DEVICE)
    loader=DataLoader(TensorDataset(Xt,yt),batch_size=BATCH_SIZE,shuffle=True)
    best_val=float("inf"); best_state=None; no_improve=0
    for ep in range(N_EPOCHS):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            loss=loss_fn(model(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),CLIP_GRAD); opt.step()
        model.eval()
        with torch.no_grad(): val_loss=loss_fn(model(Xv),yv).item()
        if val_loss<best_val-1e-6:
            best_val=val_loss; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_improve=0
        else:
            no_improve+=1
            if no_improve>=PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    return model

def predict_mlp(model,X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X,dtype=torch.float32).to(DEVICE)).cpu().numpy()

def rmse_orig(y_true,pred_log): return float(np.sqrt(np.mean((y_true-np.exp(pred_log))**2)))

# ── Cargar datos ───────────────────────────────────────────────────────────────
print(f"ROOT={ROOT}  device={DEVICE}",flush=True)
gdf=gpd.read_file(ROOT/"datos"/"dataset.gpkg",layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"]=gdf["predio_join"].astype(int)
split_df=pd.read_csv(ROOT/"data_split"/"split.csv"); split_df["predio_join"]=split_df["predio_join"].astype(int)
gdf=gdf.merge(split_df[["predio_join","split"]],on="predio_join",how="left")
gdf=gdf.sort_values("predio_join").reset_index(drop=True)
y_orig=gdf["valor_m2"].values.astype(float); y_log=np.log(y_orig)
X_raw, FEAT_NAMES=build_feature_matrix(gdf)
train_mask=(gdf["split"]=="train").values; test_mask=(gdf["split"]=="test").values
train_idx=np.where(train_mask)[0]; test_idx=np.where(test_mask)[0]
n_feat=X_raw.shape[1]
print(f"n_train={len(train_idx)}  n_test={len(test_idx)}",flush=True)

scaler=StandardScaler()
Xtr_sc=scaler.fit_transform(X_raw[train_mask]).astype(np.float32)
Xte_sc=scaler.transform(X_raw[test_mask]).astype(np.float32)
ytr_log=y_log[train_mask].astype(np.float32)
rng_v=np.random.default_rng(RANDOM_STATE); vm=rng_v.random(len(train_idx))<VAL_FRAC; tm=~vm
y_mean=float(ytr_log[tm].mean()); y_std=float(ytr_log[tm].std())+1e-8
ytr_norm=((ytr_log[tm]-y_mean)/y_std); yval_norm=((ytr_log[vm]-y_mean)/y_std)

print("Entrenando MLP seed=42 ...",flush=True)
_random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
if DEVICE.type=="cuda": torch.cuda.manual_seed(RANDOM_STATE)
t0=time.time()
model=train_mlp(Xtr_sc[tm],ytr_norm,Xtr_sc[vm],yval_norm,n_feat)
print(f"Entrenado en {time.time()-t0:.0f}s",flush=True)
model.eval()

pred_base=predict_mlp(model,Xte_sc)*y_std+y_mean
rmse_base=rmse_orig(y_orig[test_mask],pred_base)
print(f"RMSE base: {rmse_base:.4f}",flush=True)

print(f"\nPermutation Importance ({N_REPEATS} reps x {len(FEAT_NAMES)} vars) ...",flush=True)
rng2=np.random.default_rng(RANDOM_STATE+1)
rows=[]
for k,var in enumerate(FEAT_NAMES):
    deltas=[]
    for _ in range(N_REPEATS):
        Xp=X_raw[test_mask].copy()
        Xp[:,k]=rng2.permutation(Xp[:,k])
        Xp_sc=scaler.transform(Xp).astype(np.float32)
        pred_p=predict_mlp(model,Xp_sc)*y_std+y_mean
        deltas.append(rmse_orig(y_orig[test_mask],pred_p)-rmse_base)
    md=float(np.mean(deltas)); sd=float(np.std(deltas))
    rows.append({"variable":var,"importance_mean":round(md,4),"importance_std":round(sd,4),
                 "rmse_perm_mean":round(rmse_base+md,4)})
    print(f"  [{k+1:2d}/{len(FEAT_NAMES)}] {var:30s}  dRMSE={md:+.3f} +- {sd:.3f}",flush=True)

imp_df=pd.DataFrame(rows).sort_values("importance_mean",ascending=False)
imp_df.to_csv(OUT_DIR/"interp_mlp_permutation_importance.csv",index=False)
print("\nTop-10:")
print(imp_df.head(10)[["variable","importance_mean","importance_std"]].to_string(index=False))

n_show=min(20,len(imp_df)); top=imp_df.head(n_show).iloc[::-1]
fig,ax=plt.subplots(figsize=(10,7))
colors=["#E53935" if v>0 else "#43A047" for v in top["importance_mean"]]
ax.barh(range(n_show),top["importance_mean"],xerr=top["importance_std"],capsize=3,
        color=colors,alpha=0.82,edgecolor="black",lw=0.5,error_kw={"elinewidth":1.0})
ax.axvline(0,color="black",lw=1.0)
ax.set_yticks(range(n_show)); ax.set_yticklabels(top["variable"],fontsize=9)
ax.set_xlabel("dRMSE (USD/m2) al permutar la variable",fontsize=11)
ax.set_title(f"Permutation Importance - MLP (baseline no-espacial)\n"
             f"Holdout 20% (n={len(test_idx)})  |  5 reps  |  RMSE base={rmse_base:.1f} USD/m2",
             fontsize=11,fontweight="bold")
ax.grid(axis="x",alpha=0.3,linestyle="--"); ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(OUT_DIR/"interp_mlp_perm_importance.png",dpi=180,bbox_inches="tight")
fig.savefig(OUT_DIR/"interp_mlp_perm_importance.pdf",bbox_inches="tight")
plt.close(fig)
print(f"[OK] {OUT_DIR}/interp_mlp_perm_importance.png/pdf")
print(f"[OK] {OUT_DIR}/interp_mlp_permutation_importance.csv")
