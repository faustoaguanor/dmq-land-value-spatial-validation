"""
mlp_log_replicas.py — Réplicas MLP con seeds [42, 2011, 456, 777, 2026] — solo holdout 20%.
================================================================================
Estabilidad del MLP ante inicialización aleatoria, misma config que mlp_log.py.
Re-siembra antes de entrenar el modelo final (holdout controlado).
"""
from __future__ import annotations
import sys, time, random as _random
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from libpysal.weights import KNN as KNNWeights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

COVARIABLES = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]
HIDDEN_LAYERS=[256,128,64]; N_EPOCHS=300; PATIENCE=40; BATCH_SIZE=64
LR=0.001; WD=1e-4; VAL_FRAC=0.10; MORAN_K=8; SEEDS=[42,2011,456,777,2026,99,1234,888,314,7]  # 10 seeds one-hot (uniforme con SANNWR/GSAWR)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR=Path(__file__).parent; ROOT=BASE_DIR.parent.parent
DATA_PATH=ROOT/"datos"/"dataset.gpkg"; SPLIT_PATH=ROOT/"data_split"/"split.csv"
OUT_DIR=BASE_DIR/"output_log"; OUT_DIR.mkdir(exist_ok=True)

def pr(*a,**k): print(*a,**k,flush=True)

class MLPRegressor(nn.Module):
    def __init__(self, n_feat, hidden):
        super().__init__()
        layers=[]; in_dim=n_feat
        for h in hidden:
            layers += [nn.Linear(in_dim,h), nn.BatchNorm1d(h), nn.ReLU()]; in_dim=h
        layers.append(nn.Linear(in_dim,1))
        self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in"); nn.init.constant_(m.bias,0.0)
    def forward(self,x): return self.net(x).squeeze(-1)

def train_mlp(X_tr,y_tr,X_val,y_val,n_feat):
    model=MLPRegressor(n_feat,HIDDEN_LAYERS).to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=LR,weight_decay=WD); loss_fn=nn.HuberLoss(delta=1.0)
    Xt=torch.tensor(X_tr,dtype=torch.float32); yt=torch.tensor(y_tr,dtype=torch.float32)
    Xv=torch.tensor(X_val,dtype=torch.float32).to(DEVICE); yv=torch.tensor(y_val,dtype=torch.float32).to(DEVICE)
    loader=DataLoader(TensorDataset(Xt,yt),batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
    best,bs,no_imp=float("inf"),None,0
    for ep in range(N_EPOCHS):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad(): vl=loss_fn(model(Xv),yv).item()
        if vl<best: best=vl; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp+=1
            if no_imp>=PATIENCE: break
    if bs is not None: model.load_state_dict(bs)
    return model

def predict_mlp(model,X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X,dtype=torch.float32).to(DEVICE)).cpu().numpy()

def metrics(y_to,y_po,y_tl,y_pl):
    e=y_to-y_po; mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2)))
    mape=float(np.mean(np.abs(e/y_to))*100)
    r2=round(1-np.sum(e**2)/np.sum((y_to-y_to.mean())**2),4)
    el=y_tl-y_pl; r2l=round(1-np.sum(el**2)/np.sum((y_tl-y_tl.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l}

def moran(v,c,k=MORAN_K):
    w=KNNWeights.from_array(c,k=k); w.transform="r"; z=v-v.mean(); return float((z@(w.sparse@z))/(z@z))

# Datos
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
n_feat=X_raw.shape[1]

scaler=StandardScaler()
X_tr_f=scaler.fit_transform(X_raw[train_mask]); X_te_f=scaler.transform(X_raw[test_mask])
y_tr_log_f=y_log[train_mask].astype(np.float32)

records=[]
for seed in SEEDS:
    pr(f"\n[MLP Replica seed={seed}]")
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type=="cuda": torch.cuda.manual_seed_all(seed)
    vm=np.random.default_rng(seed).random(len(train_idx))<VAL_FRAC; tm=~vm
    ym=float(y_tr_log_f[tm].mean()); ys=float(y_tr_log_f[tm].std())+1e-8
    ytr=((y_tr_log_f[tm]-ym)/ys).astype(np.float32); yval=((y_tr_log_f[vm]-ym)/ys).astype(np.float32)
    t0=time.time()
    model=train_mlp(X_tr_f[tm],ytr,X_tr_f[vm],yval,n_feat)
    pred_log=predict_mlp(model,X_te_f)*ys+ym; pred_ori=np.exp(pred_log)
    m=metrics(y_orig[test_mask],pred_ori,y_log[test_mask],pred_log)
    mo=moran(y_log[test_mask]-pred_log,coords[test_mask]) if LIBPYSAL_AVAILABLE else float("nan")
    pr(f"  seed={seed}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  ({time.time()-t0:.0f}s)")
    records.append({"modelo":"MLP","seed":seed,"n_test":len(test_idx),**m,"Moran_I_holdout":round(mo,6)})

df=pd.DataFrame(records); df.to_csv(OUT_DIR/"mlp_log_replicas.csv",index=False)
pd.DataFrame([{"modelo":"MLP","seeds":str(SEEDS),"n_replicas":len(SEEDS),
    "RMSE_mean":round(df["RMSE"].mean(),4),"RMSE_std":round(df["RMSE"].std(),4),
    "MAE_mean":round(df["MAE"].mean(),4),"MAE_std":round(df["MAE"].std(),4),
    "MAPE_mean":round(df["MAPE"].mean(),4),"MAPE_std":round(df["MAPE"].std(),4),
    "R2_mean":round(df["R2"].mean(),4),"R2_std":round(df["R2"].std(),4),
}]).to_csv(OUT_DIR/"mlp_log_replicas_summary.csv",index=False)
pr(f"\n{df[['seed','RMSE','R2']].to_string(index=False)}")
pr(f"RMSE_mean={df['RMSE'].mean():.2f} ± {df['RMSE'].std():.2f}")
