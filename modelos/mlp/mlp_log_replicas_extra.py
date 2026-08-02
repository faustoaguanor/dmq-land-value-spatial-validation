"""
mlp_log_replicas_extra.py
==========================
Agrega 5 seeds adicionales [99, 1234, 888, 314, 7] al CSV existente de replicas MLP.
Resultado final: 10 seeds [42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7].
"""
from __future__ import annotations
import sys, time, random as _random
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
if not (ROOT / "datos" / "dataset.gpkg").exists():
    ROOT = Path("/workspace/pod")

try:
    from libpysal.weights import KNN as KNNWeights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

COVARIABLES = [
    "suscept_codigo","pc_pnbi","dist_metro","dist_centr_metro",
    "dist_centr_zonal","dist_cc","dist_universidad","dist_hospital",
    "dist_parque_metro","dist_industrial","dist_via_principal",
    "uso_suelo_cod","cos_num","dist_quebrada","dist_mercado_mayorista",
    "dist_plataforma_gub","log_area","frente_m","area_const_m2",
    "tiene_const","num_pisos","antiguedad","topografia_factor",
    "conservacion_cod","acabados_cod","es_ph","pendiente_grados",
]
HIDDEN_LAYERS=[256,128,64]; N_EPOCHS=300; PATIENCE=40; BATCH_SIZE=64
LR=0.001; WD=1e-4; CLIP_GRAD=5.0; VAL_FRAC=0.10; MORAN_K=8
RANDOM_STATE=42; SEEDS_EXTRA=[99,1234,888,314,7]
ALL_SEEDS=[42,2011,456,777,2026,99,1234,888,314,7]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT/"modelos"/"mlp"/"output_log"; OUT_DIR.mkdir(exist_ok=True)

class MLPRegressor(nn.Module):
    def __init__(self,n_feat,hidden):
        super().__init__()
        layers=[]; in_dim=n_feat
        for h in hidden:
            layers+=[nn.Linear(in_dim,h),nn.BatchNorm1d(h),nn.ReLU()]; in_dim=h
        layers.append(nn.Linear(in_dim,1)); self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in"); nn.init.constant_(m.bias,0.0)
    def forward(self,x): return self.net(x).squeeze(-1)

def metrics_log(y_orig,y_pred_log,y_log):
    yp=np.exp(y_pred_log); e=y_orig-yp
    mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2)))
    mape=float(np.mean(np.abs(e/y_orig))*100)
    r2=round(1-np.sum(e**2)/np.sum((y_orig-y_orig.mean())**2),4)
    el=y_log-y_pred_log; r2l=round(1-np.sum(el**2)/np.sum((y_log-y_log.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l}

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
    with torch.no_grad(): return model(torch.tensor(X,dtype=torch.float32).to(DEVICE)).cpu().numpy()

print(f"ROOT={ROOT}  device={DEVICE}",flush=True)
gdf=gpd.read_file(ROOT/"datos"/"dataset.gpkg",layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"]=gdf["predio_join"].astype(int)
split_df=pd.read_csv(ROOT/"data_split"/"split.csv"); split_df["predio_join"]=split_df["predio_join"].astype(int)
gdf=gdf.merge(split_df[["predio_join","split"]],on="predio_join",how="left")
coords=np.column_stack([gdf.geometry.x,gdf.geometry.y])
y_orig=gdf["valor_m2"].values.astype(float); y_log=np.log(y_orig)
X_raw=gdf[COVARIABLES].values.astype(float)
train_mask=(gdf["split"]=="train").values; test_mask=(gdf["split"]=="test").values
train_idx=np.where(train_mask)[0]; test_idx=np.where(test_mask)[0]; n_feat=X_raw.shape[1]

scaler=StandardScaler()
Xtr_sc=scaler.fit_transform(X_raw[train_mask]).astype(np.float32)
Xte_sc=scaler.transform(X_raw[test_mask]).astype(np.float32)
ytr_log=y_log[train_mask].astype(np.float32)

csv_path=OUT_DIR/"mlp_log_replicas.csv"
existing=pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
print(f"Seeds ya en CSV: {list(existing['seed'].unique()) if len(existing) else []}",flush=True)

new_records=[]
for seed in SEEDS_EXTRA:
    print(f"\n[MLP Replica extra seed={seed}]",flush=True)
    t0=time.time()
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type=="cuda": torch.cuda.manual_seed(seed)
    # Val split por-seed (consistente con mlp_log_replicas.py:117)
    vm=np.random.default_rng(seed).random(len(train_idx))<VAL_FRAC; tm=~vm
    y_mean=float(ytr_log[tm].mean()); y_std=float(ytr_log[tm].std())+1e-8
    ytr_norm=((ytr_log[tm]-y_mean)/y_std); yval_norm=((ytr_log[vm]-y_mean)/y_std)
    model=train_mlp(Xtr_sc[tm],ytr_norm,Xtr_sc[vm],yval_norm,n_feat)
    pred_log=predict_mlp(model,Xte_sc)*y_std+y_mean
    m=metrics_log(y_orig[test_mask],pred_log,y_log[test_mask])
    moran_I=float("nan")
    if LIBPYSAL_AVAILABLE:
        w=KNNWeights.from_array(coords[test_mask],k=8); w.transform="r"
        resid=y_log[test_mask]-pred_log; z=resid-resid.mean()
        moran_I=float((z@(w.sparse@z))/(z@z))
    print(f"  seed={seed}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  Moran={moran_I:.4f}  ({time.time()-t0:.0f}s)",flush=True)
    new_records.append({"modelo":"MLP","seed":seed,"n_test":len(test_idx),**m,"Moran_I_holdout":round(moran_I,6)})

df_new=pd.DataFrame(new_records)
df_all=pd.concat([existing,df_new],ignore_index=True) if len(existing) else df_new
df_all.to_csv(csv_path,index=False)
print(f"\nCSV actualizado: {len(df_all)} seeds")
print(df_all[["seed","RMSE","R2"]].to_string(index=False))

summary={"modelo":"MLP","seeds":str(ALL_SEEDS),"n_replicas":len(df_all),
          "RMSE_mean":round(df_all["RMSE"].mean(),4),"RMSE_std":round(df_all["RMSE"].std(),4),
          "MAE_mean":round(df_all["MAE"].mean(),4),"MAE_std":round(df_all["MAE"].std(),4),
          "MAPE_mean":round(df_all["MAPE"].mean(),4),"MAPE_std":round(df_all["MAPE"].std(),4),
          "R2_mean":round(df_all["R2"].mean(),4),"R2_std":round(df_all["R2"].std(),4)}
pd.DataFrame([summary]).to_csv(OUT_DIR/"mlp_log_replicas_summary.csv",index=False)
print(f"RMSE: {summary['RMSE_mean']:.2f} +- {summary['RMSE_std']:.2f}")
