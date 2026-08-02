"""
mlp_log_cv_replicas.py
======================
Replicas de CV (RandomKFold + SpatialBlock) para MLP con 5 seeds.
Para cada seed se corre CV completo (5 folds x 2 estrategias).

Salidas: modelos/mlp/output_log/
  mlp_log_cv_replicas.csv          -- por seed x estrategia x fold
  mlp_log_cv_replicas_seed.csv     -- RMSE_mean+-std por seed x estrategia
  mlp_log_cv_replicas_summary.csv  -- RMSE_mean+-std por estrategia sobre seeds
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
if not (ROOT / "datos" / "dataset.csv").exists():
    ROOT = Path("/workspace/pod")

sys.path.insert(0, str(ROOT / "spatial_cv"))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV

# ── Config IDENTICA a mlp_log.py ──────────────────────────────────────────────
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
LR=0.001; WD=1e-4; CLIP_GRAD=5.0; VAL_FRAC=0.10
RANDOM_STATE=42; SEEDS=[42,2011,456,777,2026]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH  = ROOT/"datos"/"dataset.gpkg"
SPLIT_PATH = ROOT/"data_split"/"split.csv"
FOLDS_PATH = ROOT/"spatial_cv"/"output"/"fold_assignments.csv"
OUT_DIR    = ROOT/"modelos"/"mlp"/"output_log"
OUT_DIR.mkdir(exist_ok=True)

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

def train_mlp(Xtr,ytr,Xval,yval,n_feat):
    model=MLPRegressor(n_feat,HIDDEN_LAYERS).to(DEVICE)
    opt=optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    loss_fn=nn.HuberLoss(delta=1.0)
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.float32)
    Xv=torch.tensor(Xval,dtype=torch.float32).to(DEVICE); yv=torch.tensor(yval,dtype=torch.float32).to(DEVICE)
    loader=DataLoader(TensorDataset(Xt,yt),batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
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

# ── Cargar datos ───────────────────────────────────────────────────────────────
print(f"ROOT={ROOT}  device={DEVICE}",flush=True)
gdf=gpd.read_file(DATA_PATH,layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"]=gdf["predio_join"].astype(int)
split_df=pd.read_csv(SPLIT_PATH); split_df["predio_join"]=split_df["predio_join"].astype(int)
folds_df=pd.read_csv(FOLDS_PATH); folds_df["predio_join"]=folds_df["predio_join"].astype(int)
gdf=gdf.merge(folds_df[["predio_join","fold"]],on="predio_join",how="left")
gdf=gdf.merge(split_df[["predio_join","split"]],on="predio_join",how="left")
y_orig=gdf["valor_m2"].values.astype(float); y_log=np.log(y_orig)
import sys as _s; from pathlib import Path as _P
_s.path.insert(0, str(_P(__file__).resolve().parent.parent))
from features import build_feature_matrix
X_raw, _FEAT = build_feature_matrix(gdf)   # one-hot uso_suelo_cod (27->31)
sp_folds=gdf["fold"].values.astype(int)
train_mask=(gdf["split"]=="train").values
train_idx=np.where(train_mask)[0]
n_feat=X_raw.shape[1]
print(f"MLP CV replicas  device={DEVICE}  train={len(train_idx)}  seeds={SEEDS}",flush=True)

cv_rand=RandomKFoldCV(n_splits=5,random_state=RANDOM_STATE)
cv_block=SpatialBlockCV(folds=sp_folds[train_mask])
splits_r=list(cv_rand.split(X_raw[train_mask]))
splits_b=list(cv_block.split(X_raw[train_mask]))

# ── Pre-calcular datos de fold (fuera del loop de seeds) ──────────────────────
fold_cache={}
for strat,splits in [("RandomKFold",splits_r),("SpatialBlock",splits_b)]:
    fold_cache[strat]=[]
    for fid,(tr,te) in enumerate(splits):
        sc=StandardScaler()
        Xtr=sc.fit_transform(X_raw[train_mask][tr]).astype(np.float32)
        Xte=sc.transform(X_raw[train_mask][te]).astype(np.float32)
        ytr=y_log[train_mask][tr].astype(np.float32)
        ytel=y_log[train_mask][te].astype(np.float32)
        yteo=y_orig[train_mask][te]
        # Val split fijo por fold (independiente de la seed del modelo)
        rng_v=np.random.default_rng(RANDOM_STATE+fid)
        vm=rng_v.random(len(tr))<VAL_FRAC; tm=~vm
        # Target normalization (fija, basada en fold train)
        y_mean=float(ytr[tm].mean()); y_std=float(ytr[tm].std())+1e-8
        fold_cache[strat].append({
            "fid":fid,
            "Xtr":Xtr,"Xte":Xte,"ytr":ytr,"ytel":ytel,"yteo":yteo,
            "vm":vm,"tm":tm,"y_mean":y_mean,"y_std":y_std,
        })

# ── Loop seeds ─────────────────────────────────────────────────────────────────
records_fold=[]; records_seed=[]

for seed in SEEDS:
    print(f"\n{'='*60}\n[MLP CV Replica seed={seed}]\n{'='*60}",flush=True)
    for strat in ["RandomKFold","SpatialBlock"]:
        fold_metrics=[]
        for fd in fold_cache[strat]:
            fid=fd["fid"]; t0=time.time()
            _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if DEVICE.type=="cuda": torch.cuda.manual_seed(seed)

            ytr_norm=((fd["ytr"][fd["tm"]]-fd["y_mean"])/fd["y_std"])
            yval_norm=((fd["ytr"][fd["vm"]]-fd["y_mean"])/fd["y_std"])
            model=train_mlp(fd["Xtr"][fd["tm"]],ytr_norm,fd["Xtr"][fd["vm"]],yval_norm,n_feat)
            pred_tr_log=predict_mlp(model,fd["Xtr"])*fd["y_std"]+fd["y_mean"]
            s_M=smearing_factor(fd["ytr"],pred_tr_log)
            pred_log=predict_mlp(model,fd["Xte"])*fd["y_std"]+fd["y_mean"]
            m=metrics_log(fd["yteo"],pred_log,fd["ytel"],s_M=s_M)
            print(f"  [{strat}] fold{fid+1} seed={seed}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  ({time.time()-t0:.0f}s)",flush=True)
            records_fold.append({"seed":seed,"estrategia":strat,"fold":fid,**m})
            fold_metrics.append(m)

        rmse_vals=[x["RMSE"] for x in fold_metrics]
        r2_vals=[x["R2"] for x in fold_metrics]
        row={"seed":seed,"estrategia":strat,
             "RMSE_mean":round(np.mean(rmse_vals),4),"RMSE_std":round(np.std(rmse_vals,ddof=1),4),
             "R2_mean":round(np.mean(r2_vals),4),"R2_std":round(np.std(r2_vals,ddof=1),4)}
        records_seed.append(row)
        print(f"  [{strat}] seed={seed}  RMSE={row['RMSE_mean']:.2f}+-{row['RMSE_std']:.2f}",flush=True)

df_fold=pd.DataFrame(records_fold); df_seed=pd.DataFrame(records_seed)
df_fold.to_csv(OUT_DIR/"mlp_log_cv_replicas.csv",index=False)
df_seed.to_csv(OUT_DIR/"mlp_log_cv_replicas_seed.csv",index=False)
meta=df_seed.groupby("estrategia").agg(
    n_seeds=("seed","count"),
    RMSE_mean=("RMSE_mean","mean"),RMSE_std=("RMSE_mean","std"),
    R2_mean=("R2_mean","mean"),R2_std=("R2_mean","std")).reset_index().round(4)
meta.to_csv(OUT_DIR/"mlp_log_cv_replicas_summary.csv",index=False)
print(f"\n{'='*60}\nRESUMEN MLP CV REPLICAS\n{'='*60}")
print(meta.to_string(index=False))
