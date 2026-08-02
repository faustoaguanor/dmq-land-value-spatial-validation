"""
gnnwr_log_replicas_extra.py
============================
Agrega 5 seeds adicionales [99, 1234, 888, 314, 7] al CSV existente de replicas GNNWR.
Resultado final: 10 seeds [42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7].
Appendea al CSV existente sin re-correr los seeds anteriores.
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
DENSE_LAYERS=[2048,1024,512,256,64]; DROP_OUT=0.2; BATCH_NORM=True
N_EPOCHS=1000; PATIENCE=200; BATCH_SIZE=64; START_LR=0.2; MORAN_K=8
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS_EXTRA=[99,1234,888,314,7]
ALL_SEEDS=[42,2011,456,777,2026,99,1234,888,314,7]

OUT_DIR = ROOT/"modelos"/"gnnwr"/"output_log"
OUT_DIR.mkdir(exist_ok=True)

class SWNN(nn.Module):
    def __init__(self,insize,outsize,dense_layers,drop_out,batch_norm):
        super().__init__()
        act=nn.PReLU(init=0.1); layers=[]; last=insize
        for h in dense_layers:
            layers+=[nn.Linear(last,h)]
            if batch_norm: layers+=[nn.BatchNorm1d(h)]
            layers+=[act,nn.Dropout(drop_out)]; last=h
        layers+=[nn.Linear(last,outsize)]; self.fc=nn.Sequential(*layers)
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
        self.out.weight=nn.Parameter(torch.tensor(ols_coeff.reshape(1,-1),dtype=torch.float32),requires_grad=False)
    def forward(self,dis,x): return self.out(self.swnn(dis)*x)

def add_intercept(X): return np.column_stack([np.ones(len(X)),X])
def compute_ols(Xi,y): return LinearRegression(fit_intercept=False).fit(Xi,y).coef_.flatten().astype(np.float32)
def metrics_log(y_orig,y_pred_log,y_log):
    yp=np.exp(y_pred_log); e=y_orig-yp
    mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2)))
    mape=float(np.mean(np.abs(e/y_orig))*100)
    r2=round(1-np.sum(e**2)/np.sum((y_orig-y_orig.mean())**2),4)
    el=y_log-y_pred_log; r2l=round(1-np.sum(el**2)/np.sum((y_log-y_log.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"MAPE":round(mape,4),"R2":r2,"R2_log":r2l}
def make_loader(dis,X,y,bs,shuf):
    ds=TensorDataset(torch.tensor(dis,dtype=torch.float32),torch.tensor(X,dtype=torch.float32),
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
    model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=False))
def predict(model,loader):
    model.eval(); ps=[]
    with torch.no_grad():
        for dis,x,_ in loader: ps.append(model(dis.to(DEVICE),x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(ps)

print(f"ROOT={ROOT}  device={DEVICE}",flush=True)
gdf=gpd.read_file(ROOT/"datos"/"dataset.gpkg",layer="puntos_mercado").to_crs(epsg=32717)
gdf["predio_join"]=gdf["predio_join"].astype(int)
split_df=pd.read_csv(ROOT/"data_split"/"split.csv"); split_df["predio_join"]=split_df["predio_join"].astype(int)
gdf=gdf.merge(split_df[["predio_join","split"]],on="predio_join",how="left")
coords=np.column_stack([gdf.geometry.x,gdf.geometry.y])
y_orig=gdf["valor_m2"].values.astype(float); y_log=np.log(y_orig)
X_raw=gdf[COVARIABLES].values.astype(float)
train_mask=(gdf["split"]=="train").values; test_mask=(gdf["split"]=="test").values
train_idx=np.where(train_mask)[0]; test_idx=np.where(test_mask)[0]
print(f"GNNWR replicas extra  seeds={SEEDS_EXTRA}",flush=True)

print("Calculando distancias ...",flush=True)
sc_f=StandardScaler()
Xtr_f=sc_f.fit_transform(X_raw[train_mask]); Xte_f=sc_f.transform(X_raw[test_mask])
Xtr_if=add_intercept(Xtr_f).astype(np.float32); Xte_if=add_intercept(Xte_f).astype(np.float32)
ols_cf=compute_ols(Xtr_if,y_log[train_mask]); n_feat_f=Xtr_if.shape[1]
ctr_f=coords[train_mask]; cte_f=coords[test_mask]
dis_tr_f=cdist(ctr_f,ctr_f); dis_te_f=cdist(cte_f,ctr_f)
ds_f=StandardScaler()
dis_tr_f_sc=ds_f.fit_transform(dis_tr_f).astype(np.float32)
dis_te_f_sc=ds_f.transform(dis_te_f).astype(np.float32)
print("Distancias listas.",flush=True)

# Leer CSV existente
csv_path=OUT_DIR/"gnnwr_log_replicas.csv"
existing=pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
print(f"Seeds ya en CSV: {list(existing['seed'].unique()) if len(existing) else []}",flush=True)

new_records=[]
for seed in SEEDS_EXTRA:
    print(f"\n{'='*60}\n[GNNWR Replica extra seed={seed}]\n{'='*60}",flush=True)
    t0=time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    if DEVICE.type=="cuda": torch.cuda.manual_seed(seed)
    rng_f=np.random.default_rng(seed); vm_f=rng_f.random(len(train_idx))<0.10; tm_f=~vm_f
    tr_ld=make_loader(dis_tr_f_sc[tm_f],Xtr_if[tm_f],y_log[train_mask][tm_f].astype(np.float32),BATCH_SIZE,True)
    val_ld=make_loader(dis_tr_f_sc[vm_f],Xtr_if[vm_f],y_log[train_mask][vm_f].astype(np.float32),BATCH_SIZE,False)
    te_ld=make_loader(dis_te_f_sc,Xte_if,y_log[test_mask].astype(np.float32),BATCH_SIZE,False)
    mp=OUT_DIR/f"_replica_gnnwr_seed{seed}.pt"
    model=GNNWRModel(len(train_idx),n_feat_f,DENSE_LAYERS,DROP_OUT,BATCH_NORM,ols_cf).to(DEVICE)
    print("  Entrenando ...",flush=True)
    train_model(model,tr_ld,val_ld,mp)
    pred=predict(model,te_ld)
    m=metrics_log(y_orig[test_mask],pred,y_log[test_mask])
    moran_I=float("nan")
    if LIBPYSAL_AVAILABLE:
        from libpysal.weights import KNN as KNNWeights
        w=KNNWeights.from_array(cte_f,k=8); w.transform="r"
        resid=y_log[test_mask]-pred; z=resid-resid.mean()
        moran_I=float((z@(w.sparse@z))/(z@z))
    print(f"  seed={seed}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  Moran={moran_I:.4f}  ({time.time()-t0:.0f}s)",flush=True)
    new_records.append({"modelo":"GNNWR","seed":seed,"n_test":len(test_idx),**m,"Moran_I_holdout":round(moran_I,6)})
    mp.unlink(missing_ok=True)
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

# Merge y guardar
df_new=pd.DataFrame(new_records)
df_all=pd.concat([existing,df_new],ignore_index=True) if len(existing) else df_new
df_all.to_csv(csv_path,index=False)
print(f"\nCSV actualizado: {len(df_all)} seeds totales")
print(df_all[["seed","RMSE","R2","Moran_I_holdout"]].to_string(index=False))

summary={"modelo":"GNNWR","seeds":str(ALL_SEEDS),"n_replicas":len(df_all),
          "RMSE_mean":round(df_all["RMSE"].mean(),4),"RMSE_std":round(df_all["RMSE"].std(),4),
          "MAE_mean":round(df_all["MAE"].mean(),4),"MAE_std":round(df_all["MAE"].std(),4),
          "MAPE_mean":round(df_all["MAPE"].mean(),4),"MAPE_std":round(df_all["MAPE"].std(),4),
          "R2_mean":round(df_all["R2"].mean(),4),"R2_std":round(df_all["R2"].std(),4)}
pd.DataFrame([summary]).to_csv(OUT_DIR/"gnnwr_log_replicas_summary.csv",index=False)
print(f"\nRMSE: {summary['RMSE_mean']:.2f} +- {summary['RMSE_std']:.2f}")
