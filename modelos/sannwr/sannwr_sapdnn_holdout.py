"""
sannwr_sapdnn_holdout.py — Experimento controlado: ¿la red que APRENDE la distancia
(SAPDNN, Ni et al. 2022) supera al α=0.5 FIJO del SANNWR canónico?
====================================================================================
Aísla el componente que faltaba: punto-a-punto (como el canónico) + fusión de distancia
espacial/atributiva, en dos formas:
  - alpha FIJO lineal:  d = a*d_sp + (1-a)*d_at   (a = 0.3, 0.5, 0.7)
  - SAPDNN aprendido:   d = SAPNN(d_sp, d_at)      (red 2->h->1, Ni 2022 fiel, SIN grilla)

Todo en LA MISMA máquina, MISMA seed=42, MISMO pipeline → comparación limpia (el efecto
de hardware/seed se cancela). Solo holdout 20%. NO toca los CSV canónicos.
Salida: modelos/sannwr/output_log_sapdnn_test/sapdnn_vs_alpha_holdout.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "spatial_cv")); sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix

DENSE_LAYERS=[2048,1024,512,256,64]; DROP_OUT=0.2; BATCH_NORM=True
N_EPOCHS=1000; PATIENCE=200; BATCH_SIZE=64; START_LR=0.2; SEED=42; SAPNN_HIDDEN=4
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT=Path(__file__).parent/"output_log_sapdnn_test"; OUT.mkdir(exist_ok=True)
MODELS=OUT/"models"; MODELS.mkdir(exist_ok=True)
def pr(*a,**k): print(*a,**k,flush=True)

class SWNN(nn.Module):
    def __init__(self, insize, outsize, dense_layers, drop_out=0.2, batch_norm=True):
        super().__init__()
        act=nn.PReLU(init=0.1); layers, last=[], insize
        for h in dense_layers:
            layers+=[nn.Linear(last,h)]
            if batch_norm: layers+=[nn.BatchNorm1d(h)]
            layers+=[act, nn.Dropout(drop_out)]; last=h
        layers+=[nn.Linear(last,outsize)]; self.fc=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self,x): return self.fc(x)

class SANNWR_fixed(nn.Module):
    """alpha fijo: la fusion se hace fuera (en el loader); el modelo es SWNN+OLR."""
    def __init__(self, n_train, n_features, ols_coeff):
        super().__init__()
        self.swnn=SWNN(n_train, n_features, DENSE_LAYERS, DROP_OUT, BATCH_NORM)
        self.out=nn.Linear(n_features,1,bias=False)
        self.out.weight=nn.Parameter(torch.tensor(ols_coeff.reshape(1,-1),dtype=torch.float32),requires_grad=False)
    def forward(self, fused, x): return self.out(self.swnn(fused)*x)

class SANNWR_sapdnn(nn.Module):
    """SAPDNN aprendido (Ni 2022 fiel, sin grilla): SAPNN(d_sp,d_at) -> SWNN -> OLR."""
    def __init__(self, n_train, n_features, ols_coeff, hidden=SAPNN_HIDDEN):
        super().__init__()
        self.sapnn=nn.Sequential(nn.Linear(2,hidden), nn.PReLU(init=0.1),
                                 nn.Linear(hidden,1), nn.PReLU(init=0.1))
        self.swnn=SWNN(n_train, n_features, DENSE_LAYERS, DROP_OUT, BATCH_NORM)
        self.out=nn.Linear(n_features,1,bias=False)
        self.out.weight=nn.Parameter(torch.tensor(ols_coeff.reshape(1,-1),dtype=torch.float32),requires_grad=False)
        for m in self.sapnn.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,a=0,mode="fan_in"); m.bias.data.fill_(0.0)
    def forward(self, d_sp, d_at, x):
        pair=torch.stack([d_sp,d_at],dim=-1)          # (B, n_train, 2)
        fused=self.sapnn(pair).squeeze(-1)            # (B, n_train)
        return self.out(self.swnn(fused)*x)

def add_intercept(X): return np.column_stack([np.ones(len(X)),X]).astype(np.float32)
def compute_ols(Xi,y): return LinearRegression(fit_intercept=False).fit(Xi,y).coef_.flatten().astype(np.float32)
def metrics(y_o, pred_log, y_l, s_M):
    yp=np.exp(pred_log)*s_M; e=y_o-yp
    mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e**2)))
    r2=round(1-np.sum(e**2)/np.sum((y_o-y_o.mean())**2),4)
    return {"MAE":round(mae,4),"RMSE":round(rmse,4),"R2":r2,"smearing":round(s_M,5)}

def seed_all():
    import random; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ── datos (idéntico al canónico) ──────────────────────────────────────────────
gdf=gpd.read_file(ROOT/"datos"/"dataset.gpkg",layer="puntos_mercado").to_crs(epsg=32717)
sp=pd.read_csv(ROOT/"data_split"/"split.csv")
for d in (gdf,sp): d["predio_join"]=d["predio_join"].astype(int)
gdf=gdf.merge(sp[["predio_join","split"]],on="predio_join",how="left").sort_values("predio_join").reset_index(drop=True)
coords=np.column_stack([gdf.geometry.x,gdf.geometry.y]).astype(float)
y_o=gdf["valor_m2"].values.astype(float); y_l=np.log(y_o)
X_raw,_=build_feature_matrix(gdf)
tm_=(gdf["split"]=="train").values; te_=(gdf["split"]=="test").values
pr(f"device={DEVICE} train={tm_.sum()} test={te_.sum()} feat={X_raw.shape[1]}")

# preparacion comun (seed fija): scaler features, OLS, distancias escaladas
seed_all()
scf=StandardScaler(); Xtr=scf.fit_transform(X_raw[tm_]); Xte=scf.transform(X_raw[te_])
Xtr_i=add_intercept(Xtr); Xte_i=add_intercept(Xte)
ols=compute_ols(Xtr_i,y_l[tm_]); nfeat=Xtr_i.shape[1]
ctr,cte=coords[tm_],coords[te_]
sc_sp=StandardScaler().fit(cdist(ctr,ctr)); sc_at=StandardScaler().fit(cdist(Xtr,Xtr))
dsp_tr=sc_sp.transform(cdist(ctr,ctr)).astype(np.float32); dat_tr=sc_at.transform(cdist(Xtr,Xtr)).astype(np.float32)
dsp_te=sc_sp.transform(cdist(cte,ctr)).astype(np.float32); dat_te=sc_at.transform(cdist(Xte,Xtr)).astype(np.float32)
rng=np.random.default_rng(SEED); vm=rng.random(tm_.sum())<0.10; tmk=~vm
s_M=lambda pred_tr: float(np.mean(np.exp(y_l[tm_]-pred_tr)))

def one_epoch(model, loader, opt, sapdnn):
    training=opt is not None; model.train(training); tot,n=0.0,0
    ctx=torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            *xs,y=[b.to(DEVICE) for b in batch]
            yh=model(*xs); loss=F.mse_loss(yh,y)
            if training: opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            tot+=loss.item()*len(y); n+=len(y)
    return tot/n

def train(model, tr_ld, val_ld, path, sapdnn):
    opt=torch.optim.Adadelta(model.parameters(),lr=START_LR,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=100,T_mult=3,eta_min=0.01)
    best,pat=float("inf"),0
    for ep in range(1,N_EPOCHS+1):
        tl=one_epoch(model,tr_ld,opt,sapdnn); vl=one_epoch(model,val_ld,None,sapdnn); sched.step()
        if ep%100==0: pr(f"      ep{ep} train={tl:.4f} val={vl:.4f}")
        if vl<best: best=vl; pat=0; torch.save(model.state_dict(),path)
        else:
            pat+=1
            if pat>=PATIENCE: pr(f"      early stop ep{ep}"); break
    model.load_state_dict(torch.load(path,map_location=DEVICE))

def predict(model, loader):
    model.eval(); ps=[]
    with torch.no_grad():
        for batch in loader:
            xs=[b.to(DEVICE) for b in batch[:-1]]
            ps.append(model(*xs).cpu().numpy().flatten())
    return np.concatenate(ps)

def run_fixed(alpha):
    seed_all()
    fused_tr=(alpha*dsp_tr+(1-alpha)*dat_tr).astype(np.float32)
    fused_te=(alpha*dsp_te+(1-alpha)*dat_te).astype(np.float32)
    def ld(mask,sh): return DataLoader(TensorDataset(
        torch.tensor(fused_tr[mask]),torch.tensor(Xtr_i[mask]),torch.tensor(y_l[tm_][mask].reshape(-1,1),dtype=torch.float32)),
        batch_size=BATCH_SIZE,shuffle=sh,drop_last=sh)
    te_ld=DataLoader(TensorDataset(torch.tensor(fused_te),torch.tensor(Xte_i),torch.tensor(y_l[te_].reshape(-1,1),dtype=torch.float32)),batch_size=BATCH_SIZE)
    tr_full=DataLoader(TensorDataset(torch.tensor(fused_tr),torch.tensor(Xtr_i),torch.tensor(y_l[tm_].reshape(-1,1),dtype=torch.float32)),batch_size=BATCH_SIZE)
    m=SANNWR_fixed(tm_.sum(),nfeat,ols).to(DEVICE)
    train(m,ld(tmk,True),ld(vm,False),MODELS/f"fixed_{alpha}.pt",False)
    sm=s_M(predict(m,tr_full)); return metrics(y_o[te_],predict(m,te_ld),y_l[te_],sm)

def run_sapdnn():
    seed_all()
    def ld(mask,sh): return DataLoader(TensorDataset(
        torch.tensor(dsp_tr[mask]),torch.tensor(dat_tr[mask]),torch.tensor(Xtr_i[mask]),
        torch.tensor(y_l[tm_][mask].reshape(-1,1),dtype=torch.float32)),batch_size=BATCH_SIZE,shuffle=sh,drop_last=sh)
    te_ld=DataLoader(TensorDataset(torch.tensor(dsp_te),torch.tensor(dat_te),torch.tensor(Xte_i),torch.tensor(y_l[te_].reshape(-1,1),dtype=torch.float32)),batch_size=BATCH_SIZE)
    tr_full=DataLoader(TensorDataset(torch.tensor(dsp_tr),torch.tensor(dat_tr),torch.tensor(Xtr_i),torch.tensor(y_l[tm_].reshape(-1,1),dtype=torch.float32)),batch_size=BATCH_SIZE)
    m=SANNWR_sapdnn(tm_.sum(),nfeat,ols).to(DEVICE)
    train(m,ld(tmk,True),ld(vm,False),MODELS/"sapdnn.pt",True)
    sm=s_M(predict(m,tr_full)); return metrics(y_o[te_],predict(m,te_ld),y_l[te_],sm)

recs=[]; csv=OUT/"sapdnn_vs_alpha_holdout.csv"
for name,fn in [("alpha=0.5 (canónico)",lambda:run_fixed(0.5)),
                ("SAPDNN (Ni fiel, sin grilla)",run_sapdnn),
                ("alpha=0.3",lambda:run_fixed(0.3)),
                ("alpha=0.7",lambda:run_fixed(0.7))]:
    pr(f"\n===== {name} ====="); t0=time.time()
    r=fn(); r={"modo":name,**r,"seg":round(time.time()-t0)}
    pr(f"  -> RMSE={r['RMSE']:.2f} MAE={r['MAE']:.2f} R2={r['R2']:.4f} ({r['seg']}s)")
    recs.append(r); pd.DataFrame(recs).to_csv(csv,index=False)  # guardado incremental
pr(f"\n[CSV] {csv}")
pr(pd.DataFrame(recs).to_string(index=False))
