"""
verif_gwr_intercepto.py — VERIFICACION del hallazgo adversarial (ChatGPT 2026-06-19, #1):
=========================================================================================
¿El 'colapso estructural de GWR' es un artefacto de penalizar el intercepto en el Ridge
y de un lambda no calibrado?

Reproduce los folds de SpatialBlock y SpatialBlock_buf del GWR-27 (mismo pipeline que
gwr_log_27vars.py: BW por fold con encoding continuo, prediccion one-hot+Ridge) con 4
configuraciones, reportando la MEDIANA de MAE por fold (robusta al fold que explota):
  (a) lambda=0.1, intercepto PENALIZADO   (implementacion actual)
  (b) lambda=0.1, intercepto NO penalizado
  (c) lambda=1.0, intercepto NO penalizado
  (d) lambda=10,  intercepto NO penalizado

El BW se selecciona UNA vez por fold (no depende de lambda/intercepto) y se reutiliza.
Salida: analisis/output_log/verif_gwr_intercepto.csv
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE/"spatial_cv")); sys.path.insert(0, str(BASE/"modelos"))
from estrategias_cv import SpatialBlockCV, SpatialBlockBufferedCV
from features import build_feature_matrix
try:
    from mgwr.sel_bw import Sel_BW; MGWR=True
except ImportError: MGWR=False
warnings.filterwarnings("ignore")

BW_MIN, BW_MAX, BW_FALLBACK = 40, 500, 199
BUFFER_M = 2530
DATA=BASE/"datos"/"dataset.gpkg"; SPLIT=BASE/"data_split"/"split.csv"; FOLDS=BASE/"spatial_cv"/"output"/"fold_assignments.csv"
OUT=BASE/"analisis"/"output_log"/"verif_gwr_intercepto.csv"

def add_int(X): return np.column_stack([np.ones(len(X)), X])

def select_bw(coords_tr, X_int, y_tr):
    if not MGWR: return BW_FALLBACK
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel=Sel_BW(coords_tr, y_tr.reshape(-1,1), X_int, fixed=False, kernel="bisquare")
            bw=sel.search(bw_min=BW_MIN, bw_max=BW_MAX, search_method="golden_section", criterion="AICc")
        return max(BW_MIN, int(round(float(bw))))
    except Exception: return BW_FALLBACK

def predict_gwr(coords_tr, X_tr_int, y_tr, coords_te, X_te_int, bw, lam, penalize_intercept):
    p=X_tr_int.shape[1]
    I_p=np.eye(p)
    if not penalize_intercept:
        I_p[0,0]=0.0   # NO penalizar el intercepto (columna 0)
    k_q=min(bw, len(coords_tr))
    nbrs=NearestNeighbors(n_neighbors=k_q, algorithm="ball_tree").fit(coords_tr)
    h_te=nbrs.kneighbors(coords_te)[0][:,-1]
    preds=np.zeros(len(coords_te))
    for j in range(len(coords_te)):
        d=np.sqrt(((coords_tr-coords_te[j])**2).sum(1)); u=d/(h_te[j]+1e-10)
        w=np.where(u<1,(1-u**2)**2,0.0); Xw=X_tr_int*w[:,None]
        A=Xw.T@X_tr_int+lam*I_p; b=X_tr_int.T@(w*y_tr)
        try: beta=np.linalg.solve(A,b)
        except np.linalg.LinAlgError: beta=np.linalg.solve(Xw.T@X_tr_int+(lam*10)*I_p,b)
        preds[j]=X_te_int[j]@beta
    return preds

# datos
gdf=gpd.read_file(DATA,layer="puntos_mercado").to_crs(epsg=32717)
sd=pd.read_csv(SPLIT); fd=pd.read_csv(FOLDS)
for d in (gdf,sd,fd): d["predio_join"]=d["predio_join"].astype(int)
gdf=gdf.merge(fd[["predio_join","fold"]],on="predio_join",how="left").merge(sd[["predio_join","split"]],on="predio_join",how="left")
gdf=gdf.sort_values("predio_join").reset_index(drop=True)
coords=np.column_stack([gdf.geometry.x,gdf.geometry.y]); y_orig=gdf["valor_m2"].values.astype(float); y_log=np.log(y_orig)
X_oh,_=build_feature_matrix(gdf); X_cont,_=build_feature_matrix(gdf,one_hot=False)
train_mask=(gdf["split"]=="train").values; sp_folds=gdf["fold"].values.astype(int)

CONFIGS=[("a_pen_l0.1",0.1,True),("b_nopen_l0.1",0.1,False),("c_nopen_l1",1.0,False),("d_nopen_l10",10.0,False)]
cv_sb=SpatialBlockCV(folds=sp_folds[train_mask])
cv_bb=SpatialBlockBufferedCV(folds=sp_folds[train_mask],coords=coords[train_mask],buffer=BUFFER_M)

rows=[]
for strat,cv in [("SpatialBlock",cv_sb),("SpatialBlock_buf",cv_bb)]:
    splits=list(cv.split(X_oh[train_mask]))
    fold_mae={c[0]:[] for c in CONFIGS}
    fold_mae_clip={c[0]:[] for c in CONFIGS}
    for fid,(tr,te) in enumerate(splits):
        sc=StandardScaler(); Xtr=sc.fit_transform(X_oh[train_mask][tr]); Xte=sc.transform(X_oh[train_mask][te])
        ytr=y_log[train_mask][tr]; yteo=y_orig[train_mask][te]
        ctr=coords[train_mask][tr]; cte=coords[train_mask][te]
        Xtr_bw=add_int(StandardScaler().fit_transform(X_cont[train_mask][tr]))
        bw=select_bw(ctr,Xtr_bw,ytr)
        lo,hi=ytr.min(),ytr.max()
        Xtr_i=add_int(Xtr); Xte_i=add_int(Xte)
        for name,lam,pen in CONFIGS:
            pl=predict_gwr(ctr,Xtr_i,ytr,cte,Xte_i,bw,lam,pen)
            mae=float(np.mean(np.abs(yteo-np.exp(pl))))
            mae_clip=float(np.mean(np.abs(yteo-np.exp(np.clip(pl,lo,hi)))))
            fold_mae[name].append(mae); fold_mae_clip[name].append(mae_clip)
        print(f"  {strat} fold{fid+1} bw={bw} done",flush=True)
    for name,lam,pen in CONFIGS:
        rows.append({"estrategia":strat,"config":name,"lambda":lam,"intercepto_penalizado":pen,
                     "MAE_mediana_folds":round(float(np.median(fold_mae[name])),2),
                     "MAE_mediana_folds_clip":round(float(np.median(fold_mae_clip[name])),2),
                     "MAE_folds":[round(x,1) for x in fold_mae[name]]})
        print(f"[{strat}] {name}: mediana MAE={np.median(fold_mae[name]):.2f} (clip {np.median(fold_mae_clip[name]):.2f})",flush=True)

df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
print("\n=== RESUMEN ==="); print(df[["estrategia","config","MAE_mediana_folds","MAE_mediana_folds_clip"]].to_string(index=False))
print(f"\n[CSV] {OUT}")
