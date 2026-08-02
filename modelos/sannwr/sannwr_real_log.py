"""
sannwr_real_log.py — SANNWR FIEL a Ni et al. (2022)
====================================================
Implementacion canonica para CONTRASTAR con la variante propia de la tesis
(sannwr_log.py, que usa grilla 20x20 + SAPDNN y se rotula "SANNWR*").

SANNWR original (Ni et al., 2022): igual que GNNWR (una SWNN aprende pesos que
modulan los coeficientes OLS locales), pero la ENTRADA de la SWNN combina la
distancia ESPACIAL y la distancia ATRIBUTIVA a cada punto de entrenamiento
mediante un escalar alpha:

    d_hibrida(i,j) = alpha * d_espacial(i,j) + (1 - alpha) * d_atributiva(i,j)

    - d_espacial   = distancia euclidiana entre coordenadas (x,y) UTM
    - d_atributiva = distancia euclidiana en el espacio de atributos estandarizados
    - alpha        = hiperparametro escalar (0..1); por defecto 0.5

Esta es la unica diferencia estructural respecto a GNNWR. Permite responder si la
variante propia (grilla+SAPDNN) cambia o no las conclusiones vs el diseno publicado.

REQUIERE GPU (matriz n_train x n_train + SWNN grande). Target: log(valor_m2).
Salidas: output_log_real/sannwr_real_log_*.csv

NOTA DE PROTOCOLO (2026-07-12): las semillas de los folds de CV ahora se derivan
por estrategia/fold (set_global_seed abajo). Los artefactos almacenados en
output_log_real/ fueron generados con el protocolo anterior (RNG secuencial entre
folds), por lo que una re-ejecucion de la seccion CV producira valores distintos
a los publicados en la tesis (v13, Tablas 2 y 6). El holdout final conserva la
semilla 42 y sigue reproduciendo el artefacto almacenado.
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

_CV_DIR = Path(__file__).parent.parent.parent / "spatial_cv"
sys.path.insert(0, str(_CV_DIR))
from estrategias_cv import RandomKFoldCV, SpatialBlockCV, SpatialBlockBufferedCV

# one-hot de uso_suelo_cod via punto unico de verdad (modelos/features.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import build_feature_matrix

try:
    from libpysal.weights import KNN as KNNWeights
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

# ── Config (alineada con gnnwr_log.py para una comparacion justa) ──────────────
COVARIABLES = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]
ALPHA = 0.5            # peso espacial vs atributivo (hiperparametro escalar del paper)
DENSE_LAYERS = [2048, 1024, 512, 256, 64]; DROP_OUT = 0.2; BATCH_NORM = True
N_EPOCHS = 1000; PATIENCE = 200; BATCH_SIZE = 64
START_LR = 0.2; RANDOM_STATE = 42; MORAN_K = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR.parent.parent / "datos" / "dataset.gpkg"
SPLIT_PATH = BASE_DIR.parent.parent / "data_split" / "split.csv"
FOLDS_PATH = BASE_DIR.parent.parent / "spatial_cv" / "output" / "fold_assignments.csv"
OUT_DIR    = BASE_DIR / "output_log_real"
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)

def pr(*a, **k): print(*a, **k, flush=True)

STRAT_SEED_OFFSETS = {
    "RandomKFold": 0,
    "SpatialBlock": 1000,
    "SpatialBlock_buf": 2000,
    "Holdout": 3000,
}

def set_global_seed(seed: int) -> None:
    """Fija las fuentes de aleatoriedad usadas por numpy, python y torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── SWNN identica a GNNWR (la innovacion de SANNWR esta en la ENTRADA) ─────────
class SWNN(nn.Module):
    def __init__(self, insize, outsize, dense_layers, drop_out=0.2, batch_norm=True):
        super().__init__()
        act = nn.PReLU(init=0.1)
        layers, last = [], insize
        for h in dense_layers:
            layers += [nn.Linear(last, h)]
            if batch_norm: layers += [nn.BatchNorm1d(h)]
            layers += [act, nn.Dropout(drop_out)]
            last = h
        layers += [nn.Linear(last, outsize)]
        self.fc = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self, x): return self.fc(x)

class SANNWRRealModel(nn.Module):
    def __init__(self, n_train, n_features, dense_layers, drop_out, batch_norm, ols_coeff):
        super().__init__()
        self.swnn = SWNN(n_train, n_features, dense_layers, drop_out, batch_norm=batch_norm)
        self.out  = nn.Linear(n_features, 1, bias=False)
        self.out.weight = nn.Parameter(
            torch.tensor(ols_coeff.reshape(1, -1), dtype=torch.float32), requires_grad=False)
    def forward(self, dis, x):
        return self.out(self.swnn(dis) * x)

# ── Funciones ──────────────────────────────────────────────────────────────────
def add_intercept(X): return np.column_stack([np.ones(len(X)), X])
def compute_ols(X_int, y):
    return LinearRegression(fit_intercept=False).fit(X_int, y).coef_.flatten().astype(np.float32)

def metrics_log(y_true_orig, y_pred_log, y_true_log):
    y_pred = np.exp(y_pred_log); e = y_true_orig - y_pred
    mae = float(np.mean(np.abs(e))); rmse = float(np.sqrt(np.mean(e**2)))
    mape = float(np.mean(np.abs(e/y_true_orig))*100)
    r2 = round(1 - np.sum(e**2)/np.sum((y_true_orig-y_true_orig.mean())**2), 4)
    el = y_true_log - y_pred_log
    r2l = round(1 - np.sum(el**2)/np.sum((y_true_log-y_true_log.mean())**2), 4)
    return {"MAE": round(mae,4), "RMSE": round(rmse,4), "MAPE": round(mape,4), "R2": r2, "R2_log": r2l}

def compute_moran(v, c, k=MORAN_K):
    w = KNNWeights.from_array(c, k=k); w.transform = "r"
    z = v - v.mean(); return float((z@(w.sparse@z))/(z@z)), -1/(len(v)-1)

def hybrid_dist(coords_q, X_q, coords_tr, X_tr, sc_sp, sc_at):
    """d_hibrida = alpha*d_espacial_escalada + (1-alpha)*d_atributiva_escalada.
    Los scalers (sc_sp, sc_at) se ajustan SOLO con las distancias de train."""
    d_sp = sc_sp.transform(cdist(coords_q, coords_tr))
    d_at = sc_at.transform(cdist(X_q, X_tr))
    return (ALPHA * d_sp + (1.0 - ALPHA) * d_at).astype(np.float32)

def make_loader(dis, X, y, bs, shuf):
    ds = TensorDataset(torch.tensor(dis, dtype=torch.float32),
                       torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y.reshape(-1,1), dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=shuf)

def one_epoch(model, loader, opt):
    training = opt is not None; model.train(training)
    tot, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for dis, x, y in loader:
            dis, x, y = dis.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
            yh = model(dis, x); loss = F.mse_loss(yh, y)
            if training:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()
            tot += loss.item()*len(y); n += len(y)
    return tot/n

def train_model(model, tr_loader, val_loader, path):
    opt = torch.optim.Adadelta(model.parameters(), lr=START_LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=3, eta_min=0.01)
    best, pat = float("inf"), 0
    for ep in range(1, N_EPOCHS+1):
        tl = one_epoch(model, tr_loader, opt); vl = one_epoch(model, val_loader, None); sched.step()
        if ep % 50 == 0 or ep == 1: pr(f"    ep {ep:4d}  train={tl:.4f}  val={vl:.4f}")
        if vl < best: best = vl; pat = 0; torch.save(model.state_dict(), path)
        else:
            pat += 1
            if pat >= PATIENCE: pr(f"    Early stop ep {ep}"); break
    model.load_state_dict(torch.load(path, map_location=DEVICE))

def predict(model, loader):
    model.eval(); ps = []
    with torch.no_grad():
        for dis, x, _ in loader:
            ps.append(model(dis.to(DEVICE), x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(ps)

def fit_scalers(coords_tr, X_tr):
    sc_sp = StandardScaler().fit(cdist(coords_tr, coords_tr))
    sc_at = StandardScaler().fit(cdist(X_tr, X_tr))
    return sc_sp, sc_at

# ── Cargar datos (one-hot, predio int, orden determinista) ────────────────────
gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT_PATH); folds_df = pd.read_csv(FOLDS_PATH)
for d in (gdf, split_df, folds_df): d["predio_join"] = d["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join","fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)   # orden determinista + 5051 filas

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y]).astype(float)
y_orig = gdf["valor_m2"].values.astype(float); y_log = np.log(y_orig)
X_raw, FEAT = build_feature_matrix(gdf)            # one-hot (30 feat)
sp_folds = gdf["fold"].values.astype(int)
train_mask = (gdf["split"]=="train").values; test_mask = (gdf["split"]=="test").values
train_idx = np.where(train_mask)[0]; test_idx = np.where(test_mask)[0]
pr(f"SANNWR-real (alpha={ALPHA})  device={DEVICE}  train={len(train_idx)}  test={len(test_idx)}  feat={X_raw.shape[1]}")

cv_rand = RandomKFoldCV(n_splits=5, random_state=RANDOM_STATE)
cv_block = SpatialBlockCV(folds=sp_folds[train_mask])
splits_r = list(cv_rand.split(X_raw[train_mask])); splits_b = list(cv_block.split(X_raw[train_mask]))
BUFFER_M = 2530  # rango residual exponencial (~2.5 km): SpatialBlock con separacion train-test garantizada (Roberts et al. 2017)
cv_block_buf = SpatialBlockBufferedCV(folds=sp_folds[train_mask], coords=coords[train_mask], buffer=BUFFER_M)
splits_bb = list(cv_block_buf.split(X_raw[train_mask]))
all_records = []; moran_records = []

def run_fold(strat, tr, te, tag, fold_id):
    fold_seed = RANDOM_STATE + STRAT_SEED_OFFSETS[strat] + int(fold_id)
    set_global_seed(fold_seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_raw[train_mask][tr]); Xte = sc.transform(X_raw[train_mask][te])
    ytr = y_log[train_mask][tr]; yte_l = y_log[train_mask][te]; yte_o = y_orig[train_mask][te]
    ctr = coords[train_mask][tr]; cte = coords[train_mask][te]
    Xtr_i = add_intercept(Xtr); Xte_i = add_intercept(Xte)
    ols_c = compute_ols(Xtr_i, ytr); n_feat = Xtr_i.shape[1]
    sc_sp, sc_at = fit_scalers(ctr, Xtr)              # distancias escaladas con train
    dis_tr = hybrid_dist(ctr, Xtr, ctr, Xtr, sc_sp, sc_at)
    dis_te = hybrid_dist(cte, Xte, ctr, Xtr, sc_sp, sc_at)
    rng = np.random.default_rng(fold_seed)
    vm = rng.random(len(tr)) < 0.10; tm = ~vm
    tr_ld  = make_loader(dis_tr[tm], Xtr_i[tm], ytr[tm], BATCH_SIZE, True)
    val_ld = make_loader(dis_tr[vm], Xtr_i[vm], ytr[vm], BATCH_SIZE, False)
    te_ld  = make_loader(dis_te, Xte_i, yte_l, BATCH_SIZE, False)
    model = SANNWRRealModel(len(tr), n_feat, DENSE_LAYERS, DROP_OUT, BATCH_NORM, ols_c).to(DEVICE)
    train_model(model, tr_ld, val_ld, MODELS_DIR/f"sannwr_real_{tag}.pt")
    return predict(model, te_ld), yte_o, yte_l

for strat, splits in [("RandomKFold", splits_r), ("SpatialBlock", splits_b), ("SpatialBlock_buf", splits_bb)]:
    pr(f"\n{'='*55}\n[SANNWR-real / {strat}]\n{'='*55}")
    for fid, (tr, te) in enumerate(splits):
        pr(f"\n  fold {fid+1}/{len(splits)}  train={len(tr)}  test={len(te)}")
        t0 = time.time()
        pred_log, yte_o, yte_l = run_fold(strat, tr, te, f"{strat}_fold{fid}", fid)
        m = metrics_log(yte_o, pred_log, yte_l)
        pr(f"    MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}  ({time.time()-t0:.0f}s)")
        all_records.append({"modelo":"SANNWR-real","estrategia":strat,"fold":fid,**m})

# ── Holdout final ──────────────────────────────────────────────────────────────
pr(f"\n{'='*55}\n[SANNWR-real / Holdout final]\n{'='*55}")
# Semilla 42 directa: protocolo original con el que se genero el artefacto
# almacenado sannwr_real_log_holdout.csv (el holdout siempre re-sembraba 42).
set_global_seed(RANDOM_STATE)
scf = StandardScaler(); Xtr_f = scf.fit_transform(X_raw[train_mask]); Xte_f = scf.transform(X_raw[test_mask])
Xtr_if = add_intercept(Xtr_f); Xte_if = add_intercept(Xte_f)
ols_cf = compute_ols(Xtr_if, y_log[train_mask]); n_feat_f = Xtr_if.shape[1]
ctr_f = coords[train_mask]; cte_f = coords[test_mask]
sc_sp_f, sc_at_f = fit_scalers(ctr_f, Xtr_f)
dis_tr_f = hybrid_dist(ctr_f, Xtr_f, ctr_f, Xtr_f, sc_sp_f, sc_at_f)
dis_te_f = hybrid_dist(cte_f, Xte_f, ctr_f, Xtr_f, sc_sp_f, sc_at_f)
rng_f = np.random.default_rng(RANDOM_STATE); vm_f = rng_f.random(len(train_idx)) < 0.10; tm_f = ~vm_f
tr_ld_f  = make_loader(dis_tr_f[tm_f], Xtr_if[tm_f], y_log[train_mask][tm_f], BATCH_SIZE, True)
val_ld_f = make_loader(dis_tr_f[vm_f], Xtr_if[vm_f], y_log[train_mask][vm_f], BATCH_SIZE, False)
te_ld_f  = make_loader(dis_te_f, Xte_if, y_log[test_mask], BATCH_SIZE, False)
model_f = SANNWRRealModel(len(train_idx), n_feat_f, DENSE_LAYERS, DROP_OUT, BATCH_NORM, ols_cf).to(DEVICE)
train_model(model_f, tr_ld_f, val_ld_f, MODELS_DIR/"sannwr_real_final.pt")
pred_log_test = predict(model_f, te_ld_f)
m_test = metrics_log(y_orig[test_mask], pred_log_test, y_log[test_mask])
pr(f"  HOLDOUT: MAE={m_test['MAE']:.2f}  RMSE={m_test['RMSE']:.2f}  R2={m_test['R2']:.4f}  R2_log={m_test['R2_log']:.4f}")
if LIBPYSAL_AVAILABLE:
    Ih, EIh = compute_moran(y_log[test_mask]-pred_log_test, cte_f)
    moran_records.append({"modelo":"SANNWR-real","estrategia":"Holdout20%","I":round(Ih,6),"EI":round(EIh,6)})

# ── Predicciones por predio (test+train) para smearing/Duan, DM test y CDF ─────
tr_ld_full = make_loader(dis_tr_f, Xtr_if, y_log[train_mask], BATCH_SIZE, False)
pred_log_train_all = predict(model_f, tr_ld_full)
preds = pd.concat([
    pd.DataFrame({"predio_join": gdf.loc[test_mask, "predio_join"].astype(int).values,
                  "split": "test",  "y_obs_log": y_log[test_mask],  "y_pred_log": pred_log_test}),
    pd.DataFrame({"predio_join": gdf.loc[train_mask, "predio_join"].astype(int).values,
                  "split": "train", "y_obs_log": y_log[train_mask], "y_pred_log": pred_log_train_all}),
], ignore_index=True)
preds["modelo"] = "SANNWR-real"
preds.to_csv(OUT_DIR/"sannwr_real_log_predictions.csv", index=False)

# ── Guardar ────────────────────────────────────────────────────────────────────
res = pd.DataFrame(all_records)
res.to_csv(OUT_DIR/"sannwr_real_log_results.csv", index=False)
res.groupby(["modelo","estrategia"]).agg(
    MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
    RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
    MAPE_mean=("MAPE","mean"), MAPE_std=("MAPE","std"),
    R2_mean=("R2","mean"), R2_std=("R2","std")).reset_index().round(4).to_csv(OUT_DIR/"sannwr_real_log_summary.csv", index=False)
pd.DataFrame([{"modelo":"SANNWR-real","estrategia":"Holdout20%","n_test":len(test_idx),**m_test}]
             ).to_csv(OUT_DIR/"sannwr_real_log_holdout.csv", index=False)
if moran_records: pd.DataFrame(moran_records).to_csv(OUT_DIR/"sannwr_real_log_moran.csv", index=False)
pr(f"\n[CSV] {OUT_DIR}")
