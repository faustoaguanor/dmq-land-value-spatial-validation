"""
o4_control_muestral_gpu.py
==========================
O4 (parte neural): control de tamaño muestral del buffer para MLP y GNNWR.
Mismo diseño que o4_control_muestral_cpu.py: por cada fold buffered se compara, sobre
el MISMO test fold, el entrenamiento BUFFERED (sep. > 2,530 m) contra un entrenamiento
ALEATORIO del mismo tamaño extraído del complemento (R seeds, sin criterio espacial).

  ΔMAE = MAE_buffered − MAE_aleatorio_mismo_n  > 0  => degradación por SEPARACIÓN espacial.

Uso:
  python analisis/o4_control_muestral_gpu.py --models mlp           # rápido
  python analisis/o4_control_muestral_gpu.py --models gnnwr --r 3    # lento
  python analisis/o4_control_muestral_gpu.py --models mlp,gnnwr

Salida: analisis/output_log/o4_control_muestral_<modelo>.csv
"""
from __future__ import annotations
import sys, math, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings; warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "spatial_cv"))
sys.path.insert(0, str(BASE / "modelos"))
from estrategias_cv import SpatialBlockBufferedCV
from features import build_feature_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_M = 2530
OUTDIR = BASE / "analisis" / "output_log"

# ── MLP (idéntico a modelos/mlp/mlp_log.py) ──────────────────────────────────────
MLP_HIDDEN = [256, 128, 64]; MLP_EPOCHS = 300; MLP_PAT = 40
MLP_BS = 64; MLP_LR = 1e-3; MLP_WD = 1e-4; MLP_CLIP = 5.0; VAL_FRAC = 0.10

class MLPReg(nn.Module):
    def __init__(self, n_feat, hidden):
        super().__init__()
        L, d = [], n_feat
        for h in hidden:
            L += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]; d = h
        L.append(nn.Linear(d, 1)); self.net = nn.Sequential(*L)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in"); nn.init.constant_(m.bias, 0.0)
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(Xtr, ytr, Xv, yv, n_feat, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = MLPReg(n_feat, MLP_HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    lf = nn.HuberLoss(delta=1.0)
    ld = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(ytr, dtype=torch.float32)),
                    batch_size=MLP_BS, shuffle=True, drop_last=True)
    Xv = torch.tensor(Xv, dtype=torch.float32).to(DEVICE); yv = torch.tensor(yv, dtype=torch.float32).to(DEVICE)
    best, bs, ni = float("inf"), None, 0
    for ep in range(MLP_EPOCHS):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
            loss = lf(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MLP_CLIP); opt.step()
        model.eval()
        with torch.no_grad(): vl = lf(model(Xv), yv).item()
        if vl < best - 1e-6: best, bs, ni = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            ni += 1
            if ni >= MLP_PAT: break
    if bs: model.load_state_dict(bs)
    return model

def mlp_predict_log(Xtr, ytr_log, Xte, seed):
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    rng = np.random.default_rng(seed); vm = rng.random(len(Xtr_s)) < VAL_FRAC; tm = ~vm
    ym, ys = float(ytr_log[tm].mean()), float(ytr_log[tm].std()) + 1e-8
    model = train_mlp(Xtr_s[tm], ((ytr_log[tm]-ym)/ys).astype(np.float32),
                      Xtr_s[vm], ((ytr_log[vm]-ym)/ys).astype(np.float32), Xtr.shape[1], seed)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return out * ys + ym

# ── GNNWR (idéntico a modelos/gnnwr/gnnwr_log.py) ────────────────────────────────
G_DENSE = [2048, 1024, 512, 256, 64]; G_DROP = 0.2; G_EPOCHS = 1000; G_PAT = 200
G_BS = 64; G_LR = 0.2

class SWNN(nn.Module):
    def __init__(self, insize, outsize, dense, drop):
        super().__init__()
        act = nn.PReLU(init=0.1); L, last = [], insize
        for h in dense:
            L += [nn.Linear(last, h), nn.BatchNorm1d(h), act, nn.Dropout(drop)]; last = h
        L += [nn.Linear(last, outsize)]; self.fc = nn.Sequential(*L)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self, x): return self.fc(x)

class GNNWRModel(nn.Module):
    def __init__(self, n_train, n_feat, ols_coeff):
        super().__init__()
        self.swnn = SWNN(n_train, n_feat, G_DENSE, G_DROP)
        self.out = nn.Linear(n_feat, 1, bias=False)
        self.out.weight = nn.Parameter(torch.tensor(ols_coeff.reshape(1, -1), dtype=torch.float32), requires_grad=False)
    def forward(self, dis, x): return self.out(self.swnn(dis) * x)

def gnnwr_predict_log(Xtr, ytr_log, ctr, Xte, cte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    Xtr_i = np.column_stack([np.ones(len(Xtr_s)), Xtr_s])
    Xte_i = np.column_stack([np.ones(len(Xte_s)), Xte_s])
    ols = LinearRegression(fit_intercept=False).fit(Xtr_i, ytr_log).coef_.flatten().astype(np.float32)
    dis_tr = cdist(ctr, ctr); dis_te = cdist(cte, ctr)
    ds = StandardScaler(); dis_tr = ds.fit_transform(dis_tr); dis_te = ds.transform(dis_te)
    rng = np.random.default_rng(seed); vm = rng.random(len(Xtr_i)) < 0.10; tm = ~vm
    def loader(d, x, y, shuf):
        return DataLoader(TensorDataset(torch.tensor(d, dtype=torch.float32),
                                        torch.tensor(x, dtype=torch.float32),
                                        torch.tensor(y.reshape(-1, 1), dtype=torch.float32)),
                          batch_size=G_BS, shuffle=shuf, drop_last=shuf)
    tl = loader(dis_tr[tm], Xtr_i[tm], ytr_log[tm], True)
    vl = loader(dis_tr[vm], Xtr_i[vm], ytr_log[vm], False)
    model = GNNWRModel(len(Xtr_i), Xtr_i.shape[1], ols).to(DEVICE)
    opt = torch.optim.Adadelta(model.parameters(), lr=G_LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=3, eta_min=0.01)
    def run(ld, train):
        model.train(train); tot, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for d, x, y in ld:
                d, x, y = d.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
                loss = F.mse_loss(model(d, x), y)
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                tot += loss.item()*len(y); n += len(y)
        return tot/n
    best, bs, ni = float("inf"), None, 0
    for ep in range(1, G_EPOCHS+1):
        run(tl, True); v = run(vl, False); sch.step()
        if v < best: best, bs, ni = v, {k: val.cpu().clone() for k, val in model.state_dict().items()}, 0
        else:
            ni += 1
            if ni >= G_PAT: break
    if bs: model.load_state_dict(bs)
    model.eval(); preds = []
    te = loader(dis_te, Xte_i, np.zeros(len(Xte_i)), False)
    with torch.no_grad():
        for d, x, _ in te: preds.append(model(d.to(DEVICE), x.to(DEVICE)).cpu().numpy().flatten())
    return np.concatenate(preds)

# ── Datos ──────────────────────────────────────────────────────────────────────
def load():
    gdf = gpd.read_file(BASE/"datos"/"dataset.gpkg", layer="puntos_mercado").to_crs(epsg=32717)
    sp = pd.read_csv(BASE/"data_split"/"split.csv"); fo = pd.read_csv(BASE/"spatial_cv"/"output"/"fold_assignments.csv")
    for df in (gdf, sp, fo): df["predio_join"] = df["predio_join"].astype(int)
    gdf = gdf.merge(fo[["predio_join","fold"]], on="predio_join", how="left").merge(sp[["predio_join","split"]], on="predio_join", how="left")
    gdf = gdf.sort_values("predio_join").reset_index(drop=True)
    C = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    yl = np.log(gdf["valor_m2"].values.astype(float)); yo = gdf["valor_m2"].values.astype(float)
    X, _ = build_feature_matrix(gdf)
    tm = (gdf["split"] == "train").values
    return C[tm], yl[tm], yo[tm], X[tm], gdf["fold"].values.astype(int)[tm]

def mae(yo, pred_log): return float(np.mean(np.abs(yo - np.exp(pred_log))))

def run_model(name, predict_fn, needs_coords, R_SEEDS):
    C, YL, YO, X, folds = load()
    cv = SpatialBlockBufferedCV(folds=folds, coords=C, buffer=BUFFER_M)
    buf = list(cv.split(X)); all_idx = np.arange(len(folds)); rows = []
    for fid, (tr_buf, te) in enumerate(buf):
        comp = all_idx[~np.isin(all_idx, te)]; n_buf = len(tr_buf)
        t0 = time.time()
        def predict(tr):
            if needs_coords: return predict_fn(X[tr], YL[tr], C[tr], X[te], C[te], 42)
            return predict_fn(X[tr], YL[tr], X[te], 42)
        m_buf = mae(YO[te], predict(tr_buf))
        rnd = []
        for s in R_SEEDS:
            rng = np.random.default_rng(s + fid); tr = rng.choice(comp, size=n_buf, replace=False)
            if needs_coords: p = predict_fn(X[tr], YL[tr], C[tr], X[te], C[te], s)
            else: p = predict_fn(X[tr], YL[tr], X[te], s)
            rnd.append(mae(YO[te], p))
        r = {"fold": fid, "n_test": len(te), "n_buf": n_buf,
             f"{name}_buf": round(m_buf, 2), f"{name}_rnd": round(float(np.mean(rnd)), 2),
             f"{name}_delta": round(m_buf - float(np.mean(rnd)), 2)}
        rows.append(r)
        print(f"[{name}] fold {fid}: n_buf={n_buf} buf={r[name+'_buf']} rnd={r[name+'_rnd']} "
              f"Δ={r[name+'_delta']} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUTDIR / f"o4_control_muestral_{name}.csv", index=False)
    b, rr = df[f"{name}_buf"].median(), df[f"{name}_rnd"].median()
    print(f"=== {name}: MAE_buffered(med)={b:.1f}  MAE_aleatorio(med)={rr:.1f}  Δ={b-rr:+.1f}")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp")
    ap.add_argument("--r", type=int, default=3)
    a = ap.parse_args()
    R = [42, 2011, 456, 777, 2026][:a.r]
    print(f"device={DEVICE}  models={a.models}  R={R}")
    for mdl in a.models.split(","):
        mdl = mdl.strip()
        if mdl == "mlp":
            run_model("mlp", mlp_predict_log, False, R)
        elif mdl == "gnnwr":
            run_model("gnnwr", gnnwr_predict_log, True, R)
        else:
            print("modelo desconocido:", mdl)
