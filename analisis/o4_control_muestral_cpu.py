"""
o4_control_muestral_cpu.py
==========================
O4: ¿la degradación bajo SpatialBlock_buf se debe a la SEPARACIÓN espacial o a la
REDUCCIÓN del tamaño de entrenamiento (el buffer retira 20-41% del train)?

Control: por cada fold buffered, se compara — sobre el MISMO test fold y con el MISMO
bandwidth — el entrenamiento BUFFERED (separación > 2,530 m) contra un entrenamiento
ALEATORIO del mismo tamaño extraído del complemento sin criterio espacial (R seeds).

  ΔMAE = MAE_buffered − MAE_aleatorio_mismo_n
    ΔMAE > 0 (y grande)  -> la degradación es por SEPARACIÓN espacial (hallazgo central OK)
    ΔMAE ≈ 0             -> la degradación es por TAMAÑO muestral (reformular)

Cubre los modelos CPU/deterministas: OLS y GWR-27. Los neurales (MLP/GNNWR/SANNWR)
van en el bundle GPU (o4_control_muestral_gpu.py).

Salida: analisis/output_log/o4_control_muestral_cpu.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "spatial_cv"))
sys.path.insert(0, str(BASE / "modelos"))
sys.path.insert(0, str(BASE / "modelos" / "gwr"))
from estrategias_cv import SpatialBlockBufferedCV
from features import build_feature_matrix
from gwr_core import select_bw, select_lambda, predict_gwr, add_intercept as add_int  # GWR canonico
warnings.filterwarnings("ignore")

BUFFER_M = 2530
R_SEEDS = [42, 2011, 456, 777, 2026]   # repeticiones del control aleatorio

DATA  = BASE / "datos" / "dataset.gpkg"
SPLIT = BASE / "data_split" / "split.csv"
FOLDS = BASE / "spatial_cv" / "output" / "fold_assignments.csv"
OUT   = BASE / "analisis" / "output_log" / "o4_control_muestral_cpu.csv"


def mae(y_true_ori, pred_log):
    return float(np.mean(np.abs(y_true_ori - np.exp(pred_log))))


# ── Datos ──────────────────────────────────────────────────────────────────────
gdf = gpd.read_file(DATA, layer="puntos_mercado").to_crs(epsg=32717)
sp = pd.read_csv(SPLIT); fo = pd.read_csv(FOLDS)
for df in (gdf, sp, fo): df["predio_join"] = df["predio_join"].astype(int)
gdf = gdf.merge(fo[["predio_join", "fold"]], on="predio_join", how="left")
gdf = gdf.merge(sp[["predio_join", "split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_log = np.log(gdf["valor_m2"].values.astype(float))
y_ori = gdf["valor_m2"].values.astype(float)
X_oh, _ = build_feature_matrix(gdf)
X_co, _ = build_feature_matrix(gdf, one_hot=False)
tm = (gdf["split"] == "train").values
folds_tr = gdf["fold"].values.astype(int)[tm]

# datos restringidos al 80% train (donde vive la CV)
C = coords[tm]; YL = y_log[tm]; YO = y_ori[tm]; XOH = X_oh[tm]; XCO = X_co[tm]
cv = SpatialBlockBufferedCV(folds=folds_tr, coords=C, buffer=BUFFER_M)
buf_splits = list(cv.split(XOH))           # (train_buf, test) por fold
all_idx = np.arange(len(folds_tr))

rows = []
for fid, (tr_buf, te) in enumerate(buf_splits):
    complement = all_idx[folds_tr != folds_tr[te][0]] if False else all_idx[~np.isin(all_idx, te)]
    n_buf = len(tr_buf)
    cte = C[te]; yte_ori = YO[te]
    # BW seleccionado sobre el train buffered (igual que la tesis), reutilizado en ambos brazos
    bw = select_bw(C[tr_buf], add_int(StandardScaler().fit_transform(XCO[tr_buf])), YL[tr_buf])

    def fit_predict(tr_idx):
        sc = StandardScaler()
        Xtr = add_int(sc.fit_transform(XOH[tr_idx])); Xte = add_int(sc.transform(XOH[te]))
        ytr = YL[tr_idx]
        # OLS
        ols = LinearRegression(fit_intercept=False).fit(Xtr, ytr)
        pred_ols = ols.predict(Xte)
        # GWR canonico (gwr_core): lambda por CV espacial anidado DENTRO de cada brazo
        # (reoptimiza el hiperparametro por brazo, hallazgo #14); intercepto no penalizado.
        lam, _ = select_lambda(C[tr_idx], Xtr, ytr, bw)
        pred_gwr, _ = predict_gwr(C[tr_idx], Xtr, ytr, cte, Xte, bw, lam=lam)
        return {"OLS": mae(yte_ori, pred_ols), "GWR": mae(yte_ori, pred_gwr)}

    m_buf = fit_predict(tr_buf)
    # control aleatorio: R draws del complemento, mismo tamaño n_buf
    rnd = {k: [] for k in ("OLS", "GWR")}
    for s in R_SEEDS:
        rng = np.random.default_rng(s + fid)
        tr_rnd = rng.choice(complement, size=n_buf, replace=False)
        mr = fit_predict(tr_rnd)
        for k in rnd: rnd[k].append(mr[k])
    row = {"fold": fid, "n_test": len(te), "n_buf": n_buf, "n_comp": len(complement), "bw": bw}
    for k in ("OLS", "GWR"):
        row[f"{k}_buf"] = round(m_buf[k], 2)
        row[f"{k}_rnd"] = round(float(np.mean(rnd[k])), 2)
        row[f"{k}_delta"] = round(m_buf[k] - float(np.mean(rnd[k])), 2)
    rows.append(row)
    print(f"fold {fid}: n_buf={n_buf} bw={bw} | "
          f"OLS buf={row['OLS_buf']} rnd={row['OLS_rnd']} Δ={row['OLS_delta']} | "
          f"GWR buf={row['GWR_buf']} rnd={row['GWR_rnd']} Δ={row['GWR_delta']}", flush=True)

df = pd.DataFrame(rows)
OUT.parent.mkdir(exist_ok=True)
df.to_csv(OUT, index=False)

print("\n=== RESUMEN (mediana de folds, robusta) ===")
for k in ("OLS", "GWR"):
    b = df[f"{k}_buf"].median(); r = df[f"{k}_rnd"].median()
    print(f"  {k:9s}  MAE_buffered(med)={b:8.1f}  MAE_aleatorio(med)={r:8.1f}  Δ={b-r:+7.1f}")
print(f"\n[CSV] {OUT}")
