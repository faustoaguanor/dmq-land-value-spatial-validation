"""
gwr_buffered_robusto.py  [SCRIPT HISTORICO — demostracion del ARTEFACTO de especificacion]
=========================================================================================
IMPORTANTE (2026-06-20): este script usa INTENCIONALMENTE la especificacion VIEJA del
GWR (lambda=0.1 con intercepto PENALIZADO) para reproducir el artefacto de la media de
fold-MAE = 6,185,047 USD/m2. NO representa el modelo GWR defendido en la tesis.

El modelo principal usa la especificacion canonica (intercepto NO penalizado, lambda por
CV espacial anidado) implementada en `modelos/gwr/gwr_core.py`, con la cual GWR NO diverge
(no hay artefacto 6.2M). El analisis de sensibilidad CANONICO al intercepto/lambda es
`analisis/verif_gwr_intercepto.py` (compara ambas especificaciones lado a lado).

Se conserva este script solo como evidencia reproducible de que el "colapso" reportado en
versiones previas era un artefacto de penalizar el intercepto, no una propiedad estructural
de GWR. No debe usarse para validar el modelo vigente.

Salida: analisis/output_log/gwr_buffered_robusto.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "spatial_cv"))
sys.path.insert(0, str(BASE / "modelos"))
from estrategias_cv import SpatialBlockBufferedCV
from features import build_feature_matrix

try:
    from mgwr.sel_bw import Sel_BW
    MGWR = True
except ImportError:
    MGWR = False

warnings.filterwarnings("ignore")

LAMBDA_RIDGE = 0.1
BW_MIN, BW_MAX, BW_FALLBACK = 40, 500, 199
BUFFER_M = 2530
SEED = 42

DATA  = BASE / "datos" / "dataset.gpkg"
SPLIT = BASE / "data_split" / "split.csv"
FOLDS = BASE / "spatial_cv" / "output" / "fold_assignments.csv"
OUT   = BASE / "analisis" / "output_log" / "gwr_buffered_robusto.csv"


def add_int(X): return np.column_stack([np.ones(len(X)), X])


def select_bw(coords_tr, X_int, y_tr):
    if not MGWR:
        return BW_FALLBACK
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = Sel_BW(coords_tr, y_tr.reshape(-1, 1), X_int, fixed=False, kernel="bisquare")
            bw = sel.search(bw_min=BW_MIN, bw_max=BW_MAX,
                            search_method="golden_section", criterion="AICc")
        return max(BW_MIN, int(round(float(bw))))
    except Exception:
        return BW_FALLBACK


def predict_gwr(coords_tr, X_tr_int, y_tr, coords_te, X_te_int, bw, lam=LAMBDA_RIDGE):
    p = X_tr_int.shape[1]; I_p = np.eye(p)
    k_q = min(bw, len(coords_tr))
    nbrs = NearestNeighbors(n_neighbors=k_q, algorithm="ball_tree").fit(coords_tr)
    h_te = nbrs.kneighbors(coords_te)[0][:, -1]
    preds = np.zeros(len(coords_te))
    for j in range(len(coords_te)):
        d = np.sqrt(((coords_tr - coords_te[j]) ** 2).sum(axis=1))
        u = d / (h_te[j] + 1e-10)
        w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        Xw = X_tr_int * w[:, None]
        A = Xw.T @ X_tr_int + lam * I_p
        b = X_tr_int.T @ (w * y_tr)
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.solve(Xw.T @ X_tr_int + (lam * 10) * I_p, b)
        preds[j] = X_te_int[j] @ beta
    return preds


# ── Datos ──────────────────────────────────────────────────────────────────────
gdf = gpd.read_file(DATA, layer="puntos_mercado").to_crs(epsg=32717)
split_df = pd.read_csv(SPLIT); folds_df = pd.read_csv(FOLDS)
for df in (gdf, split_df, folds_df):
    df["predio_join"] = df["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join", "fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join", "split"]], on="predio_join", how="left")
gdf = gdf.sort_values("predio_join").reset_index(drop=True)

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_orig = gdf["valor_m2"].values.astype(float)
y_log  = np.log(y_orig)
X_oh,  _ = build_feature_matrix(gdf)               # one-hot -> prediccion
X_cont, _ = build_feature_matrix(gdf, one_hot=False)  # continuo -> seleccion BW
train_mask = (gdf["split"] == "train").values
sp_folds = gdf["fold"].values.astype(int)

cv = SpatialBlockBufferedCV(folds=sp_folds[train_mask],
                            coords=coords[train_mask], buffer=BUFFER_M)
splits = list(cv.split(X_oh[train_mask]))

# ── CV buffered, capturando predicciones por punto ───────────────────────────────
rows = []          # per-fold
pool = []          # per-point (todos los folds)
for fid, (tr, te) in enumerate(splits):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_oh[train_mask][tr]);  Xte = sc.transform(X_oh[train_mask][te])
    ytr_log = y_log[train_mask][tr];  yte_log = y_log[train_mask][te]
    yte_ori = y_orig[train_mask][te]
    ctr = coords[train_mask][tr];  cte = coords[train_mask][te]

    Xtr_bw = add_int(StandardScaler().fit_transform(X_cont[train_mask][tr]))
    bw = select_bw(ctr, Xtr_bw, ytr_log)
    pred_log = predict_gwr(ctr, add_int(Xtr), ytr_log, cte, add_int(Xte), bw)

    # --- naive (sin acotar) ---
    pred_naive = np.exp(pred_log)
    # --- salvaguarda: clip en escala log al rango observado de y_train ---
    lo, hi = ytr_log.min(), ytr_log.max()
    pred_clip = np.exp(np.clip(pred_log, lo, hi))

    e_naive = yte_ori - pred_naive
    e_clip  = yte_ori - pred_clip
    e_log   = yte_log - pred_log
    rows.append({
        "fold": fid, "n_train": len(tr), "n_test": len(te), "bw": bw,
        "MAE_naive": np.mean(np.abs(e_naive)),
        "MAE_clip":  np.mean(np.abs(e_clip)),
        "MAE_log":   np.mean(np.abs(e_log)),
        "n_pred_fuera_rango": int(((pred_log < lo) | (pred_log > hi)).sum()),
    })
    for k in range(len(te)):
        pool.append({"fold": fid, "y_ori": yte_ori[k],
                     "pred_naive": pred_naive[k], "pred_clip": pred_clip[k],
                     "y_log": yte_log[k], "pred_log": pred_log[k]})

fold_df = pd.DataFrame(rows)
pool_df = pd.DataFrame(pool)

# ── Metricas agregadas ───────────────────────────────────────────────────────────
def med_abs(a): return float(np.median(np.abs(a)))

ae_naive = pool_df["y_ori"] - pool_df["pred_naive"]
ae_clip  = pool_df["y_ori"] - pool_df["pred_clip"]
ae_log   = pool_df["y_log"] - pool_df["pred_log"]

summary = {
    # O2: lo que hoy se reporta (media de fold-MAE) vs robusto
    "MAE_media_folds_naive":   float(fold_df["MAE_naive"].mean()),     # ~6.2M (artefacto)
    "MAE_mediana_folds_naive": float(fold_df["MAE_naive"].median()),   # robusto a fold outlier
    "MedAE_pooled_naive_USD":  med_abs(ae_naive),                      # robusto a punto outlier
    "MAE_log_pooled":          float(np.mean(np.abs(ae_log))),         # escala log (sin exp)
    "MedAE_log_pooled":        med_abs(ae_log),
    # O3: efecto de la salvaguarda (clip al rango de y_train)
    "MAE_media_folds_clip":    float(fold_df["MAE_clip"].mean()),
    "MAE_mediana_folds_clip":  float(fold_df["MAE_clip"].median()),
    "MedAE_pooled_clip_USD":   med_abs(ae_clip),
    "n_pred_fuera_rango_total": int(fold_df["n_pred_fuera_rango"].sum()),
    "n_test_total":            int(fold_df["n_test"].sum()),
}

OUT.parent.mkdir(exist_ok=True)
fold_df.round(4).to_csv(OUT, index=False)
with open(OUT.with_name("gwr_buffered_robusto_summary.csv"), "w") as f:
    pd.DataFrame([summary]).round(4).T.rename(columns={0: "valor"}).to_csv(f)

print("=== PER-FOLD ===")
print(fold_df.round(2).to_string(index=False))
print("\n=== RESUMEN ===")
for k, v in summary.items():
    print(f"  {k:28s} = {v:,.4f}" if isinstance(v, float) else f"  {k:28s} = {v}")
print(f"\n[CSV] {OUT}")
