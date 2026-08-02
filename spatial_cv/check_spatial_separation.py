"""
check_spatial_separation.py
Calcula la distribucion de distancias minimas tren-test para los 3 esquemas
de validacion: Holdout 80/20, SpatialBlock K=5, RandomKFold K=5.
Genera check_spatial_separation.csv y check_spatial_separation.png.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

BASE   = Path(__file__).resolve().parent.parent
DATA   = BASE / "datos" / "dataset.gpkg"
SPLIT  = BASE / "data_split" / "split.csv"
FOLDS  = BASE / "spatial_cv" / "output" / "fold_assignments.csv"
OUT    = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

CRS_UTM   = "EPSG:32717"
RANGE_M   = 5621.8
K_FOLDS   = 5

# ── Cargar datos ──────────────────────────────────────────────────────────────
gdf = gpd.read_file(DATA).to_crs(CRS_UTM)
# Clave canonica: cast a int en ambos lados (misma convencion que los pipelines de
# modelos, p.ej. gwr_log_27vars.py / sannwr_real_log.py). astype(str) NO cruza el
# Convención canónica: normalizar las claves como enteros antes de cruzarlas.
gdf["predio_join"] = gdf["predio_join"].astype(int)
split_df = pd.read_csv(SPLIT); split_df["predio_join"] = split_df["predio_join"].astype(int)
folds_df = pd.read_csv(FOLDS); folds_df["predio_join"] = folds_df["predio_join"].astype(int)
gdf = gdf.merge(folds_df[["predio_join","fold"]], on="predio_join", how="left")
gdf = gdf.merge(split_df[["predio_join","split"]], on="predio_join", how="left")
if gdf["predio_join"].duplicated().any():
    raise ValueError("predio_join duplicado despues de los merges; las distancias train-test no son confiables.")
if gdf["fold"].isna().any() or gdf["split"].isna().any():
    raise ValueError("Hay predios sin fold o split asignado; revise fold_assignments.csv y split.csv.")
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

train_mask = (gdf["split"] == "train").values
test_mask  = (gdf["split"] == "test").values
sp_folds   = gdf["fold"].values.astype(int)

rows = []

def dist_stats(min_dists, label, fold_id=None):
    d = min_dists
    row = {
        "esquema": label,
        "fold": fold_id if fold_id is not None else "all",
        "n_test": len(d),
        "min_m":   round(d.min(), 1),
        "p5_m":    round(np.percentile(d, 5), 1),
        "p25_m":   round(np.percentile(d, 25), 1),
        "median_m":round(np.median(d), 1),
        "p75_m":   round(np.percentile(d, 75), 1),
        "max_m":   round(d.max(), 1),
        "pct_below_range": round(100 * (d < RANGE_M).mean(), 1),
        "pct_below_500m":  round(100 * (d < 500).mean(), 1),
    }
    rows.append(row)
    print(f"  {label:<30} fold={str(fold_id):<4}  n={len(d):>5}  "
          f"min={d.min():>7.0f}m  p5={np.percentile(d,5):>7.0f}m  "
          f"median={np.median(d):>7.0f}m  "
          f"<rango={row['pct_below_range']:>5.1f}%  <500m={row['pct_below_500m']:>5.1f}%")

print("=" * 90)
print("DISTANCIA MÍNIMA TREN → TEST POR ESQUEMA DE VALIDACIÓN")
print(f"Rango del semivariograma: {RANGE_M} m")
print("=" * 90)

# 1. HOLDOUT
print("\n[1] Holdout 80/20")
tree_tr = cKDTree(coords[train_mask])
d_ho, _ = tree_tr.query(coords[test_mask], k=1)
dist_stats(d_ho, "Holdout", fold_id=None)

# 2. SPATIALBLOCK K=5
print("\n[2] SpatialBlock K=5")
c_tr = coords[train_mask]
sf   = sp_folds[train_mask]
all_sb = []
for f in range(K_FOLDS):
    te_idx = np.where(sf == f)[0]
    tr_idx = np.where(sf != f)[0]
    tree = cKDTree(c_tr[tr_idx])
    d, _ = tree.query(c_tr[te_idx], k=1)
    dist_stats(d, "SpatialBlock", fold_id=f)
    all_sb.append(d)
dist_stats(np.concatenate(all_sb), "SpatialBlock", fold_id="ALL")

# 3. RANDOMKFOLD K=5
print("\n[3] RandomKFold K=5")
rng = np.random.default_rng(42)
idx = np.arange(len(c_tr))
rng.shuffle(idx)
fold_labels = np.zeros(len(c_tr), dtype=int)
for i, j in enumerate(idx):
    fold_labels[j] = i % K_FOLDS
all_rk = []
for f in range(K_FOLDS):
    te_idx = np.where(fold_labels == f)[0]
    tr_idx = np.where(fold_labels != f)[0]
    tree = cKDTree(c_tr[tr_idx])
    d, _ = tree.query(c_tr[te_idx], k=1)
    dist_stats(d, "RandomKFold", fold_id=f)
    all_rk.append(d)
dist_stats(np.concatenate(all_rk), "RandomKFold", fold_id="ALL")

# ── CSV ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUT / "check_spatial_separation.csv", index=False)
print(f"\n[CSV] {OUT / 'check_spatial_separation.csv'}")

# ── Figura ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

datasets = [
    ("Holdout 80/20", d_ho),
    ("SpatialBlock (todos los folds)", np.concatenate(all_sb)),
    ("RandomKFold (todos los folds)", np.concatenate(all_rk)),
]

for ax, (title, d) in zip(axes, datasets):
    ax.hist(d / 1000, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(RANGE_M / 1000, color="red", lw=1.5, ls="--",
               label=f"Rango variograma ({RANGE_M/1000:.2f} km)")
    pct = 100 * (d < RANGE_M).mean()
    ax.set_title(f"{title}\n{pct:.1f}% puntos dentro del rango", fontsize=10)
    ax.set_xlabel("Distancia mínima al train más cercano (km)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8)

plt.suptitle("Separación espacial tren-test por esquema de validación\n"
             f"DMQ — {len(gdf):,} predios — EPSG:32717", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "check_spatial_separation.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "check_spatial_separation.pdf", bbox_inches="tight")
print(f"[Fig] {OUT / 'check_spatial_separation.png'}")
print("\nInterpretación:")
print(f"  • Puntos con dist < {RANGE_M:.0f}m (rango) tienen vecinos tren en zona de autocorrelación")
print(f"  • Puntos con dist < 500m tienen vecinos tren casi colindantes")
