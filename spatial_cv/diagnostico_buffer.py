"""
diagnostico_buffer.py
=====================
Justifica empiricamente el tamano del BUFFER de la validacion espacial rigurosa
(SpatialBlockBufferedCV). Evidencia para la tesis (cierra el cargo C1).

Calcula:
  (1) Rango de autocorrelacion de los RESIDUOS OLS (serie estacionaria) — el
      rango relevante para el leakage, NO el del target crudo (5.62 km, inflado
      por la tendencia de precio de la ciudad).
  (2) Para varios buffers: retencion de training y separacion minima real
      train-test por fold del checkerboard.

Salida: spatial_cv/output/diagnostico_buffer.csv

Resultado (DMQ): rango residual exponencial ~2530 m. buffer=2530 m garantiza
separacion >= 2.5 km reteniendo 59-80% del training (viable). buffer=5622 m
(rango del target crudo) inaniza el training (folds con <600 obs) -> inviable.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "spatial_cv" / "output"; OUT.mkdir(exist_ok=True)
COV = ["suscept_codigo","pc_pnbi","dist_metro","dist_centr_metro","dist_centr_zonal",
       "dist_cc","dist_universidad","dist_hospital","dist_parque_metro","dist_industrial",
       "dist_via_principal","uso_suelo_cod","cos_num","dist_quebrada","dist_mercado_mayorista",
       "dist_plataforma_gub","log_area","frente_m","area_const_m2","tiene_const","num_pisos",
       "antiguedad","topografia_factor","conservacion_cod","acabados_cod","es_ph","pendiente_grados"]
BUFFERS = [0, 2000, 2530, 4000, 5621.8]

gdf = gpd.read_file(ROOT/"datos/dataset.gpkg", layer="puntos_mercado").to_crs(32717)
folds = pd.read_csv(ROOT/"spatial_cv/output/fold_assignments.csv")
folds["predio_join"] = folds["predio_join"].astype(int); gdf["predio_join"] = gdf["predio_join"].astype(int)
gdf = gdf.merge(folds[["predio_join","fold"]], on="predio_join", how="left")
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y = np.log(gdf["valor_m2"].values.astype(float)); fold = gdf["fold"].values.astype(int)

# (1) rango residual
X = StandardScaler().fit_transform(gdf[COV].values.astype(float))
resid = y - LinearRegression().fit(X, y).predict(X)
print("== Rango de autocorrelacion de RESIDUOS OLS (serie estacionaria) ==")
try:
    import skgstat as skg
    for mdl in ["exponential","spherical","gaussian"]:
        V = skg.Variogram(coordinates=coords, values=resid, model=mdl, maxlag="median", n_lags=25)
        print(f"  {mdl:11s}: range={V.parameters[0]:8.1f} m  rmse={V.rmse:.4f}")
except Exception as e:
    print("  skgstat no disponible:", e)

# (2) retencion y separacion por buffer
rows = []
for buf in BUFFERS:
    for k in sorted(set(fold)):
        te = np.where(fold == k)[0]; tr_full = np.where(fold != k)[0]
        if buf > 0:
            d, _ = cKDTree(coords[te]).query(coords[tr_full], k=1)
            keep = tr_full[d > buf]
        else:
            keep = tr_full
        mind = float(cKDTree(coords[keep]).query(coords[te], k=1)[0].min()) if len(keep) else np.nan
        rows.append({"buffer_m": buf, "fold": k, "n_test": len(te),
                     "n_train_full": len(tr_full), "n_train_buf": len(keep),
                     "pct_retenido": round(100*len(keep)/len(tr_full), 1),
                     "min_dist_train_test_m": round(mind, 1)})
df = pd.DataFrame(rows)
df.to_csv(OUT/"diagnostico_buffer.csv", index=False)
print(f"\n[CSV] {OUT/'diagnostico_buffer.csv'}")
print(df.to_string(index=False))
