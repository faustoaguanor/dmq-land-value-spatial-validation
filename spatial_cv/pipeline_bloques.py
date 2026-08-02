"""
pipeline_bloques.py
===================
Pasos 1 y 2: Semivariograma empirico y bloques espaciales.
Roberts et al. (2017)

Referencia: "Cross-validation strategies for data with temporal, spatial,
hierarchical, or phylogenetic structure" (Roberts et al., 2017, Ecography)

Equivalencia con el codigo R de referencia (Box 1 y Box 4):
  - Paso 1: autofitVariogram(resids ~ 1, ..., model="Sph")  ->  skgstat.Variogram
  - Paso 2: grid.fun() + asignacion sistematica checkerboard

Datos   : ../datos/dataset.gpkg, capa 'puntos_mercado', CRS EPSG:4326
Salidas : output/semivariogram.png, output/spatial_folds_map.png,
          output/fold_assignments.csv

Ejecutar desde spatial_cv/:
    .venv/Scripts/python pipeline_bloques.py
"""

import os
import numpy as np
import geopandas as gpd
import skgstat as skg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # spatial_cv/
DATA_PATH  = BASE_DIR / ".." / "datos" / "dataset.gpkg"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Parámetros ────────────────────────────────────────────────────────────────
LAYER   = "puntos_mercado"
TARGET  = "valor_m2"
K_FOLDS = 5
CRS_UTM = 32717          # UTM zona 17S — sistema métrico para Ecuador/Quito
N_LAGS  = 25
MODELS  = ["spherical", "exponential", "gaussian"]   # se elige el de menor RMSE


# ═════════════════════════════════════════════════════════════════════════════
# PASO 1: Semivariograma empírico
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PASO 1: Semivariograma empírico")
print("=" * 60)

# 1.1 Cargar GeoPackage y reproyectar a metros (UTM 17S)
print(f"\n[1] Cargando {DATA_PATH.resolve()} ...")
gdf = gpd.read_file(DATA_PATH, layer=LAYER)
print(f"    CRS original : {gdf.crs}")
gdf = gdf.to_crs(epsg=CRS_UTM)
print(f"    CRS reproyect: EPSG:{CRS_UTM} (UTM 17S, metros)")
print(f"    Observaciones: {len(gdf):,}")

# 1.2 Extraer coordenadas y variable objetivo
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
values = gdf[TARGET].values
print(f"\n[2] Variable objetivo '{TARGET}':")
print(f"    min={values.min():.2f}  max={values.max():.2f}  "
      f"media={values.mean():.2f}  std={values.std():.2f}")

# 1.3 Ajustar semivariograma con varios modelos y elegir el de menor RMSE
# Equivalente R: autofitVariogram(resids ~ 1, resid.spatial, model=c("Sph"))
print(f"\n[3] Ajustando semivariograma (n_lags={N_LAGS}, maxlag='median') ...")
best_model, best_rmse, best_V = None, np.inf, None

for model in MODELS:
    try:
        V = skg.Variogram(
            coordinates=coords,
            values=values,
            model=model,
            maxlag="median",
            n_lags=N_LAGS,
        )
        rmse = V.rmse
        print(f"    {model:12s}  RMSE={rmse:.4f}  "
              f"range={V.parameters[0]:.1f} m  sill={V.parameters[1]:.4f}")
        if rmse < best_rmse:
            best_rmse, best_model, best_V = rmse, model, V
    except Exception as e:
        print(f"    {model:12s}  ERROR: {e}")

V          = best_V
range_m    = V.parameters[0]    # rango de autocorrelación en metros
sill       = V.parameters[1]
nugget     = V.parameters[2] if len(V.parameters) > 2 else 0.0

print(f"\n>>> Modelo seleccionado : {best_model}  (RMSE={best_rmse:.4f})")
print(f"    Rango              : {range_m:.1f} m  ({range_m/1000:.2f} km)")
print(f"    Sill               : {sill:.4f}")
print(f"    Nugget             : {nugget:.4f}")

# 1.4 Graficar y guardar
fig, ax = plt.subplots(figsize=(8, 5))
V.plot(axes=ax, hist=False)
ax.set_title(
    f"Semivariograma empírico — {TARGET}\n"
    f"Modelo: {best_model}  |  Rango = {range_m:.0f} m ({range_m/1000:.2f} km)",
    fontsize=11
)
ax.set_xlabel("Distancia (m)")
ax.set_ylabel("Semivarianza")
ax.axvline(range_m, color="red", linestyle="--", linewidth=1.2,
           label=f"Rango = {range_m:.0f} m")
ax.legend()
plt.tight_layout()
out_semivar = OUTPUT_DIR / "semivariogram.png"
fig.savefig(out_semivar, dpi=150)
plt.close(fig)
print(f"\n    Gráfico guardado: {out_semivar}")


# ═════════════════════════════════════════════════════════════════════════════
# PASO 2: Grilla espacial y asignación de K-Folds (checkerboard)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PASO 2: Bloques espaciales y asignación de K-Folds")
print("=" * 60)

# 2.1 Crear grilla con tamaño de celda = rango del semivariograma
# Equivalente R: grid.fun() de Box 4
xmin, ymin, xmax, ymax = gdf.total_bounds
block_size = range_m

n_cols = int(np.ceil((xmax - xmin) / block_size))
n_rows = int(np.ceil((ymax - ymin) / block_size))
n_blocks = n_cols * n_rows

print(f"\n[4] Parámetros de la grilla:")
print(f"    block_size = {block_size:.1f} m ({block_size/1000:.2f} km)  "
      f"[= rango del semivariograma]")
print(f"    Extensión  : {(xmax-xmin)/1000:.1f} km (E-O) × "
      f"{(ymax-ymin)/1000:.1f} km (N-S)")
print(f"    Grilla     : {n_cols} col × {n_rows} filas = {n_blocks} bloques")

# 2.2 Asignar cada punto a su celda de la grilla
gdf["block_col"] = ((gdf.geometry.x - xmin) / block_size).astype(int)
gdf["block_row"] = ((gdf.geometry.y - ymin) / block_size).astype(int)
gdf["block_id"]  = (gdf["block_col"].astype(str) + "_" +
                    gdf["block_row"].astype(str))

n_occupied = gdf["block_id"].nunique()
print(f"    Bloques ocupados: {n_occupied} de {n_blocks}")

# 2.3 Asignación sistemática checkerboard a K=5 folds
# Equivalente R extendido a K folds: fold = (block_col + block_row) % K
# Roberts et al. (2017): asignación sistemática evita aislar regiones extremas
gdf["fold"] = (gdf["block_col"] + gdf["block_row"]) % K_FOLDS

print(f"\n[5] Distribución de puntos por fold (K={K_FOLDS}):")
fold_counts = gdf["fold"].value_counts().sort_index()
for fold, count in fold_counts.items():
    bar = "#" * (count // 50)
    print(f"    Fold {fold}: {count:5d} obs  {bar}")

# 2.4 Mapa de folds
fig, ax = plt.subplots(figsize=(10, 10))
colors = plt.cm.Set1(np.linspace(0, 1, K_FOLDS))

for k in range(K_FOLDS):
    mask = gdf["fold"] == k
    gdf[mask].plot(ax=ax, color=colors[k], markersize=2, alpha=0.7,
                   label=f"Fold {k} (n={mask.sum():,})")

ax.set_title(
    f"Spatial Block CV — {K_FOLDS} Folds (Roberts et al., 2017)\n"
    f"Tamaño de bloque: {block_size:.0f} m ({block_size/1000:.2f} km) "
    f"= rango autocorrelación",
    fontsize=11
)
ax.set_xlabel("Este (m, UTM 17S)")
ax.set_ylabel("Norte (m, UTM 17S)")
ax.legend(loc="upper right", markerscale=4, framealpha=0.8)
plt.tight_layout()
out_map = OUTPUT_DIR / "spatial_folds_map.png"
fig.savefig(out_map, dpi=150)
plt.close(fig)
print(f"\n    Mapa guardado: {out_map}")

# 2.5 Exportar asignaciones a CSV
out_csv = OUTPUT_DIR / "fold_assignments.csv"
gdf[["predio_join", "block_id", "block_col", "block_row", "fold"]].to_csv(
    out_csv, index=False
)
print(f"    CSV guardado : {out_csv}  ({len(gdf):,} filas)")

# ── Resumen final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"  Observaciones totales : {len(gdf):,}")
print(f"  Modelo variograma     : {best_model}")
print(f"  Rango autocorrelación : {range_m:.1f} m ({range_m/1000:.2f} km)")
print(f"  Tamaño de bloque      : {block_size:.1f} m ({block_size/1000:.2f} km)")
print(f"  Bloques ocupados      : {n_occupied}")
print(f"  K-Folds               : {K_FOLDS}")
print(f"  Patrón asignación     : checkerboard sistemático "
      f"[(col+fila) % {K_FOLDS}]")
print(f"\n  Salidas:")
print(f"    {out_semivar}")
print(f"    {out_map}")
print(f"    {out_csv}")
print("=" * 60)
