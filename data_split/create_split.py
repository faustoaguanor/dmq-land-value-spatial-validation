"""
create_split.py
===============
Crea la particion holdout 80/20 para todos los modelos de la tesis.

Estrategia
----------
  Estratificada espacialmente: se divide el DMQ en 10 zonas usando KMeans
  sobre las coordenadas UTM, luego se muestrea 20% de cada zona al azar
  (seed=42). Esto garantiza que el test set cubra toda el area geográfica.

Salida
------
  split.csv  — columnas: idx (posicion en el dataset), predio_join, split (train|test)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.cluster import KMeans

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "datos" / "dataset.gpkg"
OUT_PATH  = BASE_DIR / "split.csv"

N_STRATA   = 10
TEST_FRAC  = 0.20
RANDOM_STATE = 42

def main():
    gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado")
    gdf = gdf.to_crs(epsg=32717)
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    n = len(gdf)
    print(f"Dataset: {n} observaciones")

    # Estratificar por zona espacial (KMeans)
    km = KMeans(n_clusters=N_STRATA, random_state=RANDOM_STATE, n_init=10)
    zona = km.fit_predict(coords)

    rng = np.random.default_rng(RANDOM_STATE)
    test_mask = np.zeros(n, dtype=bool)

    for z in range(N_STRATA):
        idx_z = np.where(zona == z)[0]
        n_test_z = max(1, round(len(idx_z) * TEST_FRAC))
        chosen = rng.choice(idx_z, size=n_test_z, replace=False)
        test_mask[chosen] = True

    n_test  = test_mask.sum()
    n_train = n - n_test
    print(f"Train: {n_train} ({100*n_train/n:.1f}%)  |  Test: {n_test} ({100*n_test/n:.1f}%)")

    split_df = pd.DataFrame({
        "idx":        np.arange(n),
        "predio_join": gdf["predio_join"].values,
        "zona":       zona,
        "split":      np.where(test_mask, "test", "train"),
    })
    split_df.to_csv(OUT_PATH, index=False)
    print(f"[CSV] {OUT_PATH}")

    # Verificar cobertura espacial por zona
    print("\nCobertura por zona:")
    for z in range(N_STRATA):
        idx_z = np.where(zona == z)[0]
        n_test_z = test_mask[idx_z].sum()
        print(f"  zona {z:2d}: n={len(idx_z):4d}  test={n_test_z:3d} ({100*n_test_z/len(idx_z):.0f}%)")

if __name__ == "__main__":
    main()
