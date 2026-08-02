"""
features.py — Construccion UNICA de la matriz de features para todos los modelos.
================================================================================
`uso_suelo_cod` es la unica variable NOMINAL del dataset (categorias [1,2,3,5,6,10]
sin orden natural). Tratarla como continua es un error de especificacion (un codigo
arbitrario no implica que "uso 10" pese 10x "uso 1"). Aqui se codifica one-hot;
el resto de variables se mantienen (ordinales/binarias/conteo/continuas, defendibles
como continuas).

Todos los modelos (OLS, MLP, GWR-27, GNNWR, SANNWR, GSAWR) deben importar
`build_feature_matrix(gdf)` para garantizar EXACTAMENTE la misma matriz y el mismo
orden de columnas. Llamar SIEMPRE sobre el gdf completo y luego indexar por mascara
de train/test/fold, para que las columnas dummy sean consistentes.

Evidencia del impacto: ver analisis/sensibilidad_encoding.py (OLS holdout:
151.19 -> 144.72 RMSE, -6.5 USD/m2 a favor de one-hot).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Las 27 variables originales de la tesis (orden canonico).
COVARIABLES = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]

NOMINAL = "uso_suelo_cod"
# Categorias con < RARE_MIN_COUNT obs se colapsan en "otros". Necesario porque
# uso_suelo_cod tiene singletons (cat 5 = 1 obs, cat 10 = 2 obs, ambos en train):
# sus dummies serian todo-cero en test (memorizacion inutil) y ademas generan
# matrices locales singulares que rompen la seleccion de bandwidth de GWR (mgwr).
# Colapsar categorias raras es practica estandar de codificacion categorica.
RARE_MIN_COUNT = 30


def build_feature_matrix(gdf, one_hot: bool = True, rare_min_count: int = RARE_MIN_COUNT,
                         covariables=None):
    """Devuelve (X, feature_names).

    one_hot=True  -> `uso_suelo_cod` (nominal) reemplazada por dummies (drop_first),
                     con categorias raras (< rare_min_count) colapsadas en "otros".
                     Con las 27 vars del DMQ: categorias {1,2,3,6,otros} -> 4 dummies -> 30 cols.
    one_hot=False -> codificacion original, para reproducir runs viejos.
    covariables   -> lista de columnas a usar (default: las 27 de COVARIABLES). Permite
                     reusar la misma logica one-hot para subconjuntos, p.ej. GWR-17.
                     Si la lista no contiene `uso_suelo_cod`, no se generan dummies.

    IMPORTANTE: llamar sobre el gdf COMPLETO y luego indexar X por posicion/mascara.
    Conteos y categorias se fijan sobre todo el dataset (deterministas, sin fuga del
    target). La categoria dominante (cat 1, 85.5%) queda como referencia (drop_first).
    """
    cov = list(covariables) if covariables is not None else list(COVARIABLES)
    if not one_hot or NOMINAL not in cov:
        return gdf[cov].astype(float).values, list(cov)

    base = [c for c in cov if c != NOMINAL]
    Xnum = gdf[base].astype(float).reset_index(drop=True)

    cats = gdf[NOMINAL].astype(int)
    vc = cats.value_counts()
    rare = set(vc[vc < rare_min_count].index)            # DMQ: {5, 10}
    labels = cats.map(lambda c: "otros" if c in rare else f"{int(c):02d}")
    dummies = pd.get_dummies(labels, prefix="uso", drop_first=True).astype(float).reset_index(drop=True)
    X = pd.concat([Xnum, dummies], axis=1)
    return X.values.astype(float), list(X.columns)


if __name__ == "__main__":
    # Auto-test: consistencia de columnas y de indexado por mascara.
    from pathlib import Path
    import geopandas as gpd
    ROOT = Path(__file__).parent.parent
    gdf = gpd.read_file(ROOT / "datos" / "dataset.gpkg", layer="puntos_mercado")
    sp = pd.read_csv(ROOT / "data_split" / "split.csv")
    # Clave canonica int (como los pipelines de modelos): astype(str) pierde el
    # Evita perder claves catastrales con ceros iniciales al cruzar archivos.
    sp["predio_join"] = sp["predio_join"].astype(int)
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    gdf = gdf.merge(sp[["predio_join", "split"]], on="predio_join", how="left")
    assert gdf["split"].notna().all(), "predios sin split tras el merge"

    X27, n27 = build_feature_matrix(gdf, one_hot=False)
    X31, n31 = build_feature_matrix(gdf, one_hot=True)
    print(f"continuo : {X27.shape}  cols={len(n27)}")
    print(f"one-hot  : {X31.shape}  cols={len(n31)}")
    print(f"dummies anadidas: {[c for c in n31 if c.startswith('uso_')]}")
    # indexar por mascara debe preservar columnas y filas
    tr = (gdf['split'] == 'train').values
    assert X31[tr].shape[1] == X31.shape[1], "columnas inconsistentes al indexar"
    assert not np.isnan(X31).any(), "hay NaN en la matriz one-hot"
    print(f"train rows={tr.sum()}  sin NaN  OK  | n_features one-hot = {X31.shape[1]}")
