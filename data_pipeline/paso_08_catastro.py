"""
paso_08_catastro.py
====================
Join con Catastro Municipal y extracción de variables físicas del predio.

INPUT:  config.INTERMEDIOS["paso07"]    → 5,051 puntos con 18 vars espaciales
        config.FUENTES["catastro_csv"]  → Predio_variables.csv (1,053,655 predios)
OUTPUT: config.INTERMEDIOS["paso08"]   → +10 variables de catastro
"""
from __future__ import annotations
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

VARS_CATASTRO = [
    "log_area", "frente_m", "area_const_m2", "tiene_const",
    "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph",
]


def run(cfg=None) -> gpd.GeoDataFrame:
    if cfg is None:
        import config as cfg

    LOG: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        LOG.append(str(msg))

    # ── 8.1 Cargar puntos ────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 8.1: Cargando dataset espacial")
    log("═" * 60)

    puntos = gpd.read_file(str(cfg.INTERMEDIOS["paso07"]), layer=cfg.LAYER_PUNTOS)
    n_puntos = len(puntos)
    log(f"  Puntos: {n_puntos}  |  Variables espaciales: {len(puntos.columns) - 1}")

    # ── 8.2 Cargar catastro ──────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 8.2: Cargando Catastro Municipal (Predio_variables.csv)")
    log("═" * 60)

    cat = pd.read_csv(
        str(cfg.FUENTES["catastro_csv"]),
        sep=";", decimal=",", encoding="latin1", low_memory=False,
    )
    log(f"  Registros catastro: {len(cat):,}  |  Columnas: {len(cat.columns)}")

    if "NUMERO_PREDIO" not in cat.columns:
        raise ValueError(f"NUMERO_PREDIO no encontrado. Columnas: {list(cat.columns[:10])}")
    cat["predio_join"] = cat["NUMERO_PREDIO"].astype(str)

    # ── 8.3 Join ─────────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 8.3: Join tabular por predio_join")
    log("═" * 60)

    puntos["predio_join"] = puntos["predio_join"].astype(str)
    result = puntos.merge(cat, on="predio_join", how="left")
    result = result.drop_duplicates(subset="predio_join", keep="first")

    n_con = result["NUMERO_PREDIO"].notna().sum()
    n_sin = result["NUMERO_PREDIO"].isna().sum()
    log(f"  Match: {n_con} ({n_con/len(result)*100:.1f}%)  |  Sin match: {n_sin}")

    # ── 8.4 Crear variables procesadas ───────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 8.4: Creando variables procesadas del catastro")
    log("═" * 60)

    result["log_area"]          = np.log1p(result["AREA_TERREN_PROPOR_ALICUOTA"])
    result["frente_m"]          = result["FRENTE_PREDIO"]
    result["area_const_m2"]     = result["AREA_CONSTRUCCION"]
    result["tiene_const"]       = (result["AREA_CONSTRUCCION"] > 0).astype(int)
    result["num_pisos"]         = result["NUMERO_PISOS"].fillna(0).astype(int)

    anios = result["ANIO_CONSTRUCCION"].copy()
    anios[anios < 1900] = np.nan
    result["antiguedad"]        = (cfg.ANIO_REFERENCIA - anios).fillna(0).clip(lower=0)

    result["topografia_factor"] = result["FACTOR_TOPOGRAFIA"].fillna(1.0)
    result["conservacion_cod"]  = result["ESTADO_CONSERVACION"].str.upper().map(
        cfg.MAPA_CONSERVACION).fillna(0).astype(int)
    result["acabados_cod"]      = result["TIPO_ACABADOS"].str.upper().map(
        cfg.MAPA_ACABADOS).fillna(0).astype(int)
    result["es_ph"]             = (result["PROPIEDAD"] == "HOR").astype(int).fillna(0)

    # Imputar nulos con mediana para variables numéricas continuas
    for var in ("log_area", "frente_m", "area_const_m2", "topografia_factor"):
        n_nulos = result[var].isna().sum()
        if n_nulos > 0:
            mediana = result[var].median()
            result[var] = result[var].fillna(mediana)
            log(f"  {var}: {n_nulos} nulos imputados con mediana={mediana:.2f}")

    result["tiene_const"]  = result["tiene_const"].fillna(0).astype(int)

    # ── 8.5 Validar ──────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 8.5: Validando variables de catastro")
    log("═" * 60)

    for var in VARS_CATASTRO:
        n_nulos = result[var].isna().sum()
        if n_nulos > 0:
            log(f"  ⚠️ {var}: {n_nulos} nulos")
        else:
            log(f"  ✓ {var}")

    # Eliminar columnas crudas del catastro
    cols_drop = [c for c in result.columns if c in cat.columns and c != "predio_join"]
    result = result.drop(columns=cols_drop)
    log(f"\n  Columnas totales: {len(result.columns)}")

    # ── 8.6 Exportar ─────────────────────────────────────────────────────────
    out = cfg.INTERMEDIOS["paso08"]
    result.to_file(str(out), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"\n  ✅ {out.name}  (+{len(VARS_CATASTRO)} variables catastro)")

    # CSV auxiliar
    csv_out = cfg.OUT_DIR / "paso08_dataset_casi_final.csv"
    result.drop(columns=["geometry"]).to_csv(str(csv_out), index=False)
    log(f"  ✅ {csv_out.name}")

    (cfg.OUT_DIR / "paso08_reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    log("\n  PASO 8 COMPLETADO")
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
