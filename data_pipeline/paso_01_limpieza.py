"""
paso_01_limpieza.py
===================
Limpieza y validación de puntos de investigación de mercado.

INPUT:  config.FUENTES["mercado_gdb"] / layer mercado_layer  → 5,166 puntos
OUTPUT: config.INTERMEDIOS["paso01"]                         → 5,051 puntos limpios
"""
from __future__ import annotations
import re
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore")


def run(cfg=None) -> gpd.GeoDataFrame:
    if cfg is None:
        import config as cfg

    LOG: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        LOG.append(str(msg))

    # ── 1.1 Cargar datos ─────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 1.1: Cargando datos originales")
    log("═" * 60)

    gdf = gpd.read_file(str(cfg.FUENTES["mercado_gdb"]), layer=cfg.FUENTES["mercado_layer"])
    n_original = len(gdf)
    log(f"  Puntos cargados: {n_original}")
    log(f"  CRS original: {gdf.crs}")
    log(f"  Columnas: {list(gdf.columns)}")

    if len(gdf) > 0 and gdf.geometry.iloc[0].has_z:
        gdf["geometry"] = gdf.geometry.apply(lambda g: Point(g.x, g.y))
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=cfg.CRS_WGS84)
        log("  Coordenada Z eliminada → convertido a 2D")

    # ── 1.2 Limpiar campo PREDIO ──────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 1.2: Limpiando campo PREDIO")
    log("═" * 60)

    def limpiar_predio(val: object) -> str | None:
        s = str(val).strip()
        partes = re.split(r"[\s:,\-/\ny]+", s, flags=re.IGNORECASE)
        numeros = [p.strip() for p in partes if p.strip().isdigit()]
        return numeros[0] if numeros else None

    gdf["predio_clean"] = gdf["PREDIO"].apply(limpiar_predio)
    n_ok = gdf["predio_clean"].notna().sum()
    n_inv = gdf["predio_clean"].isna().sum()
    log(f"  Predios válidos: {n_ok}  |  Inválidos: {n_inv}")

    mask_multi = gdf["PREDIO"].astype(str).str.contains(r"[\s:,\-/]", regex=True)
    log(f"  Predios múltiples detectados: {mask_multi.sum()}")

    # ── 1.3 Validar valores de mercado ────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 1.3: Validando valores de mercado")
    log("═" * 60)

    gdf["valor_valido"] = (gdf["VALOR"] >= cfg.VALOR_MIN) & (gdf["VALOR"] <= cfg.VALOR_MAX)
    n_bajo = (gdf["VALOR"] < cfg.VALOR_MIN).sum()
    n_alto = (gdf["VALOR"] > cfg.VALOR_MAX).sum()
    log(f"  Rango [{cfg.VALOR_MIN}–{cfg.VALOR_MAX}] USD/m²: OK={n_original - n_bajo - n_alto}  bajo={n_bajo}  alto={n_alto}")

    # ── 1.4 Duplicados ────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 1.4: Verificando duplicados")
    log("═" * 60)

    gdf_con_predio = gdf[gdf["predio_clean"].notna()].copy()
    n_dup = gdf_con_predio.duplicated(subset="predio_clean", keep=False).sum()
    log(f"  Predios duplicados: {n_dup}  (estrategia: se conserva el primero)")

    # ── 1.5 Filtros ───────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 1.5: Aplicando filtros de limpieza")
    log("═" * 60)

    mask_final = (
        gdf["predio_clean"].notna()
        & gdf["valor_valido"]
        & ~gdf.duplicated(subset="predio_clean", keep="first")
    )
    gdf_limpio = gdf[mask_final].copy()
    n_final = len(gdf_limpio)
    log(f"  Original: {n_original}  |  Descartados: {n_original - n_final}  |  Finales: {n_final}")

    # ── 1.6 Preparar para exportación ────────────────────────────────────────
    columnas_salida = ["predio_clean", "VALOR", "VALOR_MUES", "PREDIO", "geometry"]
    columnas_disponibles = [c for c in columnas_salida if c in gdf_limpio.columns]
    gdf_export = gdf_limpio[columnas_disponibles].copy()
    gdf_export = gdf_export.rename(columns={"predio_clean": "predio_join", "VALOR": "valor_m2"})

    # ── 1.7 Exportar ─────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 1.7: Exportando resultados")
    log("═" * 60)

    out = cfg.INTERMEDIOS["paso01"]
    gdf_export.to_file(str(out), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"  ✅ {out.name}  ({n_final} registros)")

    log_path = cfg.OUT_DIR / "paso01_reporte.txt"
    log_path.write_text("\n".join(LOG), encoding="utf-8")

    log("\n  PASO 1 COMPLETADO")
    return gdf_export


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
