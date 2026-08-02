"""
paso_06_quebradas.py
====================
Distancia a quebradas (polígonos) y mercados mayoristas (puntos).

INPUT:  config.INTERMEDIOS["paso05"]  → puntos con PUGS
        config.FUENTES["dmot1_gdb"]   → Quebrada_viva, Infraestructura_comercial
OUTPUT: config.INTERMEDIOS["paso06"] → +dist_quebrada, +dist_mercado_mayorista
"""
from __future__ import annotations
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.ops import unary_union

warnings.filterwarnings("ignore")


def _cargar_dmot(gdb: str, layer: str, crs_sires: str, crs_utm: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(str(gdb), layer=layer)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_sires, allow_override=True)
    return gdf.to_crs(crs_utm)


def _dist_nearest_polygon(puntos_gdf: gpd.GeoDataFrame,
                           poligonos_gdf: gpd.GeoDataFrame,
                           nombre: str, log) -> np.ndarray:
    """Distancia mínima al borde del conjunto unido de polígonos (shapely)."""
    log(f"  {nombre}: {len(poligonos_gdf)} polígonos (unary_union)…")
    union = unary_union(poligonos_gdf.geometry)
    dists = np.array([pt.distance(union) for pt in puntos_gdf.geometry])
    log(f"    min={dists.min():.0f}m  med={dists.mean():.0f}m  max={dists.max():.0f}m")
    return dists


def _dist_nearest_point(puntos_gdf: gpd.GeoDataFrame,
                         destinos_gdf: gpd.GeoDataFrame,
                         nombre: str, log) -> np.ndarray:
    pts = np.array([(g.x, g.y) for g in puntos_gdf.geometry])
    dest = np.array([(g.x, g.y) for g in destinos_gdf.geometry])
    tree = cKDTree(dest)
    dists, _ = tree.query(pts, k=1)
    log(f"  {nombre}: {len(destinos_gdf)} destinos | "
        f"min={dists.min():.0f}m  med={dists.mean():.0f}m  max={dists.max():.0f}m")
    return dists


def run(cfg=None) -> gpd.GeoDataFrame:
    if cfg is None:
        import config as cfg

    LOG: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        LOG.append(str(msg))

    # ── 6.1 Cargar puntos ────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 6.1: Cargando puntos")
    log("═" * 60)

    puntos = gpd.read_file(str(cfg.INTERMEDIOS["paso05"]), layer=cfg.LAYER_PUNTOS)
    n_puntos = len(puntos)
    puntos_utm = puntos.to_crs(cfg.CRS_UTM)
    log(f"  Puntos: {n_puntos}")

    # ── 6.2 Quebradas ────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 6.2: Distancia a quebradas (riesgo de inundación)")
    log("═" * 60)

    quebradas = _cargar_dmot(cfg.FUENTES["dmot1_gdb"], "Quebrada_viva",
                             cfg.CRS_SIRES, cfg.CRS_UTM)
    puntos_utm["dist_quebrada"] = _dist_nearest_polygon(
        puntos_utm, quebradas, "dist_quebrada", log)

    # ── 6.3 Mercados mayoristas ───────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 6.3: Distancia a mercados mayoristas")
    log("═" * 60)

    mercados = _cargar_dmot(cfg.FUENTES["dmot1_gdb"], "Infraestructura_comercial",
                            cfg.CRS_SIRES, cfg.CRS_UTM)
    puntos_utm["dist_mercado_mayorista"] = _dist_nearest_point(
        puntos_utm, mercados, "dist_mercado_mayorista", log)

    # ── 6.4 Validar ──────────────────────────────────────────────────────────
    assert len(puntos_utm) == n_puntos
    assert puntos_utm["dist_quebrada"].isna().sum() == 0
    assert puntos_utm["dist_mercado_mayorista"].isna().sum() == 0
    log("  ✓ 0 nulos en dist_quebrada y dist_mercado_mayorista")

    # ── 6.5 Exportar ─────────────────────────────────────────────────────────
    result_wgs84 = puntos_utm.to_crs(cfg.CRS_WGS84)
    out = cfg.INTERMEDIOS["paso06"]
    result_wgs84.to_file(str(out), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"\n  ✅ {out.name}  (+dist_quebrada, +dist_mercado_mayorista)")

    (cfg.OUT_DIR / "paso06_reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    log("\n  PASO 6 COMPLETADO")
    return result_wgs84


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
