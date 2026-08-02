"""
paso_07_parques_plataforma.py
==============================
Actualiza dist_parque_metro con 24 parques (3 grandes + 21 adicionales)
y agrega dist_plataforma_gub.

INPUT:  config.INTERMEDIOS["paso06"]  → puntos completos (distancias parciales)
        config.FUENTES["dmot1_gdb"]   → Nodo_funcional, Otro_espacio_metropolitano,
                                         Equipamiento_administrativo
OUTPUT: config.INTERMEDIOS["paso07"] → dist_parque_metro actualizada
                                        + dist_plataforma_gub
"""
from __future__ import annotations
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


def _cargar_dmot(gdb: str, layer: str, crs_sires: str, crs_utm: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(str(gdb), layer=layer)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_sires, allow_override=True)
    return gdf.to_crs(crs_utm)


def _dist_nearest(puntos_gdf: gpd.GeoDataFrame,
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

    # ── 7.1 Cargar puntos ────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 7.1: Cargando puntos")
    log("═" * 60)

    puntos = gpd.read_file(str(cfg.INTERMEDIOS["paso06"]), layer=cfg.LAYER_PUNTOS)
    n_puntos = len(puntos)
    puntos_utm = puntos.to_crs(cfg.CRS_UTM)
    log(f"  Puntos: {n_puntos}")

    # ── 7.2 Actualizar dist_parque_metro (3 + 21 = 24 parques) ───────────────
    log("\n" + "═" * 60)
    log("  PASO 7.2: Actualizando dist_parque_metro (3+21=24 parques)")
    log("═" * 60)

    nodos = _cargar_dmot(cfg.FUENTES["dmot1_gdb"], "Nodo_funcional",
                         cfg.CRS_SIRES, cfg.CRS_UTM)
    nodos_urb = nodos[nodos["clasifica"] == "Urbano"].copy()
    parques_grandes = nodos_urb[nodos_urb["funcion"] == "Recreativo Ambiental"].copy()
    log(f"  Parques grandes (Nodo_funcional): {len(parques_grandes)}")

    parques_adicionales = _cargar_dmot(cfg.FUENTES["dmot1_gdb"], "Otro_espacio_metropolitano",
                                       cfg.CRS_SIRES, cfg.CRS_UTM)
    log(f"  Parques adicionales (Otro_espacio_metropolitano): {len(parques_adicionales)}")

    # Centroides para polígonos adicionales
    parques_adicionales_pts = parques_adicionales.copy()
    parques_adicionales_pts["geometry"] = parques_adicionales_pts.geometry.centroid

    parques_combinados = pd.concat(
        [parques_grandes[["geometry"]], parques_adicionales_pts[["geometry"]]],
        ignore_index=True,
    )
    parques_combinados = gpd.GeoDataFrame(parques_combinados, geometry="geometry", crs=cfg.CRS_UTM)
    log(f"  Total parques combinados: {len(parques_combinados)}")

    dist_anterior = puntos_utm["dist_parque_metro"].copy()
    nueva_dist = _dist_nearest(puntos_utm, parques_combinados, "dist_parque_metro (actualizada)", log)
    mejoras = (nueva_dist < dist_anterior).sum()
    log(f"  Mejoras: {mejoras} puntos con mejor acceso | "
        f"media anterior={dist_anterior.mean():.0f}m → nueva={nueva_dist.mean():.0f}m")
    puntos_utm["dist_parque_metro"] = nueva_dist

    # ── 7.3 Plataforma Gubernamental ─────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 7.3: Agregando dist_plataforma_gub")
    log("═" * 60)

    plataforma = _cargar_dmot(cfg.FUENTES["dmot1_gdb"], "Equipamiento_administrativo",
                              cfg.CRS_SIRES, cfg.CRS_UTM)
    log(f"  Equipamiento administrativo: {len(plataforma)}")

    plataforma_pts = plataforma.copy()
    plataforma_pts["geometry"] = plataforma_pts.geometry.centroid
    puntos_utm["dist_plataforma_gub"] = _dist_nearest(
        puntos_utm, plataforma_pts, "dist_plataforma_gub", log)

    # ── 7.4 Validar ──────────────────────────────────────────────────────────
    assert len(puntos_utm) == n_puntos
    assert puntos_utm["dist_parque_metro"].isna().sum() == 0
    assert puntos_utm["dist_plataforma_gub"].isna().sum() == 0
    log("  ✓ 0 nulos en ambas variables")

    # ── 7.5 Exportar ─────────────────────────────────────────────────────────
    result_wgs84 = puntos_utm.to_crs(cfg.CRS_WGS84)
    out = cfg.INTERMEDIOS["paso07"]
    result_wgs84.to_file(str(out), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"\n  ✅ {out.name}  (dist_parque_metro actualizada, +dist_plataforma_gub)")
    log(f"     Variables espaciales totales: {len(result_wgs84.columns) - 1}")

    (cfg.OUT_DIR / "paso07_reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    log("\n  PASO 7 COMPLETADO — DATASET ESPACIAL COMPLETO")
    return result_wgs84


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
