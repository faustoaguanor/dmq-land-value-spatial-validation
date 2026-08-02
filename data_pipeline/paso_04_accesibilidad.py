"""
paso_04_accesibilidad.py
=========================
Calcula 9 distancias a infraestructura urbana usando cKDTree (O(n log m)).

INPUT:  config.INTERMEDIOS["paso03"]  → puntos con NBI
        config.FUENTES["dmot1_gdb"]   → capas DMOT_1 (metro, centralidades, nodos)
        config.FUENTES["pugs_gdb"]    → red vial ap030_via_l
OUTPUT: config.INTERMEDIOS["paso04"] → +9 variables dist_*
"""
from __future__ import annotations
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

VARS_DISTANCIA = [
    "dist_metro", "dist_centr_metro", "dist_centr_zonal",
    "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
]


def _cargar_dmot(gdb: str, layer: str, crs_sires: str, crs_utm: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(str(gdb), layer=layer)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_sires, allow_override=True)
    return gdf.to_crs(crs_utm)


def _dist_nearest(puntos_gdf: gpd.GeoDataFrame, destinos_gdf: gpd.GeoDataFrame,
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

    # ── 4.1 Cargar puntos ────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 4.1: Cargando puntos")
    log("═" * 60)

    puntos = gpd.read_file(str(cfg.INTERMEDIOS["paso03"]), layer=cfg.LAYER_PUNTOS)
    n_puntos = len(puntos)
    puntos_utm = puntos.to_crs(cfg.CRS_UTM)
    log(f"  Puntos: {n_puntos}")

    # ── 4.2 Metro ────────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 4.2: Distancia a estaciones de metro")
    log("═" * 60)

    metro = _cargar_dmot(cfg.FUENTES["dmot1_gdb"], "Estación_del_metro",
                         cfg.CRS_SIRES, cfg.CRS_UTM)
    puntos_utm["dist_metro"] = _dist_nearest(puntos_utm, metro, "dist_metro", log)

    # ── 4.3 Centralidades ────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 4.3: Distancia a centralidades")
    log("═" * 60)

    centr_all = _cargar_dmot(cfg.FUENTES["dmot1_gdb"], "Centralidad__punto_",
                             cfg.CRS_SIRES, cfg.CRS_UTM)
    centr_metro = centr_all[centr_all["jerarquia"] == "Metropolitana"].copy()
    centr_zonal = centr_all[centr_all["jerarquia"] == "Zonal"].copy()
    log(f"  Centralidades metropolitanas: {len(centr_metro)}  |  Zonales: {len(centr_zonal)}")
    puntos_utm["dist_centr_metro"] = _dist_nearest(puntos_utm, centr_metro, "dist_centr_metro", log)
    puntos_utm["dist_centr_zonal"] = _dist_nearest(puntos_utm, centr_zonal, "dist_centr_zonal", log)

    # ── 4.4 Nodos funcionales (CC, universidades, hospitales, parques, industrial) ──
    log("\n" + "═" * 60)
    log("  PASO 4.4: Distancia a nodos funcionales")
    log("═" * 60)

    nodos = _cargar_dmot(cfg.FUENTES["dmot1_gdb"], "Nodo_funcional",
                         cfg.CRS_SIRES, cfg.CRS_UTM)
    nodos_urb = nodos[nodos["clasifica"] == "Urbano"].copy()

    # Centros comerciales modernos
    cc_nodos = nodos_urb[nodos_urb["funcion"] == "Comercial"].copy()
    if "nam" in cc_nodos.columns:
        cc = cc_nodos[cc_nodos["nam"].isin(cfg.CC_MODERNOS)].copy()
    else:
        cc = cc_nodos
    puntos_utm["dist_cc"] = _dist_nearest(puntos_utm, cc, "dist_cc", log)

    # Universidades
    univ = nodos_urb[nodos_urb["funcion"] == "Universitario"].copy()
    puntos_utm["dist_universidad"] = _dist_nearest(puntos_utm, univ, "dist_universidad", log)

    # Hospitales
    hosp = nodos_urb[nodos_urb["funcion"] == "Salud"].copy()
    puntos_utm["dist_hospital"] = _dist_nearest(puntos_utm, hosp, "dist_hospital", log)

    # Parques (versión inicial con 3 nodos; se actualizará en paso 07)
    parques = nodos_urb[nodos_urb["funcion"] == "Recreativo Ambiental"].copy()
    puntos_utm["dist_parque_metro"] = _dist_nearest(puntos_utm, parques, "dist_parque_metro", log)

    # Zonas industriales
    indust = nodos_urb[nodos_urb["funcion"] == "Industrial"].copy()
    puntos_utm["dist_industrial"] = _dist_nearest(puntos_utm, indust, "dist_industrial", log)

    # ── 4.5 Vías principales (PUGS) ──────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 4.5: Distancia a vías principales")
    log("═" * 60)

    vias = gpd.read_file(str(cfg.FUENTES["pugs_gdb"]), layer="ap030_via_l")
    if vias.crs is None:
        vias = vias.set_crs(cfg.CRS_WGS84, allow_override=True)
    vias_utm = vias.to_crs(cfg.CRS_UTM)

    # Filtrar principales
    if "tipo" in vias_utm.columns:
        vias_ppal = vias_utm[vias_utm["tipo"].str.upper() == "PRINCIPAL"].copy()
    elif "clasifica" in vias_utm.columns:
        vias_ppal = vias_utm[vias_utm["clasifica"].str.upper() == "PRINCIPAL"].copy()
    else:
        vias_ppal = vias_utm

    log(f"  Vías principales: {len(vias_ppal)}")
    vias_ppal_pts = vias_ppal.copy()
    vias_ppal_pts["geometry"] = vias_ppal_pts.geometry.centroid
    puntos_utm["dist_via_principal"] = _dist_nearest(puntos_utm, vias_ppal_pts,
                                                      "dist_via_principal", log)

    # ── 4.6 Validar ──────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 4.6: Validando integridad")
    log("═" * 60)

    assert len(puntos_utm) == n_puntos, "ERROR: se perdieron puntos"
    for var in VARS_DISTANCIA:
        n_nulos = puntos_utm[var].isna().sum()
        assert n_nulos == 0, f"ERROR: {n_nulos} nulos en {var}"
    log(f"  ✓ Todos los {n_puntos} puntos conservados, 0 nulos en 9 variables")

    # ── 4.7 Exportar ─────────────────────────────────────────────────────────
    result_wgs84 = puntos_utm.to_crs(cfg.CRS_WGS84)
    out = cfg.INTERMEDIOS["paso04"]
    result_wgs84.to_file(str(out), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"\n  ✅ {out.name}  (+9 variables de distancia)")

    (cfg.OUT_DIR / "paso04_reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    log("\n  PASO 4 COMPLETADO")
    return result_wgs84


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
