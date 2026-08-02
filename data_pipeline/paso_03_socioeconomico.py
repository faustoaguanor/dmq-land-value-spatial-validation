"""
paso_03_socioeconomico.py
==========================
Extrae indicador socioeconómico (% pobreza NBI) desde el Atlas DMQ.

INPUT:  config.INTERMEDIOS["paso02"]   → puntos con susceptibilidad
        config.FUENTES["atlas_shp"]    → 7,221 sectores censales
OUTPUT: config.INTERMEDIOS["paso03"]  → + columna pc_pnbi (0–100%)
"""
from __future__ import annotations
import warnings
from pathlib import Path

import geopandas as gpd

warnings.filterwarnings("ignore")


def run(cfg=None) -> gpd.GeoDataFrame:
    if cfg is None:
        import config as cfg

    LOG: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        LOG.append(str(msg))

    # ── 3.1 Cargar puntos ────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 3.1: Cargando puntos con susceptibilidad")
    log("═" * 60)

    puntos = gpd.read_file(str(cfg.INTERMEDIOS["paso02"]), layer=cfg.LAYER_PUNTOS)
    n_puntos = len(puntos)
    log(f"  Puntos: {n_puntos}  |  CRS: {puntos.crs}")

    # ── 3.2 Cargar sectores censales ─────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 3.2: Cargando Atlas Socioeconómico (sectores censales)")
    log("═" * 60)

    sectores = gpd.read_file(str(cfg.FUENTES["atlas_shp"]))
    log(f"  Sectores: {len(sectores)}  |  CRS: {sectores.crs}")

    # Identificar campo NBI
    posibles = [c for c in sectores.columns if "nbi" in c.lower() or "pc_p" in c.lower()]
    campo_nbi = "pc_pnbi" if "pc_pnbi" in sectores.columns else (posibles[0] if posibles else None)
    if campo_nbi is None:
        raise ValueError(f"Campo NBI no encontrado. Columnas: {list(sectores.columns)}")
    log(f"  Campo NBI identificado: {campo_nbi}")

    # ── 3.3 Reproyectar + join ────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 3.3: Reproyectando + spatial join")
    log("═" * 60)

    puntos_utm = puntos.to_crs(cfg.CRS_UTM)
    sectores_utm = sectores.to_crs(cfg.CRS_UTM)

    cols_necesarias = ["geometry", campo_nbi]
    if "dpa_sector" in sectores_utm.columns:
        cols_necesarias.append("dpa_sector")

    result = gpd.sjoin(
        puntos_utm,
        sectores_utm[cols_necesarias],
        how="left",
        predicate="within",
    )
    if "index_right" in result.columns:
        result = result.drop(columns=["index_right"])

    if campo_nbi != "pc_pnbi":
        result = result.rename(columns={campo_nbi: "pc_pnbi"})

    n_sin = result["pc_pnbi"].isna().sum()
    log(f"  Con NBI: {n_puntos - n_sin}  |  Sin (imputar mediana): {n_sin}")

    # ── 3.4 Imputar + validar ─────────────────────────────────────────────────
    if n_sin > 0:
        mediana = result["pc_pnbi"].median()
        result["pc_pnbi"].fillna(mediana, inplace=True)
        log(f"  Imputado con mediana: {mediana:.2f}%")

    assert len(result) == n_puntos, "ERROR: se perdieron puntos"
    assert result["pc_pnbi"].isna().sum() == 0, "ERROR: nulos en pc_pnbi"
    assert ((result["pc_pnbi"] < 0) | (result["pc_pnbi"] > 100)).sum() == 0, \
        "ERROR: valores fuera de rango [0-100]"

    pnbi = result["pc_pnbi"]
    log(f"  ✓ pc_pnbi: min={pnbi.min():.1f}%  mediana={pnbi.median():.1f}%  max={pnbi.max():.1f}%")

    # ── 3.5 Exportar ─────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 3.5: Exportando resultados")
    log("═" * 60)

    result_wgs84 = result.to_crs(cfg.CRS_WGS84)
    out = cfg.INTERMEDIOS["paso03"]
    result_wgs84.to_file(str(out), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"  ✅ {out.name}  ({len(result_wgs84)} registros, +pc_pnbi)")

    (cfg.OUT_DIR / "paso03_reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    log("\n  PASO 3 COMPLETADO")
    return result_wgs84


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
