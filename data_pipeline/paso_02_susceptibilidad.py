"""
paso_02_susceptibilidad.py
===========================
Extrae susceptibilidad a movimientos en masa (spatial join punto-en-polígono).

INPUT:  config.INTERMEDIOS["paso01"]   → 5,051 puntos limpios
        config.FUENTES["dmot2_gdb"]    → 362,799 polígonos DMOT_2
OUTPUT: config.INTERMEDIOS["paso02"]  → + columna suscept_codigo (1–4)
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

    # ── 2.1 Cargar puntos ────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 2.1: Cargando puntos limpios")
    log("═" * 60)

    puntos = gpd.read_file(str(cfg.INTERMEDIOS["paso01"]), layer=cfg.LAYER_PUNTOS)
    n_puntos = len(puntos)
    log(f"  Puntos: {n_puntos}  |  CRS: {puntos.crs}")

    # ── 2.2 Cargar susceptibilidad ───────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 2.2: Cargando capa de susceptibilidad")
    log("═" * 60)

    suscept = gpd.read_file(str(cfg.FUENTES["dmot2_gdb"]), layer=cfg.FUENTES["dmot2_layer"])
    log(f"  Polígonos: {len(suscept)}  |  CRS: {suscept.crs}")

    if "nvl_vazmm" not in suscept.columns:
        raise ValueError(f"Campo 'nvl_vazmm' no encontrado. Columnas: {list(suscept.columns)}")

    # ── 2.3 Reproyectar + join ────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 2.3: Reproyectando + spatial join")
    log("═" * 60)

    puntos_reproj = puntos.to_crs(suscept.crs)

    result = gpd.sjoin(
        puntos_reproj,
        suscept[["nvl_vazmm", "nvl_dazmm", "geometry"]],
        how="left",
        predicate="within",
    )
    if "index_right" in result.columns:
        result = result.drop(columns=["index_right"])

    result = result.rename(columns={"nvl_vazmm": "suscept_codigo", "nvl_dazmm": "suscept_clase"})

    n_sin = result["suscept_codigo"].isna().sum()
    log(f"  Con susceptibilidad: {n_puntos - n_sin}  |  Sin (imputar moda): {n_sin}")

    # ── 2.4 Imputar + validar ─────────────────────────────────────────────────
    if n_sin > 0:
        moda = result["suscept_codigo"].mode()[0]
        moda_clase = result["suscept_clase"].mode()[0]
        result["suscept_codigo"].fillna(moda, inplace=True)
        result["suscept_clase"].fillna(moda_clase, inplace=True)
        log(f"  Imputado con moda: {moda_clase} (código {moda})")

    assert len(result) == n_puntos, "ERROR: se perdieron puntos en el join"
    assert result["suscept_codigo"].isna().sum() == 0, "ERROR: nulos en suscept_codigo"
    assert set(result["suscept_codigo"].unique()).issubset({1, 2, 3, 4}), \
        f"ERROR: valores inválidos {result['suscept_codigo'].unique()}"
    log(f"  ✓ Distribución: {dict(result['suscept_clase'].value_counts())}")

    # ── 2.5 Exportar ─────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 2.5: Exportando resultados")
    log("═" * 60)

    result_wgs84 = result.to_crs(cfg.CRS_WGS84)
    out = cfg.INTERMEDIOS["paso02"]
    result_wgs84.to_file(str(out), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"  ✅ {out.name}  ({len(result_wgs84)} registros, +suscept_codigo)")

    (cfg.OUT_DIR / "paso02_reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    log("\n  PASO 2 COMPLETADO")
    return result_wgs84


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
