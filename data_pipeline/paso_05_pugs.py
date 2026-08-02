"""
paso_05_pugs.py
===============
Extrae uso de suelo (cod) y COS planta baja desde el PUGS 2024.

INPUT:  config.INTERMEDIOS["paso04"]  → puntos con 9 distancias
        config.FUENTES["pugs_gdb"]    → ba003_uso_suelo_edificabilidad_a
OUTPUT: config.INTERMEDIOS["paso05"] → +uso_suelo_cod, +cos_num
"""
from __future__ import annotations
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _normalizar_cos(val: object) -> float:
    """Normaliza COS a rango 0–1 (divide por 100 si ≥ 1)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ("no aplica", "n/a", ""):
        return np.nan
    try:
        num = float(s)
        return num / 100.0 if num >= 1 else num
    except Exception:
        return np.nan


def run(cfg=None) -> gpd.GeoDataFrame:
    if cfg is None:
        import config as cfg

    LOG: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        LOG.append(str(msg))

    # ── 5.1 Cargar puntos ────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 5.1: Cargando puntos")
    log("═" * 60)

    puntos = gpd.read_file(str(cfg.INTERMEDIOS["paso04"]), layer=cfg.LAYER_PUNTOS)
    n_puntos = len(puntos)
    puntos_utm = puntos.to_crs(cfg.CRS_UTM)
    log(f"  Puntos: {n_puntos}")

    # ── 5.2 Cargar PUGS ──────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 5.2: Cargando capa PUGS 2024")
    log("═" * 60)

    pugs = gpd.read_file(str(cfg.FUENTES["pugs_gdb"]), layer=cfg.FUENTES["pugs_layer"])
    log(f"  Polígonos PUGS: {len(pugs)}  |  CRS: {pugs.crs}")
    if pugs.crs is None:
        pugs = pugs.set_crs(cfg.CRS_SIRES, allow_override=True)
    pugs_utm = pugs.to_crs(cfg.CRS_UTM)

    for campo in ("cod_uso_gr", "cos_pb_ba"):
        if campo not in pugs_utm.columns:
            raise ValueError(f"Campo '{campo}' no encontrado en PUGS. Columnas: {list(pugs_utm.columns)}")

    # ── 5.3 Spatial join ─────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 5.3: Spatial join")
    log("═" * 60)

    cols_pugs = ["cod_uso_gr", "cos_pb_ba", "geometry"]
    if "uso_gral" in pugs_utm.columns:
        cols_pugs.append("uso_gral")

    result = gpd.sjoin(
        puntos_utm,
        pugs_utm[cols_pugs],
        how="left",
        predicate="within",
    )
    if "index_right" in result.columns:
        result = result.drop(columns=["index_right"])

    n_sin_pugs = result["cod_uso_gr"].isna().sum()
    log(f"  Con uso suelo: {n_puntos - n_sin_pugs}  |  Sin (imputar moda): {n_sin_pugs}")

    # ── 5.4 Procesar uso_suelo_cod ───────────────────────────────────────────
    result["uso_suelo_cod"] = result["cod_uso_gr"].map(cfg.MAPA_USO_SUELO)
    if result["uso_suelo_cod"].isna().any():
        moda = result["uso_suelo_cod"].mode()[0]
        result["uso_suelo_cod"].fillna(moda, inplace=True)

    # ── 5.5 Procesar cos_num ─────────────────────────────────────────────────
    result["cos_num"] = result["cos_pb_ba"].apply(_normalizar_cos)
    result.loc[result["cos_num"] > 1.0, "cos_num"] = 1.0
    if result["cos_num"].isna().any():
        mediana = result["cos_num"].median()
        result.loc[result["cos_num"].isna(), "cos_num"] = mediana

    # ── 5.6 Validar ──────────────────────────────────────────────────────────
    assert len(result) == n_puntos
    assert result["uso_suelo_cod"].isna().sum() == 0
    assert result["cos_num"].isna().sum() == 0
    log(f"  ✓ uso_suelo_cod: {dict(result['uso_suelo_cod'].value_counts().head(4))}")
    log(f"  ✓ cos_num: min={result['cos_num'].min():.2f}  med={result['cos_num'].median():.2f}  max={result['cos_num'].max():.2f}")

    # ── 5.7 Exportar ─────────────────────────────────────────────────────────
    cols_drop = [c for c in ("cod_uso_gr", "uso_gral", "cos_pb_ba") if c in result.columns]
    result = result.drop(columns=cols_drop)

    result_wgs84 = result.to_crs(cfg.CRS_WGS84)
    out = cfg.INTERMEDIOS["paso05"]
    result_wgs84.to_file(str(out), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"\n  ✅ {out.name}  (+uso_suelo_cod, +cos_num)")

    (cfg.OUT_DIR / "paso05_reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    log("\n  PASO 5 COMPLETADO")
    return result_wgs84


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
