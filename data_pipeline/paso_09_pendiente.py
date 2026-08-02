"""
paso_09_pendiente.py
=====================
Agrega pendiente media del predio (MDT) y exporta el dataset final.

INPUT:  config.INTERMEDIOS["paso08"]     → 5,051 puntos con 28 variables
        config.FUENTES["pendiente_csv"]  → predio_pend.csv (1M+ predios)
OUTPUT: config.INTERMEDIOS["paso09"]    → +pendiente_grados
        config.DATASET_FINAL_GPKG       → datos/dataset.gpkg (copia final)
        config.DATASET_FINAL_CSV        → datos/dataset.csv
"""
from __future__ import annotations
import shutil
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


def run(cfg=None) -> gpd.GeoDataFrame:
    if cfg is None:
        import config as cfg

    LOG: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        LOG.append(str(msg))

    # ── 9.1 Cargar dataset casi final ────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 9.1: Cargando dataset casi final")
    log("═" * 60)

    puntos = gpd.read_file(str(cfg.INTERMEDIOS["paso08"]), layer=cfg.LAYER_PUNTOS)
    n_puntos = len(puntos)
    log(f"  Puntos: {n_puntos}  |  Variables: {len(puntos.columns)}")

    # ── 9.2 Cargar pendientes ────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 9.2: Cargando pendientes del MDT")
    log("═" * 60)

    pendientes = pd.read_csv(
        str(cfg.FUENTES["pendiente_csv"]),
        sep=";", decimal=",", encoding="latin1", low_memory=False,
    )
    log(f"  Registros pendiente: {len(pendientes):,}  |  Columnas: {list(pendientes.columns)}")

    pendientes["predio_join"] = (
        pendientes["predio"].astype(str).str.replace(".0", "", regex=False)
    )
    puntos["predio_join"] = puntos["predio_join"].astype(str)

    pend_vals = pendientes["pend_media"].dropna()
    log(f"  pend_media: min={pend_vals.min():.1f}°  med={pend_vals.median():.1f}°  max={pend_vals.max():.1f}°")

    # ── 9.3 Merge ────────────────────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 9.3: Merge con dataset")
    log("═" * 60)

    result = puntos.merge(
        pendientes[["predio_join", "pend_media"]],
        on="predio_join",
        how="left",
    )

    n_con = result["pend_media"].notna().sum()
    n_sin = result["pend_media"].isna().sum()
    log(f"  Con pendiente: {n_con}  |  Sin (imputar mediana): {n_sin}")

    # ── 9.4 Procesar + validar ───────────────────────────────────────────────
    result["pendiente_grados"] = result["pend_media"]
    if result["pendiente_grados"].isna().any():
        mediana = result["pendiente_grados"].median()
        result["pendiente_grados"] = result["pendiente_grados"].fillna(mediana)
        log(f"  Imputado con mediana: {mediana:.2f}°")

    if "pend_media" in result.columns:
        result = result.drop(columns=["pend_media"])

    assert len(result) == n_puntos, "ERROR: se perdieron puntos"
    assert result["pendiente_grados"].isna().sum() == 0, "ERROR: nulos en pendiente_grados"

    pend_f = result["pendiente_grados"]
    log(f"\n  pendiente_grados en dataset:")
    log(f"    min={pend_f.min():.1f}°  med={pend_f.median():.1f}°  max={pend_f.max():.1f}°")
    log(f"  Variables totales (incl. geometría): {len(result.columns)}")

    # ── 9.5 Exportar intermedio ──────────────────────────────────────────────
    log("\n" + "═" * 60)
    log("  PASO 9.5: Exportando dataset final")
    log("═" * 60)

    out_gpkg = cfg.INTERMEDIOS["paso09"]
    result.to_file(str(out_gpkg), layer=cfg.LAYER_PUNTOS, driver="GPKG")
    log(f"  ✅ {out_gpkg.name}  (+pendiente_grados)")

    # CSV intermedio
    csv_tmp = cfg.OUT_DIR / "paso09_dataset_final.csv"
    result.drop(columns=["geometry"]).to_csv(str(csv_tmp), index=False)

    # ── 9.6 Copiar a datos/ (destino final del proyecto) ─────────────────────
    log("\n" + "═" * 60)
    log("  PASO 9.6: Copiando dataset final a datos/")
    log("═" * 60)

    cfg.DATASET_FINAL_GPKG.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(out_gpkg), str(cfg.DATASET_FINAL_GPKG))
    log(f"  ✅ {cfg.DATASET_FINAL_GPKG}")

    result.drop(columns=["geometry"]).to_csv(str(cfg.DATASET_FINAL_CSV), index=False)
    log(f"  ✅ {cfg.DATASET_FINAL_CSV}")

    # ── 9.7 Resumen final ────────────────────────────────────────────────────
    n_vars_pred = len(result.columns) - 2  # excluye predio_join y geometry
    log("\n" + "═" * 60)
    log("  🎉 DATASET FINAL COMPLETO")
    log("═" * 60)
    log(f"  Observaciones:        {n_puntos:,}")
    log(f"  Variable dependiente: valor_m2")
    log(f"  Variables predictoras:{n_vars_pred}")
    log(f"  Completitud:          100% (0 nulos)")
    log(f"  Archivos:")
    log(f"    {cfg.DATASET_FINAL_GPKG}")
    log(f"    {cfg.DATASET_FINAL_CSV}")

    (cfg.OUT_DIR / "paso09_reporte.txt").write_text("\n".join(LOG), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run()
