"""
construccion_dataset/config.py
================================
Rutas centralizadas y constantes del pipeline de construcción del dataset.

Edita FUENTES_DIR si los datos fuente están en otra ubicación.
Todo lo demás se deriva automáticamente.
"""
from __future__ import annotations
import os
from pathlib import Path

# =============================================================================
# Directorios raíz
# =============================================================================

#: Raíz del repositorio (replicacion/)
PROYECTO = Path(__file__).parent.parent

#: Directorio externo con los datos autorizados (GDB, CSV, SHP y TIF).
#: Se configura sin editar el código:
#:   Windows PowerShell: $env:DMQ_DATA_DIR = "D:\ruta\datos"
#:   Linux/macOS:        export DMQ_DATA_DIR=/ruta/datos
FUENTES_DIR = Path(
    os.environ.get("DMQ_DATA_DIR", PROYECTO / "datos_fuente")
).expanduser().resolve()

#: Directorio de intermedios del pipeline (construccion_dataset/output/)
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Fuentes primarias
# =============================================================================

FUENTES: dict[str, Path | str] = {
    # ── Paso 1: Puntos de investigación de mercado ────────────────────────
    "mercado_gdb":   FUENTES_DIR / "procesados" / "inf_procesada.gdb",
    "mercado_layer": "puntos_inv_2024_205_V",

    # ── Paso 2: Susceptibilidad geotécnica ────────────────────────────────
    "dmot2_gdb":     FUENTES_DIR / "PUGS_DMOT" / "DMOT_2.gdb",
    "dmot2_layer":   "supceptibilidad",

    # ── Paso 3: Atlas socioeconómico (NBI) ────────────────────────────────
    "atlas_shp":     FUENTES_DIR / "catastro" / "stp_sector_a.shp",

    # ── Pasos 4/6/7: Infraestructura urbana (DMOT_1) ──────────────────────
    "dmot1_gdb":     FUENTES_DIR / "PUGS_DMOT" / "DMOT_1.gdb",

    # ── Paso 5: Zonificación PUGS 2024 ────────────────────────────────────
    "pugs_gdb":      (FUENTES_DIR / "PUG_Ciudad_Linea"
                      / "plan_de_uso_y_gestión_del_suelo_2024.gdb"
                      / "PUGS_2024.gdb"),
    "pugs_layer":    "ba003_uso_suelo_edificabilidad_a",

    # ── Paso 8: Catastro Municipal ─────────────────────────────────────────
    "catastro_csv":  FUENTES_DIR / "Predio_variables.csv",

    # ── Paso 9: Pendiente del MDT ──────────────────────────────────────────
    "pendiente_csv": FUENTES_DIR / "CATASTRO_PREDIAL_diciembre" / "predio_pend.csv",
}

# =============================================================================
# Archivos intermedios (construccion_dataset/output/)
# =============================================================================

INTERMEDIOS: dict[str, Path] = {
    "paso01": OUT_DIR / "paso01_puntos_limpios.gpkg",
    "paso02": OUT_DIR / "paso02_con_suscept.gpkg",
    "paso03": OUT_DIR / "paso03_con_nbi.gpkg",
    "paso04": OUT_DIR / "paso04_con_accesibilidad.gpkg",
    "paso05": OUT_DIR / "paso05_con_pugs.gpkg",
    "paso06": OUT_DIR / "paso06_con_quebradas.gpkg",
    "paso07": OUT_DIR / "paso07_espacial_final.gpkg",
    "paso08": OUT_DIR / "paso08_con_catastro.gpkg",
    "paso09": OUT_DIR / "paso09_dataset_final.gpkg",
}

# =============================================================================
# Salidas finales → datos/
# =============================================================================

DATASET_FINAL_GPKG = PROYECTO / "datos" / "dataset.gpkg"
DATASET_FINAL_CSV  = PROYECTO / "datos" / "dataset.csv"

# =============================================================================
# CRS
# =============================================================================

CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32717"
CRS_SIRES = "ESRI:65162"   # Sistema de referencia DMOT (capas nativas)

# =============================================================================
# Constantes de dominio
# =============================================================================

LAYER_PUNTOS = "puntos_mercado"

VALOR_MIN = 5       # USD/m² mínimo realista
VALOR_MAX = 2500    # USD/m² máximo realista

ANIO_REFERENCIA = 2025   # Para calcular antigüedad = ANIO_REFERENCIA - año_const

# Centros comerciales modernos (filtro sobre capa Nodo_funcional / Comercial)
CC_MODERNOS = ["Quicentro Shopping", "CC Iñaquito", "CC El Bosque", "La Primavera"]

# Mapeos categóricos ──────────────────────────────────────────────────────────

MAPA_USO_SUELO: dict[str, int] = {
    "R":    1,   # Residencial
    "M":    2,   # Múltiple
    "I":    3,   # Industrial
    "E":    4,   # Equipamiento
    "PE":   5,   # Protección ecológica
    "PC":   6,   # Patrimonio cultural
    "RNR":  7,   # Recurso natural renovable
    "RNNR": 8,   # Recurso natural no renovable
    "SE":   9,   # Suelo de expansión
    "CSE":  10,  # Comercios y servicios especializados
}

MAPA_CONSERVACION: dict[str, int] = {
    "DE LUJO":      5,
    "MUY BUENO":    4,
    "BUENO":        3,
    "REGULAR":      2,
    "MALO":         1,
    "EN DETERIORO": 1,
    "NO TIENE":     0,
}

MAPA_ACABADOS: dict[str, int] = {
    "DE LUJO":    5,
    "DE PRIMERA": 4,
    "NORMAL":     3,
    "POPULAR":    2,
    "ECONÓMICO":  1,
}
