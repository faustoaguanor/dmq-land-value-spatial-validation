"""
pipeline.py
===========
Runner principal del pipeline de construcción del dataset.

Ejecuta los 9 pasos en secuencia. Por defecto omite pasos cuyo output
intermedio ya existe (modo incremental). Usa --force para re-ejecutar todo.

Uso:
    # Desde construccion_dataset/
    ../.venv/Scripts/python pipeline.py            # todos los pasos
    ../.venv/Scripts/python pipeline.py --step 4  # solo paso 4
    ../.venv/Scripts/python pipeline.py --force   # re-ejecutar todo
    ../.venv/Scripts/python pipeline.py --from 6  # desde paso 6 en adelante
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Asegurar que el módulo config está en el path ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg

# ── Importar pasos ────────────────────────────────────────────────────────────
import paso_01_limpieza            as p01
import paso_02_susceptibilidad     as p02
import paso_03_socioeconomico      as p03
import paso_04_accesibilidad       as p04
import paso_05_pugs                as p05
import paso_06_quebradas           as p06
import paso_07_parques_plataforma  as p07
import paso_08_catastro            as p08
import paso_09_pendiente           as p09


# =============================================================================
# Definición del pipeline
# =============================================================================

PASOS: list[tuple[int, str, object, str]] = [
    # (número, nombre, módulo, clave_output_intermedio)
    (1, "Limpieza de puntos de mercado",          p01, "paso01"),
    (2, "Susceptibilidad (DMOT_2)",               p02, "paso02"),
    (3, "Socioeconómico NBI (Atlas DMQ)",          p03, "paso03"),
    (4, "Accesibilidad urbana (9 distancias)",     p04, "paso04"),
    (5, "Uso de suelo y COS (PUGS 2024)",          p05, "paso05"),
    (6, "Quebradas y mercados mayoristas",         p06, "paso06"),
    (7, "Parques metropolitanos y Plataforma Gub", p07, "paso07"),
    (8, "Join Catastro Municipal",                 p08, "paso08"),
    (9, "Pendiente del terreno (MDT)",             p09, "paso09"),
]


# =============================================================================
# Runner
# =============================================================================

def _sep(char: str = "─", width: int = 65) -> str:
    return char * width


def run_pipeline(steps: list[int] | None = None, force: bool = False) -> None:
    """
    Ejecuta los pasos indicados (o todos si steps=None).

    Parameters
    ----------
    steps : list[int] | None
        Números de paso a ejecutar (1-based). None = todos.
    force : bool
        Si True, re-ejecuta incluso si el output intermedio ya existe.
    """
    pasos_a_ejecutar = [p for p in PASOS if steps is None or p[0] in steps]

    print("\n" + "═" * 65)
    print("  PIPELINE DE CONSTRUCCIÓN DEL DATASET — DMQ")
    print("═" * 65)
    print(f"  Pasos a ejecutar: {[p[0] for p in pasos_a_ejecutar]}")
    print(f"  Modo: {'forzar re-ejecución' if force else 'incremental (omite si existe output)'}")
    print(f"  Output intermedios: {cfg.OUT_DIR}")
    print(f"  Dataset final: {cfg.DATASET_FINAL_GPKG}")
    print("═" * 65)

    tiempos: dict[int, float] = {}
    pipeline_start = time.time()

    for num, nombre, modulo, clave in pasos_a_ejecutar:
        out_path = cfg.INTERMEDIOS[clave]

        print(f"\n{'─' * 65}")
        print(f"  PASO {num:02d}: {nombre}")
        print(f"{'─' * 65}")

        if out_path.exists() and not force:
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"  ⏭  Output ya existe ({out_path.name}, {size_mb:.1f} MB) — omitido")
            print(f"     Usa --force para re-ejecutar")
            tiempos[num] = 0.0
            continue

        t0 = time.time()
        try:
            modulo.run(cfg)
            elapsed = time.time() - t0
            tiempos[num] = elapsed
            print(f"\n  ✅ Paso {num:02d} completado en {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\n  ❌ Paso {num:02d} FALLÓ después de {elapsed:.1f}s:")
            print(f"     {type(exc).__name__}: {exc}")
            raise SystemExit(1) from exc

    # ── Resumen ───────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print("\n" + "═" * 65)
    print("  RESUMEN DEL PIPELINE")
    print("═" * 65)
    for num, nombre, _, _ in pasos_a_ejecutar:
        t = tiempos.get(num, 0)
        status = "⏭  omitido" if t == 0.0 else f"✅  {t:.1f}s"
        print(f"  Paso {num:02d}  {nombre:<42}  {status}")
    print(f"\n  Tiempo total: {total:.1f}s ({total / 60:.1f} min)")

    if cfg.DATASET_FINAL_GPKG.exists():
        size_mb = cfg.DATASET_FINAL_GPKG.stat().st_size / 1024 / 1024
        print(f"\n  Dataset final: {cfg.DATASET_FINAL_GPKG}  ({size_mb:.1f} MB)")
        print(f"  Dataset CSV:   {cfg.DATASET_FINAL_CSV}")
    print("═" * 65)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de construcción del dataset de valor del suelo DMQ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python pipeline.py                     # ejecutar todos los pasos (incremental)
  python pipeline.py --force             # forzar re-ejecución de todos
  python pipeline.py --step 4            # solo paso 4
  python pipeline.py --step 8 9          # pasos 8 y 9
  python pipeline.py --from 6            # desde paso 6 hasta el 9
        """,
    )
    parser.add_argument(
        "--step", nargs="+", type=int, metavar="N",
        help="Ejecutar solo el/los paso(s) indicado(s) (1–9)",
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, metavar="N",
        help="Ejecutar desde el paso N hasta el 9",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ejecutar aunque el output intermedio ya exista",
    )
    args = parser.parse_args()

    steps: list[int] | None = None
    if args.step:
        steps = args.step
    elif args.from_step:
        steps = list(range(args.from_step, 10))

    run_pipeline(steps=steps, force=args.force)


if __name__ == "__main__":
    main()
