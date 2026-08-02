"""
tabla_sintesis_validacion.py — Tabla unica MAE+RMSE+R2 por modelo x esquema (auditoria 2026-07-11).
=====================================================================================================
Dos auditorias independientes (aplicada y de codigo/resultados) senalaron que la tesis reporta
RMSE en unas tablas y MAE en otras, sin una tabla unica que muestre ambas metricas por esquema
para los 8 modelos (5 focales + GSAWR exploratorio + RF/HGB). Los datos ya existen en los CSV
consolidados; este script solo los une, sin reentrenar nada.

Estimando reportado por columna (declarado explicitamente, no mezclado):
  - OLS, GWR-27 (deterministas): media +/- sd ENTRE FOLDS del run base (comparativo_cv_principal.csv).
  - MLP, GNNWR, SANNWR, GSAWR, RF, HGB en RandomKFold/SpatialBlock: media +/- sd ENTRE SEEDS,
    calculada sobre la media-entre-folds de cada semilla (mismo estimando que la Tabla 2 de la
    tesis para los modelos neurales). Requiere *_cv_replicas.csv (5 seeds) y, para RF/HGB, las
    10 seeds de baselines_tabulares_replicas.py.
  - SpatialBlock_buf, modelos neurales: 1 sola semilla (run base), declarado exploratorio.
  - SpatialBlock_buf, RF/HGB: 10 seeds (replicas), unico bloque del estudio con multi-seed en buffer.

Salida: analisis/output_log/tabla_sintesis_validacion.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analisis" / "output_log"
BOUT = ROOT / "modelos" / "baselines" / "output_log"
BOUT_R = ROOT / "modelos" / "baselines" / "output_log_replicas"

SCHEMES = ["RandomKFold", "SpatialBlock", "SpatialBlock_buf"]


def det_from_runbase(model_display, results_path):
    """OLS/GWR-27: media +/- sd ENTRE FOLDS del run base (determinista)."""
    df = pd.read_csv(results_path)
    rows = []
    for est in SCHEMES:
        s = df[df["estrategia"] == est]
        if s.empty:
            continue
        rows.append({"modelo": model_display, "estrategia": est, "estimando": "entre_folds_run_base",
                     "n": len(s),
                     "MAE_mean": s.MAE.mean(), "MAE_std": s.MAE.std(ddof=1),
                     "RMSE_mean": s.RMSE.mean(), "RMSE_std": s.RMSE.std(ddof=1),
                     "R2_mean": s.R2.mean(), "R2_std": s.R2.std(ddof=1)})
    return rows


def multiseed_from_replicas(model_display, replicas_path, schemes=("RandomKFold", "SpatialBlock")):
    """Modelos neurales: media +/- sd ENTRE SEEDS de la media-entre-folds por seed (5 seeds)."""
    df = pd.read_csv(replicas_path)
    rows = []
    for est in schemes:
        s = df[df["estrategia"] == est]
        if s.empty:
            continue
        per_seed = s.groupby("seed").agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), R2=("R2", "mean"))
        rows.append({"modelo": model_display, "estrategia": est, "estimando": "entre_seeds_media_folds",
                     "n": len(per_seed),
                     "MAE_mean": per_seed.MAE.mean(), "MAE_std": per_seed.MAE.std(ddof=1),
                     "RMSE_mean": per_seed.RMSE.mean(), "RMSE_std": per_seed.RMSE.std(ddof=1),
                     "R2_mean": per_seed.R2.mean(), "R2_std": per_seed.R2.std(ddof=1)})
    return rows


def single_seed_buffer(model_display, results_path):
    """Modelos neurales bajo SpatialBlock_buf: 1 sola semilla (run base), SD entre folds, exploratorio."""
    df = pd.read_csv(results_path)
    s = df[df["estrategia"] == "SpatialBlock_buf"]
    if s.empty:
        return []
    return [{"modelo": model_display, "estrategia": "SpatialBlock_buf", "estimando": "1_seed_entre_folds_EXPLORATORIO",
              "n": len(s),
              "MAE_mean": s.MAE.mean(), "MAE_std": s.MAE.std(ddof=1),
              "RMSE_mean": s.RMSE.mean(), "RMSE_std": s.RMSE.std(ddof=1),
              "R2_mean": s.R2.mean(), "R2_std": s.R2.std(ddof=1)}]


records = []

# --- Deterministas: OLS, GWR-27 ---
records += det_from_runbase("OLS", ROOT/"modelos"/"ols"/"output_log"/"ols_log_results.csv")
records += det_from_runbase("GWR-27", ROOT/"modelos"/"gwr"/"output_log_27vars"/"gwr27_log_results.csv")

# --- Neurales con replicas 5 seeds (RandomKFold, SpatialBlock) + 1 seed buffer ---
neural = [
    ("MLP",    ROOT/"modelos"/"mlp"/"output_log"/"mlp_log_cv_replicas.csv",              ROOT/"modelos"/"mlp"/"output_log"/"mlp_log_results.csv"),
    ("GNNWR",  ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_cv_replicas.csv",           ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_results.csv"),
    ("SANNWR", ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_cv_replicas.csv", ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_results.csv"),
    ("GSAWR",  ROOT/"modelos"/"gsawr"/"output_log"/"gsawr_log_cv_replicas.csv",           ROOT/"modelos"/"gsawr"/"output_log"/"gsawr_log_results.csv"),
]
for name, rep_path, res_path in neural:
    if rep_path.exists():
        records += multiseed_from_replicas(name, rep_path)
    if res_path.exists():
        records += single_seed_buffer(name, res_path)

# --- RF/HGB: preferir 10-seed replicas si existen; si no, single-seed (baselines_tabulares.py) ---
for name, key in [("RF", "rf"), ("HGB", "hgb")]:
    rep_summary = BOUT_R / "baseline_replicas_summary.csv"
    rep_fold = BOUT_R / "baseline_replicas_fold.csv"
    if rep_fold.exists():
        df = pd.read_csv(rep_fold)
        df = df[df["modelo"] == name]
        for est in SCHEMES:
            s = df[df["estrategia"] == est]
            if s.empty:
                continue
            per_seed = s.groupby("seed").agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), R2=("R2", "mean"))
            records.append({"modelo": name, "estrategia": est, "estimando": "entre_seeds_media_folds_10seeds",
                             "n": len(per_seed),
                             "MAE_mean": per_seed.MAE.mean(), "MAE_std": per_seed.MAE.std(ddof=1),
                             "RMSE_mean": per_seed.RMSE.mean(), "RMSE_std": per_seed.RMSE.std(ddof=1),
                             "R2_mean": per_seed.R2.mean(), "R2_std": per_seed.R2.std(ddof=1)})
    else:
        # fallback: single-seed run base de baselines_tabulares.py
        res_path = BOUT / f"{key}_log_results.csv"
        df = pd.read_csv(res_path)
        for est in SCHEMES:
            s = df[df["estrategia"] == est]
            if s.empty:
                continue
            records.append({"modelo": name, "estrategia": est, "estimando": "1_seed_entre_folds",
                             "n": len(s),
                             "MAE_mean": s.MAE.mean(), "MAE_std": s.MAE.std(ddof=1),
                             "RMSE_mean": s.RMSE.mean(), "RMSE_std": s.RMSE.std(ddof=1),
                             "R2_mean": s.R2.mean(), "R2_std": s.R2.std(ddof=1)})

out_df = pd.DataFrame(records).round(4)
order_est = {e: i for i, e in enumerate(SCHEMES)}
out_df["_o"] = out_df["estrategia"].map(order_est)
out_df = out_df.sort_values(["estrategia", "MAE_mean"]).drop(columns="_o")
OUT.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT/"tabla_sintesis_validacion.csv", index=False)
print(out_df.to_string(index=False))
print(f"\n[CSV] {OUT/'tabla_sintesis_validacion.csv'}")
