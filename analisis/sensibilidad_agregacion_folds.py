"""
sensibilidad_agregacion_folds.py
=================================
Auditoria de agregacion de folds: el pipeline actual (groupby+mean en cada
script de modelo) promedia RMSE/MAE por fold con peso IGUAL, aunque los folds
de SpatialBlock/SpatialBlock_buf tienen tamanos muy desiguales (596 a 1238
observaciones sobre 4,040 de train). Esta auditoria recalcula la version
"pooled" (ponderada por n_test de cada fold, equivalente al error esperado
por observacion en vez de por region) y compara contra el simple-mean actual.

Para RMSE, el pooled correcto es vía MSE: MSE_pooled = sum(n_k*RMSE_k^2)/sum(n_k).
Para MAE, el pooled exacto es: MAE_pooled = sum(n_k*MAE_k)/sum(n_k).

No modifica los CSV existentes (serian "contrato critico" de resultados ya
usados en tablas/figuras) -- solo cuantifica el efecto como chequeo de
sensibilidad.

Salida: analisis/output_log/sensibilidad_agregacion_folds.csv
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "analisis" / "output_log"
OUT_DIR.mkdir(exist_ok=True)

RESULTS_SOURCES = [
    ("OLS",     ROOT/"modelos"/"ols"/"output_log"/"ols_log_results.csv"),
    ("MLP",     ROOT/"modelos"/"mlp"/"output_log"/"mlp_log_results.csv"),
    ("GWR-17",  ROOT/"modelos"/"gwr"/"output_log"/"gwr_log_results.csv"),
    ("GWR-27",  ROOT/"modelos"/"gwr"/"output_log_27vars"/"gwr27_log_results.csv"),
    ("GNNWR",   ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_results.csv"),
    ("SANNWR",  ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_results.csv"),
    ("SANNWR*", ROOT/"modelos"/"sannwr"/"output_log"/"sannwr_log_results.csv"),
    ("GSAWR",   ROOT/"modelos"/"gsawr"/"output_log"/"gsawr_log_results.csv"),
]

def main():
    sp    = pd.read_csv(ROOT/"data_split"/"split.csv")
    folds = pd.read_csv(ROOT/"spatial_cv"/"output"/"fold_assignments.csv")
    sp["predio_join"]    = sp["predio_join"].astype(int)
    folds["predio_join"] = folds["predio_join"].astype(int)
    df = sp.merge(folds, on="predio_join", how="left")
    train = df[df["split"] == "train"]
    # n_test por fold de SpatialBlock (= n_test de SpatialBlock_buf, mismo test set)
    n_per_fold = train.groupby("fold").size().to_dict()
    print("n_test por fold (SpatialBlock / SpatialBlock_buf):", n_per_fold)

    rows = []
    for name, path in RESULTS_SOURCES:
        if not path.exists():
            print(f"[--] {name}: no encontrado"); continue
        res = pd.read_csv(path)
        for estrategia in ("SpatialBlock", "SpatialBlock_buf"):
            sub = res[res["estrategia"] == estrategia]
            if sub.empty:
                continue
            n = sub["fold"].map(n_per_fold).values.astype(float)
            if np.isnan(n).any():
                continue
            rmse_k = sub["RMSE"].values.astype(float)
            mae_k  = sub["MAE"].values.astype(float)

            simple_rmse = float(rmse_k.mean())
            simple_mae  = float(mae_k.mean())
            pooled_rmse = float(np.sqrt(np.sum(n * rmse_k**2) / n.sum()))
            pooled_mae  = float(np.sum(n * mae_k) / n.sum())

            rows.append({
                "modelo": name, "estrategia": estrategia, "n_folds": len(sub),
                "RMSE_simple_mean": round(simple_rmse, 4),
                "RMSE_pooled_ponderado": round(pooled_rmse, 4),
                "delta_RMSE_pct": round(100*(pooled_rmse-simple_rmse)/simple_rmse, 2),
                "MAE_simple_mean": round(simple_mae, 4),
                "MAE_pooled_ponderado": round(pooled_mae, 4),
                "delta_MAE_pct": round(100*(pooled_mae-simple_mae)/simple_mae, 2),
            })
            print(f"  {name:9s} {estrategia:18s} RMSE simple={simple_rmse:8.2f} "
                  f"pooled={pooled_rmse:8.2f} (Δ={100*(pooled_rmse-simple_rmse)/simple_rmse:+.1f}%)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "sensibilidad_agregacion_folds.csv", index=False)
    print(f"\n[OK] {OUT_DIR/'sensibilidad_agregacion_folds.csv'}")

if __name__ == "__main__":
    main()
