"""
sensibilidad_outliers.py
=========================
Auditoria de soporte comun: frente_m alcanza 6,391 m (p99=264 m, 24x) y
area_const_m2 alcanza 61,353 m2 (p99=2,208 m2, 28x) sobre el dataset completo.
Estos extremos pueden dominar el error de holdout sin que la causa sea la
arquitectura del modelo. Esta auditoria recalcula RMSE/MAE de holdout
excluyendo los predios de test que superan el p99 de frente_m O area_const_m2
(regla predefinida sobre el dataset completo, no elegida post-hoc mirando
el error de cada modelo).

No reentrena modelos: usa las predicciones de holdout ya generadas + el factor
de smearing ya calculado.

Salida: analisis/output_log/sensibilidad_outliers.csv
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "analisis" / "output_log"

MODEL_SOURCES = [
    ("OLS",       ROOT/"modelos"/"ols"/"output_log"/"ols_log_predictions.csv"),
    ("MLP",       ROOT/"modelos"/"mlp"/"output_log"/"mlp_log_predictions.csv"),
    ("GWR-17",    ROOT/"modelos"/"gwr"/"output_log"/"gwr_log_predictions.csv"),
    ("GWR-27",    ROOT/"modelos"/"gwr"/"output_log_27vars"/"gwr27_log_predictions.csv"),
    ("GNNWR",     ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_predictions.csv"),
    ("SANNWR",    ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_predictions.csv"),
    ("SANNWR*",   ROOT/"modelos"/"sannwr"/"output_log"/"sannwr_log_predictions.csv"),
    ("GSAWR",     ROOT/"modelos"/"gsawr"/"output_log"/"gsawr_log_predictions.csv"),
]

def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def mae(y, yhat):  return float(np.mean(np.abs(y - yhat)))

def smearing_factor(df_train):
    e = df_train["y_obs_log"].values - df_train["y_pred_log"].values
    return float(np.mean(np.exp(e)))

def main():
    gdf = gpd.read_file(ROOT/"datos"/"dataset.gpkg", layer="puntos_mercado")
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    p99_frente = np.percentile(gdf["frente_m"], 99)
    p99_area   = np.percentile(gdf["area_const_m2"], 99)
    print(f"p99 frente_m={p99_frente:.1f} m   p99 area_const_m2={p99_area:.1f} m2")
    outlier_ids = set(gdf.loc[
        (gdf["frente_m"] > p99_frente) | (gdf["area_const_m2"] > p99_area), "predio_join"
    ])
    print(f"Predios marcados outlier (sobre el dataset completo): {len(outlier_ids)}/{len(gdf)}")

    rows = []
    for name, path in MODEL_SOURCES:
        if not path.exists():
            print(f"[--] {name}: no encontrado"); continue
        df = pd.read_csv(path)
        df["predio_join"] = df["predio_join"].astype(int)
        df_tr = df[df["split"] == "train"]
        df_te = df[df["split"] == "test"].copy()
        s_M = smearing_factor(df_tr)
        df_te["y_obs"]  = np.exp(df_te["y_obs_log"])
        df_te["y_pred"] = np.exp(df_te["y_pred_log"]) * s_M

        full  = df_te
        clean = df_te[~df_te["predio_join"].isin(outlier_ids)]

        row = {
            "modelo": name,
            "n_test_full": len(full), "n_test_sin_outliers": len(clean),
            "n_excluidos": len(full) - len(clean),
            "RMSE_full": round(rmse(full["y_obs"].values, full["y_pred"].values), 4),
            "RMSE_sin_outliers": round(rmse(clean["y_obs"].values, clean["y_pred"].values), 4),
            "MAE_full": round(mae(full["y_obs"].values, full["y_pred"].values), 4),
            "MAE_sin_outliers": round(mae(clean["y_obs"].values, clean["y_pred"].values), 4),
        }
        row["delta_RMSE_pct"] = round(100*(row["RMSE_sin_outliers"]-row["RMSE_full"])/row["RMSE_full"], 2)
        rows.append(row)
        print(f"  {name:10s} RMSE full={row['RMSE_full']:7.2f}  sin_outliers={row['RMSE_sin_outliers']:7.2f}  "
              f"Δ={row['delta_RMSE_pct']:+.2f}%  (excluidos={row['n_excluidos']})")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "sensibilidad_outliers.csv", index=False)
    print(f"\n[OK] {OUT_DIR/'sensibilidad_outliers.csv'}")

if __name__ == "__main__":
    main()
