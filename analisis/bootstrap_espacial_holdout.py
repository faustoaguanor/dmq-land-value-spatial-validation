"""
bootstrap_espacial_holdout.py
==============================
El bootstrap actual de comparativo_holdout_ci.csv (T2 en smearing_bootstrap_cdf.py)
remuestrea observaciones individuales i.i.d., ignorando que los residuos del
holdout tienen autocorrelacion espacial significativa (Moran I, Tabla 5). Esto
puede angostar artificialmente los intervalos de confianza.

Esta auditoria implementa un bootstrap por BLOQUES espaciales (cluster bootstrap):
se asigna cada predio de test a una celda de una grilla de tamano = rango de
autocorrelacion de residuos OLS (2,530 m, ya justificado en 7.3.2/8.5.1 del
borrador -- reuso del mismo parametro, no un valor nuevo ad hoc). En cada
replica se remuestrean CELDAS completas con reemplazo (no predios individuales),
preservando la dependencia espacial dentro de cada celda.

Salida: analisis/output_log/comparativo_holdout_ci_espacial.csv
(compara ancho de CI espacial vs i.i.d. ya guardado en comparativo_holdout_ci.csv)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "analisis" / "output_log"
CELL_M   = 2530.0   # mismo rango residual ya usado para SpatialBlock_buf

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

def block_bootstrap_ci(y_obs, y_pred, block_id, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    blocks = np.unique(block_id)
    nb = len(blocks)
    # indices por bloque (lista de arrays)
    idx_by_block = [np.where(block_id == b)[0] for b in blocks]
    vals = np.empty(n_boot)
    for r in range(n_boot):
        chosen = rng.integers(0, nb, size=nb)
        sel = np.concatenate([idx_by_block[c] for c in chosen])
        vals[r] = metric_fn(y_obs[sel], y_pred[sel])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), nb

def main():
    gdf = gpd.read_file(ROOT/"datos"/"dataset.gpkg", layer="puntos_mercado").to_crs(epsg=32717)
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    gdf["x"] = gdf.geometry.x; gdf["y"] = gdf.geometry.y
    x_min, y_min = gdf["x"].min(), gdf["y"].min()
    gdf["block_id"] = (
        ((gdf["x"] - x_min) // CELL_M).astype(int).astype(str) + "_" +
        ((gdf["y"] - y_min) // CELL_M).astype(int).astype(str)
    )
    block_map = dict(zip(gdf["predio_join"], gdf["block_id"]))
    n_blocks_total = gdf["block_id"].nunique()
    print(f"Grilla de {CELL_M:.0f} m: {n_blocks_total} celdas ocupadas en todo el dataset")

    rows = []
    for name, path in MODEL_SOURCES:
        if not path.exists():
            print(f"[--] {name}: no encontrado"); continue
        df = pd.read_csv(path)
        df["predio_join"] = df["predio_join"].astype(int)
        df_tr = df[df["split"] == "train"]
        df_te = df[df["split"] == "test"].copy()
        s_M = smearing_factor(df_tr)
        df_te["block_id"] = df_te["predio_join"].map(block_map)

        y_obs  = np.exp(df_te["y_obs_log"].values)
        y_pred = np.exp(df_te["y_pred_log"].values) * s_M
        block_id = df_te["block_id"].values

        rmse_lo, rmse_hi, nb = block_bootstrap_ci(y_obs, y_pred, block_id, rmse, n_boot=1000, seed=42)
        mae_lo,  mae_hi,  _  = block_bootstrap_ci(y_obs, y_pred, block_id, mae,  n_boot=1000, seed=42)
        point_rmse = rmse(y_obs, y_pred); point_mae = mae(y_obs, y_pred)

        rows.append({
            "modelo": name, "n_test": len(df_te), "n_celdas_test": nb,
            "RMSE": round(point_rmse, 4), "RMSE_ci_lo_espacial": round(rmse_lo, 4), "RMSE_ci_hi_espacial": round(rmse_hi, 4),
            "MAE": round(point_mae, 4), "MAE_ci_lo_espacial": round(mae_lo, 4), "MAE_ci_hi_espacial": round(mae_hi, 4),
        })
        print(f"[OK] {name:10s} RMSE={point_rmse:7.2f}  CI_espacial=[{rmse_lo:7.2f},{rmse_hi:7.2f}]  "
              f"ancho={rmse_hi-rmse_lo:6.2f}  (n_celdas={nb})")

    out = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    out.to_csv(OUT_DIR / "comparativo_holdout_ci_espacial.csv", index=False)
    print(f"\n[OK] {OUT_DIR/'comparativo_holdout_ci_espacial.csv'}")

    # Comparar ancho vs CI i.i.d. ya existente
    iid_path = OUT_DIR / "comparativo_holdout_ci.csv"
    if iid_path.exists():
        iid = pd.read_csv(iid_path).set_index("modelo")
        print("\nComparacion de ancho de CI (espacial vs i.i.d.):")
        for _, r in out.iterrows():
            m = r["modelo"]
            if m in iid.index:
                ancho_esp = r["RMSE_ci_hi_espacial"] - r["RMSE_ci_lo_espacial"]
                ancho_iid = iid.loc[m, "RMSE_ci_hi"] - iid.loc[m, "RMSE_ci_lo"]
                print(f"  {m:10s} ancho_espacial={ancho_esp:6.2f}  ancho_iid={ancho_iid:6.2f}  "
                      f"razon={ancho_esp/ancho_iid:.2f}x")

if __name__ == "__main__":
    main()
