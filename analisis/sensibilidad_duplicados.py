"""
sensibilidad_duplicados.py
===========================
Auditoria de coordenadas duplicadas (mismo x,y exacto) y sensibilidad del
holdout al excluirlas.

Hallazgo de auditoria adversarial: 95 grupos de coordenadas duplicadas (213
filas), 37 grupos cruzan holdout train/test, 42/1011 puntos de test tienen
un "gemelo" de coordenada exacta en training. Esto puede inflar las metricas
de interpolacion si el gemelo en train le da al modelo informacion directa
del punto de test (mismo predio dividido en registros, o error de captura).

Esta auditoria NO elimina los duplicados del dataset (decision de diseno,
fuera de alcance reentrenar). Solo cuantifica el efecto: recalcula RMSE/MAE
con y sin los 42 puntos de test "gemelos", usando las predicciones holdout
ya generadas (sin reentrenar) + el factor de smearing ya calculado por
smearing_bootstrap_cdf.py.

Salida: analisis/output_log/sensibilidad_duplicados.csv
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "datos" / "dataset.gpkg"
SPLIT    = ROOT / "data_split" / "split.csv"
OUT_DIR  = ROOT / "analisis" / "output_log"
OUT_DIR.mkdir(exist_ok=True)

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
def mape(y, yhat): return float(np.mean(np.abs((y - yhat) / y)) * 100)

def smearing_factor(df_train):
    e = df_train["y_obs_log"].values - df_train["y_pred_log"].values
    return float(np.mean(np.exp(e)))

def main():
    gdf = gpd.read_file(DATA, layer="puntos_mercado")
    sp = pd.read_csv(SPLIT)
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    sp["predio_join"]  = sp["predio_join"].astype(int)
    gdf = gdf.merge(sp[["predio_join", "split"]], on="predio_join", how="left")
    gdf["x"] = gdf.geometry.x; gdf["y"] = gdf.geometry.y

    test_df  = gdf[gdf["split"] == "test"]
    train_df = gdf[gdf["split"] == "train"]
    train_coords = set(zip(train_df["x"], train_df["y"]))
    is_twin = test_df.apply(lambda r: (r["x"], r["y"]) in train_coords, axis=1)
    twin_ids = set(test_df.loc[is_twin, "predio_join"].astype(int))
    print(f"Puntos de test con coordenada exacta duplicada en train: {len(twin_ids)}/{len(test_df)}")

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

        full = df_te
        clean = df_te[~df_te["predio_join"].isin(twin_ids)]

        row = {
            "modelo": name,
            "n_test_full": len(full), "n_test_sin_gemelos": len(clean),
            "n_excluidos": len(full) - len(clean),
            "RMSE_full": round(rmse(full["y_obs"].values, full["y_pred"].values), 4),
            "RMSE_sin_gemelos": round(rmse(clean["y_obs"].values, clean["y_pred"].values), 4),
            "MAE_full": round(mae(full["y_obs"].values, full["y_pred"].values), 4),
            "MAE_sin_gemelos": round(mae(clean["y_obs"].values, clean["y_pred"].values), 4),
        }
        row["delta_RMSE_pct"] = round(100 * (row["RMSE_sin_gemelos"] - row["RMSE_full"]) / row["RMSE_full"], 2)
        row["delta_MAE_pct"]  = round(100 * (row["MAE_sin_gemelos"]  - row["MAE_full"])  / row["MAE_full"], 2)
        rows.append(row)
        print(f"  {name:10s} RMSE full={row['RMSE_full']:7.2f}  sin_gemelos={row['RMSE_sin_gemelos']:7.2f}  "
              f"Δ={row['delta_RMSE_pct']:+.2f}%")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "sensibilidad_duplicados.csv", index=False)
    print(f"\n[OK] {OUT_DIR/'sensibilidad_duplicados.csv'}")

if __name__ == "__main__":
    main()
