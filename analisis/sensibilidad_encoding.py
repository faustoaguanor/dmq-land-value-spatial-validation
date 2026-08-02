"""
sensibilidad_encoding.py
========================
C5 — Sensibilidad de la codificación de `uso_suelo_cod` (variable nominal).

Compara, sobre el holdout fijo 80/20, el desempeno de un modelo con
`uso_suelo_cod` tratado como (a) continuo (codificacion actual de la tesis)
vs (b) one-hot (dummies, codificacion correcta para variable nominal).

NO toca ningun resultado existente: solo escribe analisis/output_log/sensibilidad_encoding.csv.

Objetivo: cuantificar el ΔRMSE por la eleccion de codificacion para
documentarlo en 7.1.3 y respaldar la decision de migrar a one-hot.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "datos" / "dataset.gpkg"
SPLIT = ROOT / "data_split" / "split.csv"
OUT = Path(__file__).parent / "output_log" / "sensibilidad_encoding.csv"

COVARIABLES = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]
NOMINAL = "uso_suelo_cod"

def rmse_usd(y_true_log, y_pred_log):
    e = np.exp(y_true_log) - np.exp(y_pred_log)
    return float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e)))

def build_X(gdf, one_hot: bool):
    if not one_hot:
        return gdf[COVARIABLES].astype(float).values, COVARIABLES
    base = [c for c in COVARIABLES if c != NOMINAL]
    Xnum = gdf[base].astype(float)
    # one-hot sobre TODO el dataset (categorias deterministas, sin fuga del target)
    dummies = pd.get_dummies(gdf[NOMINAL].astype(int), prefix="uso", drop_first=True).astype(float)
    X = pd.concat([Xnum.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X.values, list(X.columns)

def main():
    gdf = gpd.read_file(DATA, layer="puntos_mercado").to_crs(epsg=32717)
    sp = pd.read_csv(SPLIT); sp["predio_join"] = sp["predio_join"].astype(int)
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    gdf = gdf.merge(sp[["predio_join", "split"]], on="predio_join", how="left")

    y_log = np.log(gdf["valor_m2"].astype(float).values)
    tr = (gdf["split"] == "train").values
    te = (gdf["split"] == "test").values
    print(f"n={len(gdf)}  train={tr.sum()}  test={te.sum()}")
    print(f"uso_suelo_cod categorias: {sorted(gdf[NOMINAL].unique())}")

    rows = []
    for label, oh in [("continuo (actual)", False), ("one-hot", True)]:
        X, names = build_X(gdf, oh)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = LinearRegression().fit(Xtr, y_log[tr])
        pred = m.predict(Xte)
        rmse, mae = rmse_usd(y_log[te], pred)
        print(f"  OLS  {label:18s}  p={len(names):2d}  RMSE={rmse:8.3f}  MAE={mae:7.3f}")
        rows.append({"modelo": "OLS", "encoding": label, "n_features": len(names),
                     "RMSE_usd": round(rmse, 4), "MAE_usd": round(mae, 4)})

    df = pd.DataFrame(rows)
    d_rmse = df.loc[df.encoding == "one-hot", "RMSE_usd"].iloc[0] - df.loc[df.encoding == "continuo (actual)", "RMSE_usd"].iloc[0]
    print(f"\n>>> Delta RMSE (one-hot - continuo) = {d_rmse:+.3f} USD/m2")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[CSV] {OUT}")

if __name__ == "__main__":
    main()
