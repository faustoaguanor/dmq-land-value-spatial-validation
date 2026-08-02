"""
moran_significancia.py
======================
Recalcula el Indice de Moran I de los residuos de HOLDOUT para los 7 modelos
(OLS, MLP, GWR-17, GWR-27, GNNWR, SANNWR, GSAWR) anadiendo inferencia formal:
p-valor por permutacion (999) y pseudo z-score, segun Cliff & Ord (1981) /
Anselin (1995). Hasta ahora el proyecto reportaba solo la magnitud de I sin
prueba de significancia; este script cierra ese hueco sin re-entrenar modelos
(opera sobre las predicciones holdout ya guardadas en *_predictions.csv).

Pesos: KNN k=8 fila-estandarizados (identicos a los usados en los modelos).
Predicciones neurales: run canonico seed=42 (consistente con el resto del
pipeline de analisis comparativo).

Salida: analisis/output_log/moran_holdout_significancia.csv
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN as KNNWeights

ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "datos" / "dataset.gpkg"
OUT_DIR    = ROOT / "analisis" / "output_log"
OUT_DIR.mkdir(exist_ok=True)

MORAN_K      = 8
MORAN_NPERM  = 999
RANDOM_STATE = 42

# (display, ruta al CSV de predicciones)
# SANNWR = canonico (Ni 2022, sannwr_real_log, output_log_real) -> principal.
# SANNWR* = variante propia de grilla+SAPDNN (sannwr_log, output_log) -> exploratorio/anexo.
# (Antes este script leia sannwr_log_predictions.csv bajo la etiqueta "SANNWR", mezclando
# silenciosamente el canonico con la variante en la tabla de Moran I del core.)
FUENTES = [
    ("OLS",     ROOT/"modelos"/"ols"   /"output_log"      /"ols_log_predictions.csv"),
    ("MLP",     ROOT/"modelos"/"mlp"   /"output_log"      /"mlp_log_predictions.csv"),
    ("GWR-17",  ROOT/"modelos"/"gwr"   /"output_log"      /"gwr_log_predictions.csv"),
    ("GWR-27",  ROOT/"modelos"/"gwr"   /"output_log_27vars"/"gwr27_log_predictions.csv"),
    ("GNNWR",   ROOT/"modelos"/"gnnwr" /"output_log"      /"gnnwr_log_predictions.csv"),
    ("SANNWR",  ROOT/"modelos"/"sannwr"/"output_log_real" /"sannwr_real_log_predictions.csv"),
    ("SANNWR*", ROOT/"modelos"/"sannwr"/"output_log"      /"sannwr_log_predictions.csv"),
    ("GSAWR",   ROOT/"modelos"/"gsawr" /"output_log"      /"gsawr_log_predictions.csv"),
    # Controles tabulares: RF es modelo focal (Tabla 1) y faltaba en la tabla de Moran I.
    ("RF",      ROOT/"modelos"/"baselines"/"output_log"   /"rf_log_predictions.csv"),
    ("HGB",     ROOT/"modelos"/"baselines"/"output_log"   /"hgb_log_predictions.csv"),
]


def moran_perm(values, coords, k=MORAN_K, n_perm=MORAN_NPERM, seed=RANDOM_STATE):
    """Moran I + E[I] + p-valor permutacion (unilateral, autocorr. positiva) + z_sim."""
    w = KNNWeights.from_array(coords, k=k); w.transform = "r"
    z = values - values.mean()
    denom = float(z @ z)
    I_obs = float((z @ (w.sparse @ z)) / denom)
    rng = np.random.default_rng(seed)
    I_perm = np.empty(n_perm)
    for b in range(n_perm):
        zp = rng.permutation(z)
        I_perm[b] = (zp @ (w.sparse @ zp)) / denom
    p_sim = (int(np.sum(I_perm >= I_obs)) + 1) / (n_perm + 1)
    sd = I_perm.std(ddof=1)
    z_sim = float((I_obs - I_perm.mean()) / sd) if sd > 0 else float("nan")
    return I_obs, -1.0/(len(values)-1), float(p_sim), z_sim


def main():
    gdf = gpd.read_file(DATA_PATH, layer="puntos_mercado").to_crs(epsg=32717)
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    coord_map = {pj: (x, y) for pj, x, y in
                 zip(gdf["predio_join"], gdf.geometry.x, gdf.geometry.y)}

    rows = []
    for name, path in FUENTES:
        if not path.exists():
            print(f"[--] {name}: {path.name} NO encontrado"); continue
        df = pd.read_csv(path)
        df["predio_join"] = df["predio_join"].astype(int)
        te = df[df["split"] == "test"].copy()
        resid = (te["y_obs_log"] - te["y_pred_log"]).values.astype(float)
        coords = np.array([coord_map[pj] for pj in te["predio_join"]])
        I, EI, p_sim, z_sim = moran_perm(resid, coords)
        sig = "***" if p_sim < 0.01 else "**" if p_sim < 0.05 else "*" if p_sim < 0.10 else "ns"
        rows.append({"modelo": name, "n_test": len(te), "I": round(I, 6),
                     "EI": round(EI, 6), "p_sim": round(p_sim, 4),
                     "z_sim": round(z_sim, 4), "signif": sig})
        print(f"  {name:8s}  I={I:+.4f}  E[I]={EI:+.5f}  p_perm={p_sim:.4f}  z={z_sim:+.2f}  {sig}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "moran_holdout_significancia.csv", index=False)
    print(f"\n[CSV] {OUT_DIR/'moran_holdout_significancia.csv'}")
    print("\nLeyenda: *** p<0.01  ** p<0.05  * p<0.10  ns=no significativo")
    print("p_sim alto / ns => residuos sin autocorrelacion significativa (mejor).")


if __name__ == "__main__":
    main()
