"""
variograma_sensibilidad.py
==========================
Analisis de sensibilidad del rango de autocorrelacion espacial bajo 3 series:
  (a) valor_m2 raw             — serie original no estacionaria (tendencia espacial)
  (b) log(valor_m2)            — reduce sesgo de escala, mas cercano a estacionariedad
  (c) residuales OLS           — serie mas estacionaria; referencia metodologica correcta

Para cada serie se ajustan variogramas con 5 valores de maxlag (3, 5, 7, 10, 15 km)
y se selecciona el modelo de menor RMSE entre Spherical / Exponential / Gaussian.

Objetivo (soporte de D2): mostrar que el bloque de 5621.8 m es CONSERVADOR respecto
al rango de autocorrelacion de los residuales OLS (la serie estacionaria).

Salidas: analisis/output_log/
  variograma_sensibilidad.csv
  variograma_sensibilidad.png / .pdf
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    import skgstat as skg
except ImportError:
    print("ERROR: skgstat no instalado. Correr: pip install scikit-gstat"); sys.exit(1)

ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modelos"))
from features import build_feature_matrix

OUT_DIR = ROOT / "analisis" / "output_log"
OUT_DIR.mkdir(exist_ok=True)

BLOCK_SIZE_M = 5621.8   # bloque fijo — inmutable
N_LAGS       = 25
MODELS       = ["spherical", "exponential", "gaussian"]
MAXLAG_KM    = [3, 5, 7, 10, 15]   # km → m internamente
CRS_UTM      = 32717

print("="*65)
print("VARIOGRAMA: ANALISIS DE SENSIBILIDAD DEL RANGO")
print(f"Bloque fijo de referencia: {BLOCK_SIZE_M:.1f} m = {BLOCK_SIZE_M/1000:.3f} km")
print("="*65)

# ── Cargar datos ──────────────────────────────────────────────────────────────
gdf = gpd.read_file(ROOT/"datos"/"dataset.gpkg", layer="puntos_mercado").to_crs(epsg=CRS_UTM)
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
y_raw = gdf["valor_m2"].values.astype(float)
y_log = np.log(y_raw)
print(f"n={len(gdf)}  valor_m2: min={y_raw.min():.0f} max={y_raw.max():.0f} media={y_raw.mean():.1f}")

# ── Residuales OLS sobre todos los datos (para variograma de tendencia) ───────
X_raw, FEAT_NAMES = build_feature_matrix(gdf)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_raw)
X_int  = np.column_stack([np.ones(len(X_sc)), X_sc])
ols    = LinearRegression(fit_intercept=False).fit(X_int, y_log)
resid  = y_log - ols.predict(X_int)
print(f"OLS sobre log(valor_m2): R2={1 - np.var(resid)/np.var(y_log):.4f}  "
      f"std(resid)={resid.std():.4f}")

SERIES = {
    "valor_m2_raw":  y_raw,
    "log_valor_m2":  y_log,
    "residuales_OLS": resid,
}

def fit_best_variogram(coords, values, maxlag_m):
    """Ajusta Spherical/Exponential/Gaussian; devuelve (model, range_m, rmse, V)."""
    best_model, best_rmse, best_V = None, np.inf, None
    for model in MODELS:
        try:
            V = skg.Variogram(coordinates=coords, values=values,
                              model=model, maxlag=maxlag_m, n_lags=N_LAGS)
            if V.rmse < best_rmse:
                best_rmse = V.rmse; best_model = model; best_V = V
        except Exception:
            pass
    if best_V is None:
        return None, np.nan, np.nan, None
    return best_model, float(best_V.parameters[0]), float(best_rmse), best_V

# ── Tabla de sensibilidad ─────────────────────────────────────────────────────
rows = []
best_variograms = {}   # (serie, maxlag_km) -> V para figura

print(f"\n{'Serie':20s} {'maxlag_km':>10s} {'modelo':>12s} {'rango_m':>10s} {'rango_km':>10s} {'rmse_var':>10s}")
print("-"*75)
for serie, values in SERIES.items():
    for maxlag_km in MAXLAG_KM:
        maxlag_m = maxlag_km * 1000
        model, range_m, rmse_v, V = fit_best_variogram(coords, values, maxlag_m)
        conservador = "si" if (not np.isnan(range_m) and range_m <= BLOCK_SIZE_M) else "no"
        rows.append({
            "serie": serie,
            "maxlag_km": maxlag_km,
            "modelo": model if model else "fallo",
            "rango_m": round(range_m, 1) if not np.isnan(range_m) else np.nan,
            "rango_km": round(range_m/1000, 3) if not np.isnan(range_m) else np.nan,
            "rmse_variograma": round(rmse_v, 6) if not np.isnan(rmse_v) else np.nan,
            "bloque_conservador": conservador,
        })
        if V is not None:
            best_variograms[(serie, maxlag_km)] = V
        print(f"  {serie:20s} {maxlag_km:>9}  {(model or 'fallo'):>12s} {range_m:>10.0f} {range_m/1000:>10.2f} {rmse_v:>10.4f}")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "variograma_sensibilidad.csv", index=False)
print(f"\n[OK] variograma_sensibilidad.csv guardado  ({len(df)} filas)")

# ── Resumen por serie ─────────────────────────────────────────────────────────
print("\n--- Resumen: rango medio por serie (todas las ventanas de maxlag) ---")
for serie in SERIES:
    sub = df[df["serie"] == serie]["rango_m"].dropna()
    pct_conservador = (df[(df["serie"]==serie)]["bloque_conservador"]=="si").mean()*100
    print(f"  {serie:22s}: rango {sub.min():.0f}–{sub.max():.0f} m "
          f"(media={sub.mean():.0f} m, {pct_conservador:.0f}% ventanas <= bloque 5621 m)")

# ── Figura: variogramas de residuales OLS con las 5 ventanas maxlag ───────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
serie_labels = {
    "valor_m2_raw":   "valor_m2 raw",
    "log_valor_m2":   "log(valor_m2)",
    "residuales_OLS": "Residuales OLS",
}
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(MAXLAG_KM)))

for ax, (serie, label) in zip(axes, serie_labels.items()):
    plotted = False
    for (s, km), V in best_variograms.items():
        if s != serie: continue
        color = colors[MAXLAG_KM.index(km)]
        row = df[(df["serie"]==serie) & (df["maxlag_km"]==km)].iloc[0]
        range_m = row["rango_m"]
        try:
            bins, exp = V.preprocessing(force=True)
            ax.plot(bins, exp, "o", color=color, ms=3, alpha=0.6)
            x_fit = np.linspace(0, km*1000, 200)
            y_fit = V.transform(x_fit)
            ax.plot(x_fit, y_fit, "-", color=color, lw=1.5,
                    label=f"maxlag={km}km → rango={range_m/1000:.1f}km")
            if not np.isnan(range_m):
                ax.axvline(range_m, color=color, lw=0.8, linestyle=":")
            plotted = True
        except Exception:
            pass
    ax.axvline(BLOCK_SIZE_M, color="red", lw=2, linestyle="--",
               label=f"Bloque={BLOCK_SIZE_M/1000:.2f}km")
    ax.set_xlabel("Distancia (m)", fontsize=10)
    ax.set_ylabel("Semivarianza", fontsize=10)
    ax.set_title(f"Serie: {label}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")

fig.suptitle(
    f"Sensibilidad del Rango de Autocorrelación — DMQ (n={len(gdf):,})\n"
    f"Línea roja = bloque SpatialBlock ({BLOCK_SIZE_M:.0f} m). "
    "El bloque es conservador si el rango cae a su izquierda.",
    fontsize=11, fontweight="bold", y=1.02,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "variograma_sensibilidad.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "variograma_sensibilidad.pdf", bbox_inches="tight")
plt.close(fig)
print(f"[OK] variograma_sensibilidad.png / .pdf guardado")
print(f"\n>>> El bloque de {BLOCK_SIZE_M:.0f} m es conservador si el rango de residuales OLS < {BLOCK_SIZE_M:.0f} m")
print("    (Si el rango de residuales <= bloque, la separacion SB reduce toda la autocorrelacion.)")
