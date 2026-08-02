"""
eda_completo.py
===============
Análisis Exploratorio de Datos completo sobre las 27 covariables originales
del dataset de valor del suelo DMQ (dataset.gpkg).

Propósito
---------
  Caracterizar los datos sin reducción de variables. Sirve como capítulo
  metodológico de la tesis: justifica decisiones de modelado (por qué GWR
  colapsa, por qué se necesitan kernels espaciales, etc.).

Secciones
---------
  1. Estadísticas descriptivas de todas las covariables y del target.
  2. Distribución del target valor_m2 (histograma, boxplot, test normalidad).
  3. VIF de las 27 covariables (muestra multicolinealidad).
  4. Correlación con el target (ranking y heatmap).
  5. Scatter plots de las variables más correlacionadas con valor_m2.
  6. Análisis espacial:
       a) Mapa de densidad de observaciones sobre el DMQ.
       b) Mapa coroplético de valor_m2.
       c) Índice de Moran I global sobre valor_m2.
       d) Moran I local (LISA): clusters HH, LL, HL, LH.
       e) Semivariograma empírico de valor_m2 (justifica rango CV=5621 m).

Salidas (output/)
-----------------
  eda_stats_target.csv          — stats descriptivas de valor_m2
  eda_stats_covariables.csv     — stats descriptivas de las 27 covariables
  eda_vif.csv                   — VIF de las 27 covariables
  eda_correlaciones.csv         — correlación y p-value de cada var con target
  fig_distribucion_target.png   — histograma + boxplot + QQ de valor_m2
  fig_vif.png                   — bar chart VIF con umbral VIF=10
  fig_correlaciones.png         — heatmap correlaciones entre todas las vars
  fig_scatter_top.png           — scatter plots top-6 vars correlacionadas
  fig_mapa_puntos.png           — mapa de observaciones sobre DMQ
  fig_mapa_valor.png            — mapa coroplético de valor_m2
  fig_moran_scatter.png         — Moran scatter plot (I global)
  fig_lisa.png                  — mapa LISA clusters
  fig_semivariograma.png        — semivariograma empírico de valor_m2
  eda_moran.csv                 — I global, p-value, z-score
  eda_semivariograma.csv        — distancia vs semivariance (datos del variograma)

Ejecución (desde eda/)
----------------------
    ../.venv/Scripts/python eda_completo.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# libpysal + esda para Moran
try:
    import libpysal
    from libpysal.weights import KNN as KNN_W
    from esda.moran import Moran, Moran_Local
    HAS_ESDA = True
except ImportError:
    HAS_ESDA = False
    print("[WARN] esda/libpysal no disponibles — análisis Moran omitido.")
    print("       Instalar: ../.venv/Scripts/pip install esda libpysal")

warnings.filterwarnings("ignore")

# =============================================================================
# Configuración
# =============================================================================

LAYER   = "puntos_mercado"
TARGET  = "valor_m2"
CRS_UTM = 32717           # EPSG:32717 — UTM 17S (metros)

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "datos" / "dataset.gpkg"
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

COVARIABLES: list[str] = [
    "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
    "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
    "dist_parque_metro", "dist_industrial", "dist_via_principal",
    "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
    "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
    "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
    "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
]

# Número de bins para semivariograma empírico
N_LAGS      = 20
MAX_DIST_KM = 25          # radio máximo en km (cubre el DMQ)

# Número vecinos para pesos espaciales Moran
MORAN_K = 8

PLT_STYLE = {
    "font.family": "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
}
plt.rcParams.update(PLT_STYLE)

# =============================================================================
# Carga de datos
# =============================================================================

print("=" * 60)
print("EDA COMPLETO — Dataset valor del suelo DMQ")
print("=" * 60)
print(f"\nCargando: {DATA_PATH}")
gdf = gpd.read_file(DATA_PATH, layer=LAYER)
print(f"  Observaciones : {len(gdf):,}")
print(f"  CRS original  : {gdf.crs}")

gdf_utm = gdf.to_crs(epsg=CRS_UTM)
coords   = np.column_stack([gdf_utm.geometry.x, gdf_utm.geometry.y])

y     = gdf[TARGET].values.astype(float)
log_y = np.log(y)

df_cov = gdf[COVARIABLES].copy()

# =============================================================================
# 1. Estadísticas descriptivas
# =============================================================================

print("\n[1/7] Estadísticas descriptivas...")

def stats_serie(arr: np.ndarray, name: str) -> dict:
    q1, q3 = np.percentile(arr, [25, 75])
    iqr     = q3 - q1
    n_out   = int(np.sum((arr < q1 - 3 * iqr) | (arr > q3 + 3 * iqr)))
    stat_n, p_norm = scipy_stats.normaltest(arr)
    return {
        "variable":         name,
        "n":                len(arr),
        "mean":             round(float(np.mean(arr)),   3),
        "std":              round(float(np.std(arr)),    3),
        "min":              round(float(np.min(arr)),    3),
        "p25":              round(float(q1),             3),
        "median":           round(float(np.median(arr)), 3),
        "p75":              round(float(q3),             3),
        "max":              round(float(np.max(arr)),    3),
        "skewness":         round(float(scipy_stats.skew(arr)),     4),
        "kurtosis":         round(float(scipy_stats.kurtosis(arr)), 4),
        "n_outliers_3IQR":  n_out,
        "pct_outliers":     round(100.0 * n_out / len(arr), 2),
        "normaltest_p":     round(float(p_norm), 6),
    }

stats_target = pd.DataFrame([
    stats_serie(y,     "valor_m2"),
    stats_serie(log_y, "log_valor_m2"),
])
stats_target.to_csv(OUT_DIR / "eda_stats_target.csv", index=False)
print(f"  valor_m2  — media={np.mean(y):.1f}  std={np.std(y):.1f}  "
      f"asimetría={scipy_stats.skew(y):.3f}")

stats_cov_rows = [stats_serie(df_cov[c].values.astype(float), c) for c in COVARIABLES]
pd.DataFrame(stats_cov_rows).to_csv(OUT_DIR / "eda_stats_covariables.csv", index=False)

# =============================================================================
# 2. Distribución del target
# =============================================================================

print("[2/7] Figura distribución target...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribución del Target: valor_m² (USD/m²)", fontsize=14, fontweight="bold")

for row, (arr, label) in enumerate([(y, "valor_m2 ($/m²)"), (log_y, "log(valor_m2)")]):
    skew = scipy_stats.skew(arr)
    _, p_sw = scipy_stats.shapiro(arr[:5000])    # Shapiro-Wilk max 5000 pts

    # Histograma
    ax = axes[row, 0]
    ax.hist(arr, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white", lw=0.3)
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title(f"Histograma\nAsimetría={skew:.3f}", fontsize=10)

    # Boxplot
    ax = axes[row, 1]
    bp = ax.boxplot(arr, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.7),
                    medianprops=dict(color="red", linewidth=2),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_ylabel(label, fontsize=10)
    ax.set_title("Boxplot (3×IQR outliers)", fontsize=10)
    ax.set_xticks([])

    # QQ plot
    ax = axes[row, 2]
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(arr, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.4, color="#4C72B0", rasterized=True)
    x_line = np.array([osm[0], osm[-1]])
    ax.plot(x_line, slope * x_line + intercept, color="red", lw=1.5)
    ax.set_xlabel("Cuantiles teóricos (Normal)", fontsize=9)
    ax.set_ylabel("Cuantiles empíricos", fontsize=9)
    ax.set_title(f"QQ-Plot\nShapiro-Wilk p={p_sw:.2e}", fontsize=10)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_distribucion_target.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 3. VIF de las 27 covariables
# =============================================================================

print("[3/7] Calculando VIF (27 variables)...")

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    cols = list(X.columns)
    rows = []
    Xsc  = StandardScaler().fit_transform(X.values.astype(float))
    for j, col in enumerate(cols):
        mask   = [i for i in range(len(cols)) if i != j]
        X_rest = Xsc[:, mask]
        X_j    = Xsc[:, j]
        ols    = LinearRegression().fit(X_rest, X_j)
        ss_res = np.sum((X_j - ols.predict(X_rest)) ** 2)
        ss_tot = np.sum((X_j - X_j.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif    = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        rows.append({"variable": col, "VIF": round(vif, 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)

df_vif = compute_vif(df_cov.fillna(df_cov.median()))
df_vif.to_csv(OUT_DIR / "eda_vif.csv", index=False)

n_high = (df_vif["VIF"] > 10).sum()
print(f"  Variables con VIF > 10: {n_high}/{len(df_vif)}")

# Figura VIF
fig, ax = plt.subplots(figsize=(12, 7))
colors_vif = ["#d62728" if v > 10 else "#4C72B0" for v in df_vif["VIF"]]
bars = ax.barh(df_vif["variable"][::-1], df_vif["VIF"][::-1],
               color=colors_vif[::-1], alpha=0.85, edgecolor="white")
ax.axvline(10, color="red", linestyle="--", lw=1.5, label="VIF = 10 (umbral)")
ax.set_xlabel("Factor de Inflación de Varianza (VIF)", fontsize=11)
ax.set_title("VIF de las 27 Covariables\n"
             "(rojo = VIF > 10, indica multicolinealidad severa)", fontsize=12)
ax.legend(fontsize=10)
# Anotar valores
for bar, v in zip(bars[::-1], df_vif["VIF"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}", va="center", ha="left", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_vif.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 4. Correlación con el target
# =============================================================================

print("[4/7] Correlaciones con valor_m2...")

corr_rows = []
for col in COVARIABLES:
    x_c = df_cov[col].values.astype(float)
    mask_valid = ~np.isnan(x_c)
    r, p = scipy_stats.pearsonr(x_c[mask_valid], y[mask_valid])
    rs, ps = scipy_stats.spearmanr(x_c[mask_valid], y[mask_valid])
    corr_rows.append({
        "variable": col,
        "pearson_r":  round(r,  4),
        "pearson_p":  round(p,  6),
        "spearman_r": round(rs, 4),
        "spearman_p": round(ps, 6),
        "abs_pearson": abs(r),
    })
df_corr = pd.DataFrame(corr_rows).sort_values("abs_pearson", ascending=False)
df_corr.drop(columns="abs_pearson").to_csv(OUT_DIR / "eda_correlaciones.csv", index=False)

# Heatmap de correlaciones entre todas las variables + target
print("  Calculando heatmap de correlaciones...")
df_all = df_cov.copy()
df_all["valor_m2"] = y
corr_matrix = df_all.fillna(df_all.median()).corr()

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr_matrix.columns, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Correlación de Pearson")
ax.set_title("Matriz de Correlaciones — 27 Covariables + Target", fontsize=13, pad=15)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_correlaciones.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 5. Scatter plots top-6 variables
# =============================================================================

print("[5/7] Scatter plots top-6 variables correlacionadas...")

top6 = df_corr["variable"].head(6).tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Relación con valor_m² — Top 6 Variables (|Pearson|)", fontsize=13, fontweight="bold")

for ax, col in zip(axes.flat, top6):
    x_c    = df_cov[col].values.astype(float)
    r_val  = df_corr.loc[df_corr["variable"] == col, "pearson_r"].values[0]
    # Submuestra para legibilidad
    idx    = np.random.default_rng(42).choice(len(y), min(1000, len(y)), replace=False)
    ax.scatter(x_c[idx], y[idx], s=8, alpha=0.4, color="#4C72B0", rasterized=True)
    # Línea de tendencia
    finite = np.isfinite(x_c) & np.isfinite(y)
    if finite.sum() > 2:
        m_t, b_t, *_ = scipy_stats.linregress(x_c[finite], y[finite])
        xl = np.array([np.nanmin(x_c), np.nanmax(x_c)])
        ax.plot(xl, m_t * xl + b_t, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("valor_m2 ($/m²)", fontsize=9)
    ax.set_title(f"r = {r_val:+.3f}", fontsize=10)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_scatter_top.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 6a. Mapa de observaciones
# =============================================================================

print("[6/7] Análisis espacial...")
print("  6a. Mapa de puntos...")

fig, ax = plt.subplots(figsize=(10, 12))
gdf_utm.plot(ax=ax, markersize=3, color="#4C72B0", alpha=0.5,
             rasterized=True, zorder=2)
ax.set_xlabel("Este UTM (m)", fontsize=10)
ax.set_ylabel("Norte UTM (m)", fontsize=10)
ax.set_title(f"Distribución Espacial de Observaciones\n"
             f"DMQ — n={len(gdf_utm):,} puntos de mercado", fontsize=12)
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_mapa_puntos.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 6b. Mapa coroplético de valor_m2
# =============================================================================

print("  6b. Mapa coroplético valor_m2...")

fig, ax = plt.subplots(figsize=(10, 12))
gdf_utm_plot = gdf_utm.copy()
gdf_utm_plot["valor_m2"] = y
gdf_utm_plot.plot(
    column="valor_m2", ax=ax, cmap="YlOrRd",
    markersize=4, alpha=0.7,
    legend=True,
    legend_kwds={"label": "valor_m² (USD/m²)", "shrink": 0.6},
    rasterized=True,
)
ax.set_xlabel("Este UTM (m)", fontsize=10)
ax.set_ylabel("Norte UTM (m)", fontsize=10)
ax.set_title("Valor del Suelo (USD/m²) — DMQ\n"
             "Gradiente espacial norte-sur visible", fontsize=12)
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_mapa_valor.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 6c-d. Moran I global y LISA
# =============================================================================

if HAS_ESDA:
    print("  6c. Moran I global...")
    w = KNN_W.from_array(coords, k=MORAN_K)
    w.transform = "R"

    mi = Moran(y, w, permutations=999)
    df_moran = pd.DataFrame([{
        "I":       round(mi.I,   5),
        "EI":      round(mi.EI,  5),
        "z_score": round(mi.z_sim, 4),
        "p_value": round(mi.p_sim,  5),
        "k_vecinos": MORAN_K,
        "permutaciones": 999,
    }])
    df_moran.to_csv(OUT_DIR / "eda_moran.csv", index=False)
    print(f"  Moran I = {mi.I:.4f}  p = {mi.p_sim:.4f}  z = {mi.z_sim:.2f}")

    # Moran scatter plot
    fig, ax = plt.subplots(figsize=(8, 7))
    y_std   = (y - y.mean()) / y.std()
    lag_y   = libpysal.weights.lag_spatial(w, y_std)
    ax.scatter(y_std, lag_y, s=6, alpha=0.3, color="#4C72B0", rasterized=True)
    m_s, b_s, *_ = scipy_stats.linregress(y_std, lag_y)
    xl = np.array([y_std.min(), y_std.max()])
    ax.plot(xl, m_s * xl + b_s, color="red", lw=2)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("valor_m2 estandarizado", fontsize=11)
    ax.set_ylabel("Lag espacial (media vecinos)", fontsize=11)
    ax.set_title(f"Diagrama de Dispersión de Moran\n"
                 f"I = {mi.I:.4f}   p = {mi.p_sim:.4f} (permutaciones=999)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_moran_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("  6d. LISA (Moran local)...")
    lisa = Moran_Local(y, w, permutations=999)

    # Clasificar cuadrantes
    sig   = lisa.p_sim < 0.05
    quads = lisa.q          # 1=HH, 2=LH, 3=LL, 4=HL
    labels_map = {1: "HH (Alto-Alto)", 2: "LH (Bajo-Alto)",
                  3: "LL (Bajo-Bajo)", 4: "HL (Alto-Bajo)"}
    colors_map = {1: "#d73027", 2: "#91bfdb", 3: "#4575b4", 4: "#fc8d59"}
    label_pts  = [labels_map.get(q, "No sig.") if s else "No sig."
                  for q, s in zip(quads, sig)]

    fig, ax = plt.subplots(figsize=(10, 12))
    not_sig_idx = [i for i, s in enumerate(sig) if not s]
    ax.scatter(coords[not_sig_idx, 0], coords[not_sig_idx, 1],
               s=4, color="lightgray", alpha=0.5, rasterized=True, label="No significativo")
    for q, col_q in colors_map.items():
        idx_q = [i for i, (s, qq) in enumerate(zip(sig, quads)) if s and qq == q]
        if idx_q:
            ax.scatter(coords[idx_q, 0], coords[idx_q, 1],
                       s=10, color=col_q, alpha=0.8,
                       rasterized=True, label=labels_map[q])
    ax.set_xlabel("Este UTM (m)", fontsize=10)
    ax.set_ylabel("Norte UTM (m)", fontsize=10)
    ax.set_title("Clusters LISA — Moran Local (p < 0.05)\n"
                 "valor_m² — DMQ", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_lisa.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sig_count = sig.sum()
    print(f"  LISA: {sig_count} puntos significativos ({100*sig_count/len(y):.1f}%)")
else:
    print("  [SKIP] Moran I / LISA — instalar esda+libpysal")

# =============================================================================
# 6e. Semivariograma empírico
# =============================================================================

print("  6e. Semivariograma empírico...")

# Subsample para que sea manejable (n=1500 pts → ~1.1M pares)
rng   = np.random.default_rng(42)
n_sub = min(1500, len(y))
idx_s = rng.choice(len(y), n_sub, replace=False)
c_sub = coords[idx_s]
y_sub = y[idx_s]

# Calcular pares distancia-semivariance
print(f"  Calculando pares para subsample n={n_sub}...")
diffs_pairs  = []
sv_pairs     = []
max_dist_m   = MAX_DIST_KM * 1000

for i in range(n_sub):
    dx   = c_sub[i+1:, 0] - c_sub[i, 0]
    dy   = c_sub[i+1:, 1] - c_sub[i, 1]
    dist = np.sqrt(dx**2 + dy**2)
    sv   = 0.5 * (y_sub[i+1:] - y_sub[i])**2
    mask = dist <= max_dist_m
    diffs_pairs.append(dist[mask])
    sv_pairs.append(sv[mask])

all_dist = np.concatenate(diffs_pairs)
all_sv   = np.concatenate(sv_pairs)

# Agrupar en N_LAGS bins
lag_edges  = np.linspace(0, max_dist_m, N_LAGS + 1)
lag_mid    = 0.5 * (lag_edges[:-1] + lag_edges[1:])
sv_mean    = []
sv_n       = []
for lo, hi in zip(lag_edges[:-1], lag_edges[1:]):
    mask   = (all_dist >= lo) & (all_dist < hi)
    n_pair = mask.sum()
    sv_n.append(n_pair)
    sv_mean.append(np.mean(all_sv[mask]) if n_pair > 0 else np.nan)

sv_mean = np.array(sv_mean)
sv_n    = np.array(sv_n)

df_vario = pd.DataFrame({
    "lag_m":   lag_mid,
    "semivariance": sv_mean,
    "n_pairs": sv_n,
})
df_vario.to_csv(OUT_DIR / "eda_semivariograma.csv", index=False)

# Figura semivariograma
fig, ax = plt.subplots(figsize=(11, 6))
valid = ~np.isnan(sv_mean)
ax.plot(lag_mid[valid] / 1000, sv_mean[valid],
        "o-", color="#4C72B0", markersize=7, lw=2, label="Semivarianza empírica")
# Nugget estimado (primer valor)
nugget = sv_mean[valid][0] if valid.any() else 0
# Línea varianza total (sill proxy)
sill = float(np.var(y))
ax.axhline(sill, color="red", linestyle="--", lw=1.5,
           label=f"Varianza total (sill aprox.) = {sill:,.0f}")
# Marcar rango CV=5621 m
ax.axvline(5.621, color="orange", linestyle=":", lw=2,
           label="Rango autocorrelación = 5,621 m (usado en SpatialBlockCV)")
ax.set_xlabel("Distancia (km)", fontsize=11)
ax.set_ylabel("Semivarianza  γ(h)", fontsize=11)
ax.set_title("Semivariograma Empírico de valor_m²\n"
             "Autocorrelación espacial decrece con la distancia", fontsize=12)
ax.legend(fontsize=9)
# Anotar n pares en cada punto
for xp, yp, n in zip(lag_mid[valid] / 1000, sv_mean[valid], sv_n[valid]):
    if n > 0:
        ax.annotate(f"n={n:,}", (xp, yp), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="gray")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_semivariograma.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# Resumen final
# =============================================================================

print("\n" + "=" * 60)
print("RESUMEN EDA")
print("=" * 60)
print(f"  Observaciones         : {len(gdf):,}")
print(f"  Covariables           : {len(COVARIABLES)}")
print(f"  valor_m2 — media      : {np.mean(y):.1f} USD/m²")
print(f"  valor_m2 — mediana    : {np.median(y):.1f} USD/m²")
print(f"  valor_m2 — asimetría  : {scipy_stats.skew(y):.3f}")
print(f"  VIF > 10              : {n_high} variables")
print(f"  Top corr. con target  : {df_corr['variable'].iloc[0]}  "
      f"r={df_corr['pearson_r'].iloc[0]:.3f}")
if HAS_ESDA:
    print(f"  Moran I               : {mi.I:.4f}  p={mi.p_sim:.4f}")
print(f"\nSalidas en: {OUT_DIR}")
salidas = list(OUT_DIR.iterdir())
for s in sorted(salidas):
    print(f"  {s.name}")
print("=" * 60)
