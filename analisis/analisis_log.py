"""
analisis_log.py
===============
Analisis comparativo de todos los modelos con log(valor_m2) + holdout 80/20.

Carga resultados de:
  modelos/ols/output_log/             — OLS
  modelos/mlp/output_log/             — MLP
  modelos/gwr/output_log/             — GWR-17 (17 vars VIF-clean, anexo)
  modelos/gwr/output_log_27vars/      — GWR-27 (27 vars + Ridge 0.1)
  modelos/gnnwr/output_log/           — GNNWR
  modelos/sannwr/output_log/          — SANNWR
  modelos/gsawr/output_log/           — GSAWR

SAR/SEM excluidos del ranking (estan en archivo/baselines_espaciales/ como
referencia econometrica; usan y_train directamente en prediccion lo que les
otorga ventaja estructural no comparable con los modelos de regresion).

Genera:
  analisis/output_log/comparativo_cv.csv        — CV summary todos los modelos
  analisis/output_log/comparativo_holdout.csv   — holdout todos los modelos
  analisis/output_log/comparativo_moran.csv     — Moran I todos los modelos
  analisis/output_log/dm_test_log.csv           — Diebold-Mariano entre modelos
  analisis/output_log/ranking_log.png           — figura comparativa
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

BASE_DIR = Path(__file__).parent
ROOT     = BASE_DIR.parent
OUT_DIR  = BASE_DIR / "output_log"
OUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Configuracion de modelos
# =============================================================================

MODELS = [
    {"name": "OLS",       "dir": ROOT/"modelos"/"ols",       "prefix": "ols_log",       "outdir": "output_log"},
    {"name": "MLP",       "dir": ROOT/"modelos"/"mlp",       "prefix": "mlp_log",       "outdir": "output_log"},
    {"name": "GWR-17",    "dir": ROOT/"modelos"/"gwr",       "prefix": "gwr_log",       "outdir": "output_log"},
    {"name": "GWR-27",    "dir": ROOT/"modelos"/"gwr",       "prefix": "gwr27_log",     "outdir": "output_log_27vars"},
    {"name": "GNNWR",     "dir": ROOT/"modelos"/"gnnwr",     "prefix": "gnnwr_log",     "outdir": "output_log"},
    {"name": "SANNWR",    "dir": ROOT/"modelos"/"sannwr",    "prefix": "sannwr_real_log", "outdir": "output_log_real"},  # canonico (Ni 2022) = principal
    {"name": "SANNWR*",   "dir": ROOT/"modelos"/"sannwr",    "prefix": "sannwr_log",      "outdir": "output_log"},      # variante grilla = anexo
    {"name": "GSAWR",     "dir": ROOT/"modelos"/"gsawr",     "prefix": "gsawr_log",     "outdir": "output_log"},
]

# Modelos en la tabla principal (SAR/SEM excluidos por usar y_train en prediccion;
# GWR-17 excluido por multicolinealidad — va a anexo como analisis de sensibilidad).
PRINCIPAL_MODELS = {"OLS", "MLP", "GWR-27", "GNNWR", "SANNWR"}
ANEXO_MODELS     = {"GWR-17", "GSAWR", "SANNWR*"}

def load_results(m: dict) -> tuple[pd.DataFrame|None, pd.DataFrame|None, pd.DataFrame|None]:
    out = m["dir"] / m.get("outdir", "output_log")
    p   = m["prefix"]
    def safe_read(path):
        if path.exists():
            df = pd.read_csv(path)
            df["modelo_display"] = m["name"]
            return df
        return None
    summary  = safe_read(out / f"{p}_summary.csv")
    holdout  = safe_read(out / f"{p}_holdout.csv")
    moran    = safe_read(out / f"{p}_moran.csv")
    return summary, holdout, moran

# =============================================================================
# Cargar todos los resultados
# =============================================================================

all_summary  = []
all_holdout  = []
all_moran    = []
all_results  = []   # fold-level (para DM test)

for m in MODELS:
    summary, holdout, moran = load_results(m)
    if summary is not None:
        all_summary.append(summary)
        print(f"[OK] {m['name']:12s} summary")
    else:
        print(f"[--] {m['name']:12s} summary NOT FOUND")

    if holdout is not None:
        all_holdout.append(holdout)
        print(f"[OK] {m['name']:12s} holdout")
    else:
        print(f"[--] {m['name']:12s} holdout NOT FOUND")

    if moran is not None:
        all_moran.append(moran)

    # fold-level para DM test
    results_path = m["dir"] / m.get("outdir","output_log") / f"{m['prefix']}_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        df["modelo_display"] = m["name"]
        all_results.append(df)

# =============================================================================
# CV summary
# =============================================================================

if all_summary:
    cv_df = pd.concat(all_summary, ignore_index=True)
    # Reordenar columnas de forma consistente
    cols_order = ["modelo_display","modelo","estrategia",
                  "MAE_mean","MAE_std","RMSE_mean","RMSE_std",
                  "MAPE_mean","MAPE_std","R2_mean","R2_std"]
    cols_present = [c for c in cols_order if c in cv_df.columns] + \
                   [c for c in cv_df.columns if c not in cols_order]
    cv_df = cv_df[cols_present]
    cv_df.to_csv(OUT_DIR/"comparativo_cv.csv", index=False)
    cv_df[cv_df["modelo_display"].isin(PRINCIPAL_MODELS)].to_csv(
        OUT_DIR/"comparativo_cv_principal.csv", index=False)
    cv_df[cv_df["modelo_display"].isin(PRINCIPAL_MODELS | ANEXO_MODELS)].to_csv(
        OUT_DIR/"comparativo_cv_anexo.csv", index=False)
    print(f"\n[CV summary guardado] {len(cv_df)} filas")
    print(cv_df[["modelo_display","estrategia","RMSE_mean","RMSE_std","R2_mean","R2_std"]
                ].to_string(index=False))

# =============================================================================
# Holdout summary
# =============================================================================

if all_holdout:
    hdf = pd.concat(all_holdout, ignore_index=True)
    hdf.to_csv(OUT_DIR/"comparativo_holdout.csv", index=False)
    hdf[hdf["modelo_display"].isin(PRINCIPAL_MODELS)].to_csv(
        OUT_DIR/"comparativo_holdout_principal.csv", index=False)
    hdf[hdf["modelo_display"].isin(PRINCIPAL_MODELS | ANEXO_MODELS)].to_csv(
        OUT_DIR/"comparativo_holdout_anexo.csv", index=False)
    print(f"\n[Holdout summary guardado]")
    cols_show = ["modelo_display","n_test","MAE","RMSE","MAPE","R2"]
    if "R2_log" in hdf.columns:
        cols_show.append("R2_log")
    print(hdf[[c for c in cols_show if c in hdf.columns]].to_string(index=False))

# =============================================================================
# Moran I
# =============================================================================

if all_moran:
    moran_df = pd.concat(all_moran, ignore_index=True)
    moran_df.to_csv(OUT_DIR/"comparativo_moran.csv", index=False)
    moran_df[moran_df["modelo_display"].isin(PRINCIPAL_MODELS)].to_csv(
        OUT_DIR/"comparativo_moran_principal.csv", index=False)
    moran_df[moran_df["modelo_display"].isin(PRINCIPAL_MODELS | ANEXO_MODELS)].to_csv(
        OUT_DIR/"comparativo_moran_anexo.csv", index=False)
    print(f"\n[Moran I guardado]")
    print(moran_df.to_string(index=False))

RMSE_FAIL_THRESHOLD = 50_000.0   # umbral colapso numerico (folds patologicos)

# =============================================================================
# Diebold-Mariano test sobre predicciones INDIVIDUALES holdout (n=1,011)
# Referencia: Diebold & Mariano (1995). Comparing Predictive Accuracy. JBES.
# d_i = (e_A_i)^2 - (e_B_i)^2; t-test bilateral con df=n-1.
# Limitacion: d_i puede tener autocorrelacion espacial residual; una correccion
# HAC/Conley haria los p-valores mayores, reforzando la conclusion de no-significancia.
# =============================================================================

def dm_test_individual(e_a: np.ndarray, e_b: np.ndarray) -> tuple[float, float]:
    """
    DM test sobre errores cuadraticos individuales holdout.
    d_i = (e_A_i)^2 - (e_B_i)^2, donde e_i = y_obs_i - y_pred_i (escala original).
    H0: E[d] = 0 (igual precision predictiva).
    Estadistico t con df=n-1; p bilateral.
    """
    d = e_a**2 - e_b**2
    n = len(d)
    std_d = float(d.std(ddof=1))
    t = float(d.mean() / (std_d / np.sqrt(n))) if std_d > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), df=n-1))
    return round(t, 4), round(p, 4)

print("\n" + "="*70)
print("TEST PAREADO (nombre historico DM; t-test exploratorio, NO Diebold-Mariano espacial) — holdout (n=1,011)")
print("ADVERTENCIA: los residuos estan espacialmente autocorrelacionados; n grande NO corrige la dependencia. P-valores indicativos, no inferencia espacial valida.")
print("="*70)

preds_holdout = {}   # {modelo_display: Series(error_orig) indexed by predio_join}
for m in MODELS:
    out = m["dir"] / m.get("outdir", "output_log")
    pred_path = out / f"{m['prefix']}_predictions.csv"
    if not pred_path.exists():
        print(f"[--] {m['name']}: predicciones no encontradas"); continue
    df_p = pd.read_csv(pred_path)
    test_df = df_p[df_p["split"] == "test"].copy()
    if len(test_df) == 0:
        print(f"[--] {m['name']}: sin filas de test"); continue
    y_obs  = np.exp(test_df["y_obs_log"].values)
    y_pred = np.exp(test_df["y_pred_log"].values)
    e = y_obs - y_pred
    preds_holdout[m["name"]] = pd.Series(e, index=test_df["predio_join"].astype(int).values)
    print(f"  [OK] {m['name']:8s}  n={len(e)}  RMSE={float(np.sqrt(np.mean(e**2))):.2f}")

dm_rows = []
modelos_dm = list(preds_holdout.keys())
for i, m_a in enumerate(modelos_dm):
    for m_b in modelos_dm[i+1:]:
        idx = preds_holdout[m_a].index.intersection(preds_holdout[m_b].index)
        if len(idx) < 10: continue
        e_a = preds_holdout[m_a][idx].values
        e_b = preds_holdout[m_b][idx].values
        t_stat, p_val = dm_test_individual(e_a, e_b)
        dm_rows.append({
            "estrategia":  "Holdout20%",
            "n_obs":       len(idx),
            "modelo_A":    m_a,
            "modelo_B":    m_b,
            "RMSE_A_mean": round(float(np.sqrt(np.mean(e_a**2))), 4),
            "RMSE_B_mean": round(float(np.sqrt(np.mean(e_b**2))), 4),
            "DM_t":        t_stat,
            "DM_p":        p_val,
            "significativo": "★" if p_val < 0.05 else "",
        })

if dm_rows:
    dm_df = pd.DataFrame(dm_rows)
    # Holm-Bonferroni correction (Holm 1979) sobre todos los pares
    sorted_idx = dm_df["DM_p"].argsort().values
    m_tests = len(dm_df)
    p_holm = np.ones(m_tests)
    running_max = 0.0
    for rank, idx in enumerate(sorted_idx):
        p_adj = dm_df["DM_p"].iloc[idx] * (m_tests - rank)
        running_max = max(running_max, p_adj)
        p_holm[idx] = min(running_max, 1.0)
    dm_df["DM_p_holm"] = p_holm.round(4)
    dm_df["sig_holm"] = dm_df["DM_p_holm"].apply(lambda p: "★" if p < 0.05 else "")
    dm_df.to_csv(OUT_DIR/"dm_test_log.csv", index=False)
    dm_principal = dm_df[
        dm_df["modelo_A"].isin(PRINCIPAL_MODELS) &
        dm_df["modelo_B"].isin(PRINCIPAL_MODELS)
    ].copy()
    dm_principal.to_csv(OUT_DIR/"dm_test_principal.csv", index=False)
    n_sig = (dm_df["DM_p"] < 0.05).sum()
    n_sig_holm = (dm_df["DM_p_holm"] < 0.05).sum()
    print(f"\n[DM test guardado] {len(dm_df)} pares  n={dm_df['n_obs'].iloc[0]}")
    print(f"  Significativos p<0.05: {n_sig}  |  tras Holm-Bonferroni: {n_sig_holm}")
    print("\nDM test principales — Holdout 20% (ordenado por p-valor):")
    print(dm_principal[["modelo_A","modelo_B","RMSE_A_mean","RMSE_B_mean",
                         "DM_t","DM_p","DM_p_holm","sig_holm"]].sort_values("DM_p").to_string(index=False))

# =============================================================================
# Figura comparativa: holdout + CV por estrategia
# =============================================================================

def make_ranking_figure():
    if not all_holdout:
        return

    hdf = pd.concat(all_holdout, ignore_index=True)
    # Ordenar por RMSE holdout
    order_df = hdf.sort_values("RMSE")
    modelos_sorted = order_df["modelo_display"].tolist()

    # Colores por categoria
    color_map = {
        "OLS":       "#9E9E9E",
        "MLP":       "#00897B",
        "GWR-17":    "#90A4AE",
        "GWR-27":    "#607D8B",
        "GNNWR":     "#2196F3",
        "SANNWR":    "#4CAF50",
        "GSAWR":     "#9C27B0",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    metrics = [
        ("RMSE", "RMSE (USD/m2)", "RMSE mas bajo = mejor"),
        ("MAE",  "MAE (USD/m2)",  "MAE mas bajo = mejor"),
        ("R2",   "R2 (escala original)", "R2 mas alto = mejor"),
    ]

    for ax, (metric, ylabel, note) in zip(axes, metrics):
        vals = [float(hdf[hdf["modelo_display"]==m][metric].iloc[0])
                if m in hdf["modelo_display"].values else np.nan
                for m in modelos_sorted]
        colors = [color_map.get(m, "#333") for m in modelos_sorted]
        bars   = ax.barh(range(len(modelos_sorted)), vals, color=colors, alpha=0.85,
                         edgecolor="black", linewidth=0.5)

        # Etiqueta valor
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if not np.isnan(v):
                ax.text(v * (1.01 if metric != "R2" else 1.01),
                        bar.get_y() + bar.get_height()/2,
                        f"{v:.2f}", va="center", ha="left", fontsize=8)

        ax.set_yticks(range(len(modelos_sorted)))
        ax.set_yticklabels(modelos_sorted, fontsize=9)
        ax.set_xlabel(ylabel, fontsize=10)
        ax.set_title(f"{metric} — Holdout 20%", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.invert_yaxis()

    fig.suptitle(
        "Comparativo modelos — log(valor_m2) + holdout 80/20\n"
        "DMQ Quito  |  n_test=1,011  |  CRS: EPSG:32717",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "ranking_log.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Figura] {out_path}")

make_ranking_figure()

# =============================================================================
# Figura CV: RKF vs SB por modelo
# =============================================================================

def make_cv_figure():
    if not all_summary:
        return

    cv_df = pd.concat(all_summary, ignore_index=True)
    rkf = cv_df[cv_df["estrategia"]=="RandomKFold"].copy()
    sb  = cv_df[cv_df["estrategia"]=="SpatialBlock"].copy()

    modelos = rkf["modelo_display"].tolist() if len(rkf) else []
    if not modelos:
        return

    x      = np.arange(len(modelos))
    width  = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, metric in zip(axes, ["RMSE", "R2"]):
        rkf_means = [float(rkf[rkf["modelo_display"]==m][f"{metric}_mean"].iloc[0])
                     if m in rkf["modelo_display"].values else np.nan for m in modelos]
        sb_means  = [float(sb[sb["modelo_display"]==m][f"{metric}_mean"].iloc[0])
                     if m in sb["modelo_display"].values else np.nan for m in modelos]
        rkf_stds  = [float(rkf[rkf["modelo_display"]==m][f"{metric}_std"].iloc[0])
                     if m in rkf["modelo_display"].values else np.nan for m in modelos]
        sb_stds   = [float(sb[sb["modelo_display"]==m][f"{metric}_std"].iloc[0])
                     if m in sb["modelo_display"].values else np.nan for m in modelos]

        b1 = ax.bar(x - width/2, rkf_means, width, yerr=rkf_stds, capsize=4,
                    color="#2196F3", alpha=0.8, label="RandomKFold",
                    edgecolor="black", linewidth=0.5,
                    error_kw={"elinewidth":1.0})
        b2 = ax.bar(x + width/2, sb_means,  width, yerr=sb_stds,  capsize=4,
                    color="#FF5722", alpha=0.8, label="SpatialBlock",
                    edgecolor="black", linewidth=0.5,
                    error_kw={"elinewidth":1.0})
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(f"{metric} (media 5 folds)", fontsize=10)
        ax.set_title(f"{metric} — CV RandomKFold vs SpatialBlock", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle(
        "Validacion cruzada — log(valor_m2) + holdout 80/20\n"
        "DMQ Quito  |  5 folds  |  CRS: EPSG:32717",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "cv_rkf_vs_sb_log.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figura CV] {out_path}")

make_cv_figure()

# =============================================================================
# ANALISIS ADICIONALES: Delta Metrica, Wilcoxon, Eficiencia, BufferedLOO demo
# =============================================================================

# Tiempos de entrenamiento (min) — escenario log + holdout, GPU MX150 CUDA 12.6
TIEMPOS_MIN = {          # GPU local NVIDIA MX150 4GB (desarrollo)
    "OLS":        2.0,    # sklearn LinearRegression, trivial
    "MLP":        5.0,    # PyTorch MLP 27→[256,128,64]→1
    "GWR-17":   240.0,    # BW selection por fold, muy lento (CPU)
    "GWR-27":   240.0,    # idem (CPU)
    "GNNWR":     23.0,    # ~136s × 10 folds
    "SANNWR":    45.0,
    "GSAWR":     32.0,    # ~190s × 10 folds
}
TIEMPOS_MIN_GPU_POD = {  # NVIDIA RTX 4090 24GB (RunPod) — seed=42 holdout
    "OLS":        2.0,    # CPU, igual
    "MLP":        1.0,    # ~62s
    "GWR-17":   240.0,    # CPU, igual
    "GWR-27":   240.0,    # CPU, igual
    "GNNWR":      7.0,    # ~407s
    "SANNWR":     4.0,    # ~215s
    "GSAWR":      7.0,    # ~412s
}


def _wilcoxon_test(errors_a: np.ndarray, errors_b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test no parametrico (complemento al DM test)."""
    d = errors_a - errors_b
    if np.all(d == 0) or len(d) < 2:
        return float("nan"), float("nan")
    try:
        w_stat, p_val = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")
        return float(w_stat), float(p_val)
    except Exception:
        return float("nan"), float("nan")


def _eficiencia(modelo: str, baseline: str, rmse_m: float, rmse_b: float,
                tiempos: dict | None = None) -> float:
    """E = (1 - RMSE_m/RMSE_b) / (T_m/T_b). E>1 => ganancia justifica costo."""
    if tiempos is None:
        tiempos = TIEMPOS_MIN
    if rmse_b == 0 or modelo not in tiempos or baseline not in tiempos:
        return float("nan")
    delta_err  = 1.0 - rmse_m / rmse_b
    delta_time = tiempos[modelo] / max(tiempos[baseline], 0.1)
    return round(float(delta_err / delta_time), 4) if delta_time > 0 else float("nan")


# --- A. Tabla Delta Metrica ---------------------------------------------------
print("\n" + "="*70)
print("TABLA DELTA METRICA — sesgo de validacion (H2 tesis)")
print("="*70)
try:
    cv_raw = pd.read_csv(OUT_DIR / "comparativo_cv.csv")
    rkf = (cv_raw[cv_raw["estrategia"] == "RandomKFold"]
           .set_index("modelo_display")[["RMSE_mean", "R2_mean"]]
           .rename(columns={"RMSE_mean": "RMSE_RKF", "R2_mean": "R2_RKF"}))
    sb  = (cv_raw[cv_raw["estrategia"] == "SpatialBlock"]
           .set_index("modelo_display")[["RMSE_mean", "R2_mean"]]
           .rename(columns={"RMSE_mean": "RMSE_SB", "R2_mean": "R2_SB"}))
    delta = rkf.join(sb, how="outer")

    # GWR-17 conservado en delta pese a alta varianza CV (antes se excluia como
    # "colapso numerico"; con BW por fold sin leakage el resultado es honesto
    # y documenta el costo de la seleccion VIF-clean).

    # (exclusion de SANNWR SB eliminada: el colapso RMSE=57k era de la version
    # antigua sin clip_grad; con la config corregida SANNWR SB=94.69, limpio)

    delta["delta_RMSE_abs"]  = (delta["RMSE_SB"] - delta["RMSE_RKF"]).round(2)
    delta["delta_RMSE_rel%"] = (delta["delta_RMSE_abs"] / delta["RMSE_RKF"] * 100).round(1)
    delta["delta_R2"]        = (delta["R2_SB"] - delta["R2_RKF"]).round(4)
    delta = delta.reset_index()
    delta.to_csv(OUT_DIR / "delta_metrica_bias_validacion.csv", index=False)
    print("[OK] delta_metrica_bias_validacion.csv guardado")
    print(delta[["modelo_display", "RMSE_RKF", "RMSE_SB", "delta_RMSE_rel%",
                 "R2_RKF", "R2_SB", "delta_R2"]].to_string(index=False))
except Exception as e:
    print(f"[WARN] delta_metrica: {e}")
    cv_raw = None

# --- B. Wilcoxon test pairwise -----------------------------------------------
print("\n" + "="*70)
print("WILCOXON TEST — comparacion no parametrica pairwise")
print("="*70)
try:
    if len(all_results) > 1:
        res_full  = pd.concat(all_results, ignore_index=True)
        n_dropped_w = (res_full["RMSE"] >= RMSE_FAIL_THRESHOLD).sum()
        if n_dropped_w: print(f"[FILTER] Wilcoxon: {n_dropped_w} folds con RMSE >= {RMSE_FAIL_THRESHOLD:.0f} excluidos")
        res_valid = res_full[res_full["RMSE"] < RMSE_FAIL_THRESHOLD].copy()
        modelos_w = res_valid["modelo_display"].unique().tolist()
        strats_w  = res_valid["estrategia"].unique().tolist()
        w_rows = []
        for strat in strats_w:
            sub = res_valid[res_valid["estrategia"] == strat]
            for i, m_a in enumerate(modelos_w):
                for m_b in modelos_w[i+1:]:
                    g_a = sub[sub["modelo_display"]==m_a].sort_values("fold")["RMSE"].values
                    g_b = sub[sub["modelo_display"]==m_b].sort_values("fold")["RMSE"].values
                    if len(g_a) == len(g_b) and len(g_a) >= 2:
                        w_stat, p_val = _wilcoxon_test(g_a, g_b)
                        w_rows.append({
                            "estrategia": strat,
                            "modelo_A": m_a, "modelo_B": m_b,
                            "RMSE_A_mean": round(float(g_a.mean()), 2),
                            "RMSE_B_mean": round(float(g_b.mean()), 2),
                            "W_stat": round(w_stat, 2) if not np.isnan(w_stat) else np.nan,
                            "W_p":    round(p_val, 4)  if not np.isnan(p_val)  else np.nan,
                            "significativo": "★" if (not np.isnan(p_val) and p_val < 0.05) else "",
                        })
        if w_rows:
            w_df = pd.DataFrame(w_rows)
            w_df.to_csv(OUT_DIR / "wilcoxon_test_log.csv", index=False)
            print(f"[OK] wilcoxon_test_log.csv guardado ({len(w_df)} pares)")
            print("NOTA: Con K=5 folds el p-valor minimo de Wilcoxon es 0.0625 (sin poder para alpha=0.05).")
            print("      Usar DM test (parametrico) como prueba principal; Wilcoxon es complemento.")
            ols_w = w_df[w_df["modelo_A"] == "OLS"]
            if len(ols_w):
                print("Wilcoxon vs OLS:")
                print(ols_w[["estrategia","modelo_B","RMSE_A_mean","RMSE_B_mean",
                              "W_stat","W_p","significativo"]].to_string(index=False))
    else:
        print("[WARN] No hay suficientes resultados fold-level para Wilcoxon")
except Exception as e:
    print(f"[WARN] Wilcoxon: {e}")

# --- C. Indice de eficiencia --------------------------------------------------
print("\n" + "="*70)
print("INDICE DE EFICIENCIA — ganancia predictiva vs costo temporal")
print("="*70)
try:
    h_df   = pd.read_csv(OUT_DIR / "comparativo_holdout.csv")
    rmse_h = h_df.set_index("modelo_display")["RMSE"].to_dict()
    rmse_ols   = rmse_h.get("OLS")
    rmse_gwr27 = rmse_h.get("GWR-27")
    eff_rows = []
    for modelo in TIEMPOS_MIN:
        rmse_m = rmse_h.get(modelo)
        rm = rmse_m or float("nan")
        row = {
            "modelo":             modelo,
            "RMSE_holdout":       round(rmse_m, 2) if rmse_m is not None else np.nan,
            "tiempo_local_min":   TIEMPOS_MIN[modelo],
            "tiempo_gpu_pod_min": TIEMPOS_MIN_GPU_POD.get(modelo, float("nan")),
            "E_vs_OLS_local":     _eficiencia(modelo, "OLS",    rm, rmse_ols   or float("nan"), TIEMPOS_MIN),
            "E_vs_OLS_gpu_pod":   _eficiencia(modelo, "OLS",    rm, rmse_ols   or float("nan"), TIEMPOS_MIN_GPU_POD),
            "E_vs_GWR27_local":   _eficiencia(modelo, "GWR-27", rm, rmse_gwr27 or float("nan"), TIEMPOS_MIN),
            "E_vs_GWR27_gpu_pod": _eficiencia(modelo, "GWR-27", rm, rmse_gwr27 or float("nan"), TIEMPOS_MIN_GPU_POD),
        }
        eff_rows.append(row)
    eff_df = pd.DataFrame(eff_rows)
    # Columna de interpretacion: E>0 => ganancia vs baseline; E<0 => peor que baseline
    eff_df["interpretacion_vs_GWR27"] = eff_df["E_vs_GWR27_local"].apply(
        lambda e: "mejor_que_GWR27" if e > 0 else ("igual" if e == 0 else "peor_que_GWR27"))
    eff_df.to_csv(OUT_DIR / "indice_eficiencia.csv", index=False)
    print("[OK] indice_eficiencia.csv guardado (incluye columnas local y GPU pod RTX 4090)")
    print("     E>0: modelo mejor que baseline ajustado por tiempo; E<0: peor o mas costoso")
    print(eff_df.to_string(index=False))
except Exception as e:
    print(f"[WARN] eficiencia: {e}")

# --- C2. T-test pareado sobre SpatialBlock CV replicas (n=5 seeds) -----------
print("\n" + "="*70)
print("T-TEST PAREADO SpatialBlock — CV replicas (5 seeds RTX 4090)")
print("Prueba formal de diferencias en extrapolacion espacial.")
print("="*70)
try:
    sb_models = {}
    # SANNWR = canonico (Ni 2022, output_log_real) -> es el modelo core; antes este
    # t-test usaba sannwr_log (SANNWR*, variante de grilla) bajo la etiqueta "SANNWR".
    # Corregido tras la corrida del pod (RTX4090) que dio al canonico replicas CV propias.
    REPLICA_FILES = {
        "MLP":     ROOT/"modelos"/"mlp"/"output_log"/"mlp_log_cv_replicas_seed.csv",
        "GNNWR":   ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_cv_replicas_seed.csv",
        "SANNWR":  ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_cv_replicas_seed.csv",
        "GSAWR":   ROOT/"modelos"/"gsawr"/"output_log"/"gsawr_log_cv_replicas_seed.csv",
    }
    for modelo, path in REPLICA_FILES.items():
        if path.exists():
            df_r = pd.read_csv(path)
            sb_vals = df_r[df_r["estrategia"]=="SpatialBlock"]["RMSE_mean"].values
            sb_models[modelo] = sb_vals
            print(f"  {modelo:8s} SB RMSE/seed: {sb_vals.round(2)}  mean={sb_vals.mean():.2f}+-{sb_vals.std(ddof=1):.2f}")
        else:
            print(f"  [--] {modelo}: CV replica seed CSV no encontrado")

    sb_pairs = [
        ("GNNWR","SANNWR"),("GNNWR","GSAWR"),("GNNWR","MLP"),
        ("SANNWR","GSAWR"),("SANNWR","MLP"),("GSAWR","MLP"),
    ]
    sb_rows = []
    for a, b in sb_pairs:
        if a not in sb_models or b not in sb_models: continue
        ea, eb = sb_models[a], sb_models[b]
        t_stat, p_val = stats.ttest_rel(ea, eb)
        sb_rows.append({
            "estrategia": "SpatialBlock",
            "n_seeds":    len(ea),
            "modelo_A":   a,
            "modelo_B":   b,
            "RMSE_A_mean": round(float(ea.mean()),4),
            "RMSE_B_mean": round(float(eb.mean()),4),
            "diff_mean":   round(float(ea.mean()-eb.mean()),4),
            "t_stat":      round(float(t_stat),4),
            "p_value":     round(float(p_val),4),
            "significativo": "si" if p_val < 0.05 else "no",
        })
    if sb_rows:
        sb_df = pd.DataFrame(sb_rows)
        # Holm-Bonferroni correction sobre los m pares (Holm 1979)
        m_tests = len(sb_df)
        sorted_idx = sb_df["p_value"].argsort().values
        p_holm = np.ones(m_tests)
        running_max = 0.0
        for rank, idx in enumerate(sorted_idx):
            p_adj = sb_df["p_value"].iloc[idx] * (m_tests - rank)
            running_max = max(running_max, p_adj)
            p_holm[idx] = min(running_max, 1.0)
        sb_df["p_holm"] = p_holm.round(4)
        sb_df["sig_holm"] = sb_df["p_holm"].apply(lambda p: "si" if p < 0.05 else "no")
        sb_df.to_csv(OUT_DIR/"ttest_spatialblock_replicas.csv", index=False)
        print(f"\n[T-test SpatialBlock guardado] {len(sb_df)} pares  (n=5 seeds)")
        print(sb_df[["modelo_A","modelo_B","RMSE_A_mean","RMSE_B_mean","t_stat","p_value","p_holm","sig_holm"]].to_string(index=False))
        n_sig_raw  = (sb_df["p_value"] < 0.05).sum()
        n_sig_holm = (sb_df["p_holm"]  < 0.05).sum()
        print(f"\n  Significativos p<0.05: {n_sig_raw}  |  tras Holm-Bonferroni: {n_sig_holm}")
        print("  Nota: n=5 seeds limita la potencia; tendencias son indicativas, no definitivas.")
except Exception as e:
    print(f"[WARN] t-test SpatialBlock: {e}")

# --- D. BufferedLOO demo — solo OLS ------------------------------------------
print("\n" + "="*70)
print("BUFFERLOO DEMO — OLS con buffer variograma (demostracion metodologica)")
print("="*70)
def _run_bufferloo_ols():
    try:
        import geopandas as gpd
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        _cv_dir = ROOT / "spatial_cv"
        if str(_cv_dir) not in sys.path:
            sys.path.insert(0, str(_cv_dir))
        from estrategias_cv import BufferedLOO

        COVARIABLES_OLS = [
            "suscept_codigo", "pc_pnbi", "dist_metro", "dist_centr_metro",
            "dist_centr_zonal", "dist_cc", "dist_universidad", "dist_hospital",
            "dist_parque_metro", "dist_industrial", "dist_via_principal",
            "uso_suelo_cod", "cos_num", "dist_quebrada", "dist_mercado_mayorista",
            "dist_plataforma_gub", "log_area", "frente_m", "area_const_m2",
            "tiene_const", "num_pisos", "antiguedad", "topografia_factor",
            "conservacion_cod", "acabados_cod", "es_ph", "pendiente_grados",
        ]
        DATA_PATH_  = ROOT / "datos" / "dataset.gpkg"
        SPLIT_PATH_ = ROOT / "data_split" / "split.csv"

        gdf_      = gpd.read_file(DATA_PATH_, layer="puntos_mercado").to_crs(epsg=32717)
        split_df_ = pd.read_csv(SPLIT_PATH_)
        split_df_["predio_join"] = split_df_["predio_join"].astype(int)
        gdf_["predio_join"]      = gdf_["predio_join"].astype(int)
        gdf_ = gdf_.merge(split_df_[["predio_join","split"]], on="predio_join", how="left")

        tr_gdf = gdf_[gdf_["split"] == "train"].copy().reset_index(drop=True)
        X_tr   = tr_gdf[COVARIABLES_OLS].values.astype(float)
        y_log  = np.log(tr_gdf["valor_m2"].values.astype(float))
        coords = np.column_stack([tr_gdf.geometry.x, tr_gdf.geometry.y])

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X_tr)

        bloo   = BufferedLOO(coords=coords, buffer_dist=5621.8)
        n_iter = bloo.get_n_splits()
        print(f"  Corriendo {n_iter} iteraciones LOO con buffer=5621.8m ...")
        import time as _time
        t0 = _time.time()
        rmse_list = []
        for idx, (tr_idx, te_idx) in enumerate(bloo.split(X_s)):
            if len(tr_idx) < len(COVARIABLES_OLS) + 1:
                continue
            m = LinearRegression()
            m.fit(X_s[tr_idx], y_log[tr_idx])
            yp = np.exp(m.predict(X_s[te_idx]))
            yt = np.exp(y_log[te_idx])
            rmse_list.append(float(np.sqrt(np.mean((yt - yp)**2))))
            if (idx + 1) % 1000 == 0:
                elapsed = _time.time() - t0
                print(f"    {idx+1}/{n_iter} — elapsed {elapsed:.0f}s — "
                      f"running RMSE={np.mean(rmse_list):.2f}")

        rmse_bloo = float(np.mean(rmse_list)) if rmse_list else float("nan")

        # Recuperar RMSE de RKF y SB para OLS desde CSV
        rmse_rkf_ols = rmse_sb_ols = float("nan")
        try:
            _cv = pd.read_csv(OUT_DIR / "comparativo_cv.csv")
            _ols = _cv[_cv["modelo_display"] == "OLS"]
            rmse_rkf_ols = float(_ols[_ols["estrategia"]=="RandomKFold"]["RMSE_mean"].iloc[0])
            rmse_sb_ols  = float(_ols[_ols["estrategia"]=="SpatialBlock"]["RMSE_mean"].iloc[0])
        except Exception:
            pass

        demo_df = pd.DataFrame([{
            "modelo":            "OLS",
            "RMSE_RandomKFold":  round(rmse_rkf_ols, 2),
            "RMSE_SpatialBlock": round(rmse_sb_ols, 2),
            "RMSE_BufferedLOO":  round(rmse_bloo, 2),
            "n_iter_validas":    len(rmse_list),
            "buffer_m":          5621.8,
            "nota": "BufferedLOO excluye vecinos dentro del rango del semivariograma",
        }])
        demo_df.to_csv(OUT_DIR / "ols_bufferloo_demo.csv", index=False)
        print(f"[OK] ols_bufferloo_demo.csv guardado")
        print(demo_df.to_string(index=False))
    except Exception as e:
        import traceback
        print(f"[WARN] BufferedLOO demo: {e}")
        traceback.print_exc()

_run_bufferloo_ols()

print(f"\n[Listo] Resultados en {OUT_DIR}")
