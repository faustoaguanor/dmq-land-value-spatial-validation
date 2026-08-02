"""
smearing_bootstrap_cdf.py
=========================
Ejecuta tres analisis sobre las predicciones por predio de los 6 modelos:

T5 — Correccion de Jensen (estimador de smearing de Duan, 1983):
     Para cada modelo calcula s_M = mean(exp(e_train)) con e_train = y_train_log - y_pred_train_log.
     Corrige predicciones holdout: y_pred_corr = exp(y_pred_log) * s_M.
     Emite `comparativo_holdout_smeared.csv` con RMSE/MAE/MAPE/R2 raw vs smeared.

T2 — Bootstrap CI al 95% sobre RMSE, MAE, MAPE, R2 del holdout para cada modelo.
     Resampling pareado con reemplazo (bootstrap estándar) de los residuos
     holdout; n_boot=1,000; seed=42.
     Emite `comparativo_holdout_ci.csv`.

T3 — CDF de errores absolutos por decil del precio de oferta observado.
     Segmenta holdout por decil y calcula MAE/RMSE/MAPE por decil y modelo.
     Emite `error_por_decil.csv` + figura `fig5_cdf_errores_por_decil.{png,pdf}`.

Todos los outputs van a analisis/output_log/ (CSVs) y figures/main_results/ (PNG/PDF).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = Path(__file__).parent.parent
OUT_DIR   = ROOT / "analisis" / "output_log"
FIGS_DIR  = ROOT / "figures" / "main_results"
OUT_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True, parents=True)

# ── Mapeo de modelos a rutas de predicciones ─────────────────────────────────
# SANNWR = canonico (Ni 2022, sannwr_real_log, output_log_real) -> principal.
# SANNWR* = variante propia de grilla+SAPDNN (sannwr_log, output_log) -> exploratorio/anexo.
# (Antes este script leia sannwr_log_predictions.csv bajo la etiqueta "SANNWR", mezclando
# silenciosamente el canonico con la variante en el smearing/bootstrap CI del core.)

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

# ── Utilidades ────────────────────────────────────────────────────────────────

def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def mae(y, yhat):  return float(np.mean(np.abs(y - yhat)))
def mape(y, yhat): return float(np.mean(np.abs((y - yhat) / y)) * 100)
def r2(y, yhat):
    ss_r = float(np.sum((y - yhat) ** 2))
    ss_t = float(np.sum((y - y.mean()) ** 2))
    return float(1 - ss_r/ss_t) if ss_t > 0 else float("nan")

def load_predictions():
    rows = []
    for name, path in MODEL_SOURCES:
        if not path.exists():
            print(f"[WARN] {name:10s} predicciones NO disponibles: {path}")
            continue
        df = pd.read_csv(path)
        df["modelo"] = name
        # Asegurar tipos
        df["y_obs_log"]  = df["y_obs_log"].astype(float)
        df["y_pred_log"] = df["y_pred_log"].astype(float)
        rows.append(df)
        print(f"[OK]   {name:10s} n={len(df):,}  test={len(df[df['split']=='test']):,}  train={len(df[df['split']=='train']):,}")
    return pd.concat(rows, ignore_index=True) if rows else None

# ── T5: Correccion de smearing de Duan ───────────────────────────────────────

def smearing_factor(df_train):
    """s_M = mean(exp(e_train))  con e_train en escala log."""
    e = df_train["y_obs_log"].values - df_train["y_pred_log"].values
    return float(np.mean(np.exp(e)))

def run_t5_smearing(all_preds):
    print("\n" + "=" * 70)
    print("T5 — Correccion de Jensen (Duan 1983)")
    print("=" * 70)
    rows = []
    for modelo in all_preds["modelo"].unique():
        sub = all_preds[all_preds["modelo"] == modelo]
        df_tr = sub[sub["split"] == "train"]
        df_te = sub[sub["split"] == "test"]
        if df_tr.empty or df_te.empty:
            continue
        s_M = smearing_factor(df_tr)
        y_te_obs  = np.exp(df_te["y_obs_log"].values)
        y_te_raw  = np.exp(df_te["y_pred_log"].values)
        y_te_corr = y_te_raw * s_M
        rows.append({
            "modelo":         modelo,
            "smearing_factor": round(s_M, 6),
            "n_test":         len(df_te),
            "RMSE_raw":       round(rmse(y_te_obs, y_te_raw), 4),
            "RMSE_smeared":   round(rmse(y_te_obs, y_te_corr), 4),
            "MAE_raw":        round(mae(y_te_obs, y_te_raw), 4),
            "MAE_smeared":    round(mae(y_te_obs, y_te_corr), 4),
            "MAPE_raw":       round(mape(y_te_obs, y_te_raw), 4),
            "MAPE_smeared":   round(mape(y_te_obs, y_te_corr), 4),
            "R2_raw":         round(r2(y_te_obs, y_te_raw), 6),
            "R2_smeared":     round(r2(y_te_obs, y_te_corr), 6),
        })
    out = pd.DataFrame(rows).sort_values("RMSE_smeared").reset_index(drop=True)
    out.to_csv(OUT_DIR / "comparativo_holdout_smeared.csv", index=False)
    print(out.to_string(index=False))
    print(f"[OK] {OUT_DIR/'comparativo_holdout_smeared.csv'}")
    return out

# ── T2: Bootstrap CI ──────────────────────────────────────────────────────────

def bootstrap_metric(y_obs, y_pred, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_obs)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = metric_fn(y_obs[idx], y_pred[idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def run_t2_bootstrap_ci(all_preds, smeared_df, n_boot=1000):
    print("\n" + "=" * 70)
    print(f"T2 — Bootstrap CI 95% sobre RMSE/MAE/MAPE/R2 (n_boot={n_boot})")
    print("=" * 70)
    rows = []
    sm = smeared_df.set_index("modelo")["smearing_factor"].to_dict()
    for modelo in all_preds["modelo"].unique():
        sub_te = all_preds[(all_preds["modelo"] == modelo) & (all_preds["split"] == "test")]
        if sub_te.empty:
            continue
        y_obs  = np.exp(sub_te["y_obs_log"].values)
        s_M    = sm.get(modelo, 1.0)
        y_pred = np.exp(sub_te["y_pred_log"].values) * s_M  # usar smearing para CI
        row = {"modelo": modelo, "n_test": len(sub_te), "smearing_factor": round(s_M, 6)}
        for mname, mfn in [("RMSE", rmse), ("MAE", mae), ("MAPE", mape), ("R2", r2)]:
            point = mfn(y_obs, y_pred)
            lo, hi = bootstrap_metric(y_obs, y_pred, mfn, n_boot=n_boot, seed=42)
            row[mname] = round(point, 4)
            row[f"{mname}_ci_lo"] = round(lo, 4)
            row[f"{mname}_ci_hi"] = round(hi, 4)
        rows.append(row)
        print(f"[OK] {modelo:10s} RMSE={row['RMSE']:.2f} [{row['RMSE_ci_lo']:.2f}, {row['RMSE_ci_hi']:.2f}]")
    out = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    out.to_csv(OUT_DIR / "comparativo_holdout_ci.csv", index=False)
    print(f"[OK] {OUT_DIR/'comparativo_holdout_ci.csv'}")
    return out

# ── T3: CDF por decil ────────────────────────────────────────────────────────

def run_t3_cdf_por_decil(all_preds, smeared_df):
    print("\n" + "=" * 70)
    print("T3 — CDF de errores por decil del precio de oferta")
    print("=" * 70)
    sm = smeared_df.set_index("modelo")["smearing_factor"].to_dict()

    tables = []
    for modelo in all_preds["modelo"].unique():
        sub_te = all_preds[(all_preds["modelo"] == modelo) & (all_preds["split"] == "test")].copy()
        if sub_te.empty:
            continue
        y_obs  = np.exp(sub_te["y_obs_log"].values)
        s_M    = sm.get(modelo, 1.0)
        y_pred = np.exp(sub_te["y_pred_log"].values) * s_M
        sub_te["y_obs"]  = y_obs
        sub_te["y_pred"] = y_pred
        sub_te["abs_err"] = np.abs(y_obs - y_pred)
        # Decil por valor observado — usar los mismos cortes para todos los modelos
        # (usar OLS como referencia de corte seria consistente pero depende del orden;
        # mejor usar los deciles del propio subset, cada modelo mide la misma muestra test).
        sub_te["decil"] = pd.qcut(y_obs, q=10, labels=False, duplicates="drop") + 1
        for d, g in sub_te.groupby("decil"):
            tables.append({
                "modelo": modelo,
                "decil":  int(d),
                "n":      len(g),
                "val_min": round(float(g["y_obs"].min()), 2),
                "val_max": round(float(g["y_obs"].max()), 2),
                "MAE":    round(float(g["abs_err"].mean()), 4),
                "RMSE":   round(float(np.sqrt((g["abs_err"] ** 2).mean())), 4),
                "MAPE":   round(float((g["abs_err"] / g["y_obs"]).mean() * 100), 4),
            })
    out = pd.DataFrame(tables)
    out.to_csv(OUT_DIR / "error_por_decil.csv", index=False)
    print(f"[OK] {OUT_DIR/'error_por_decil.csv'}")

    # Figura: CDF del error absoluto por modelo (siete curvas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: CDF global
    ax = axes[0]
    modelos = out["modelo"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(modelos)))
    for modelo, color in zip(modelos, colors):
        sub_te = all_preds[(all_preds["modelo"] == modelo) & (all_preds["split"] == "test")]
        y_obs  = np.exp(sub_te["y_obs_log"].values)
        s_M    = sm.get(modelo, 1.0)
        y_pred = np.exp(sub_te["y_pred_log"].values) * s_M
        err = np.sort(np.abs(y_obs - y_pred))
        cdf = np.arange(1, len(err) + 1) / len(err)
        ax.plot(err, cdf, label=modelo, color=color, lw=1.8, alpha=0.85)
    ax.set_xlabel("|error| (USD/m²)", fontsize=11)
    ax.set_ylabel("CDF empírica", fontsize=11)
    ax.set_title("CDF del error absoluto — holdout 20% (smearing aplicado)",
                 fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, linestyle="--")

    # Panel 2: MAE por decil (barras agrupadas)
    ax = axes[1]
    piv = out.pivot(index="decil", columns="modelo", values="MAE")
    piv = piv[list(modelos)]  # orden
    piv.plot(kind="bar", ax=ax, width=0.85, color=colors, edgecolor="black", lw=0.3)
    ax.set_xlabel("Decil del precio de oferta observado (1=menor, 10=mayor)", fontsize=11)
    ax.set_ylabel("MAE (USD/m²)", fontsize=11)
    ax.set_title("Distribución del error (MAE) por decil del precio de oferta",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for t in ax.get_xticklabels():
        t.set_rotation(0)

    fig.suptitle("Distribución y jerarquía de errores (Holdout 20%)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS_DIR / f"fig5_cdf_errores_por_decil.{ext}",
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {FIGS_DIR/'fig5_cdf_errores_por_decil.png/pdf'}")
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_preds = load_predictions()
    if all_preds is None:
        raise SystemExit("Sin predicciones disponibles")
    smeared = run_t5_smearing(all_preds)
    _       = run_t2_bootstrap_ci(all_preds, smeared, n_boot=1000)
    _       = run_t3_cdf_por_decil(all_preds, smeared)
    print("\n[Listo]")
