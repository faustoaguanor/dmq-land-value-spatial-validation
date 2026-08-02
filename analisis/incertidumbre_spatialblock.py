"""
incertidumbre_spatialblock.py
==============================
Descompone la variabilidad del RMSE en SpatialBlock en dos fuentes:

  sigma_seed  = std de RMSE_mean entre seeds (variabilidad de inicializacion)
  sigma_fold  = media de RMSE_std entre seeds  (variabilidad geografica fold-a-fold)

Pregunta central: ¿la diferencia entre modelos es mas grande que su variabilidad interna?

Regla defensiva: si sigma_fold >> sigma_seed, la afirmacion robusta es sobre la CLASE
(pesos espaciales), no sobre el modelo individual.

Salidas: analisis/output_log/
  incertidumbre_sb.csv
  incertidumbre_sb.png / .pdf
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "analisis" / "output_log"
OUT_DIR.mkdir(exist_ok=True)

REPLICA_FILES = {
    "GNNWR":  ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_cv_replicas_seed.csv",
    "SANNWR": ROOT/"modelos"/"sannwr"/"output_log"/"sannwr_log_cv_replicas_seed.csv",
    "GSAWR":  ROOT/"modelos"/"gsawr"/"output_log"/"gsawr_log_cv_replicas_seed.csv",
    "MLP":    ROOT/"modelos"/"mlp"/"output_log"/"mlp_log_cv_replicas_seed.csv",
}
ESTRATEGIAS = ["RandomKFold", "SpatialBlock"]
ALPHA = 0.05

print("="*65)
print("DESCOMPOSICION DE INCERTIDUMBRE: sigma_seed vs sigma_fold")
print("="*65)

# ── Cargar datos ──────────────────────────────────────────────────────────────
data = {}   # modelo -> DataFrame con columnas [seed, estrategia, RMSE_mean, RMSE_std]
for modelo, path in REPLICA_FILES.items():
    if path.exists():
        data[modelo] = pd.read_csv(path)
        n_seeds = len(data[modelo]["seed"].unique())
        print(f"  {modelo:8s}: {n_seeds} seeds, estrategias: {data[modelo]['estrategia'].unique().tolist()}")
    else:
        print(f"  [--] {modelo}: no encontrado ({path.name})")

# ── Calcular sigma_seed y sigma_fold por modelo×estrategia ───────────────────
rows = []
t_crit = {}  # df -> t_crit para IC 95%

for modelo, df in data.items():
    for est in ESTRATEGIAS:
        sub = df[df["estrategia"] == est]
        if len(sub) == 0:
            continue
        rmse_seeds = sub["RMSE_mean"].values   # un valor por seed
        rmse_std_seeds = sub["RMSE_std"].values   # desv fold dentro de cada seed
        n = len(rmse_seeds)
        if n < 2:
            continue
        tc = stats.t.ppf(1 - ALPHA/2, df=n-1)
        t_crit[n] = tc
        sigma_seed = float(rmse_seeds.std(ddof=1))
        sigma_fold = float(rmse_std_seeds.mean())
        rmse_mean  = float(rmse_seeds.mean())
        ratio      = sigma_fold / sigma_seed if sigma_seed > 0 else np.nan
        ic_lo      = rmse_mean - tc * sigma_seed / np.sqrt(n)
        ic_hi      = rmse_mean + tc * sigma_seed / np.sqrt(n)
        rows.append({
            "modelo":         modelo,
            "estrategia":     est,
            "n_seeds":        n,
            "RMSE_media":     round(rmse_mean, 2),
            "sigma_seed":     round(sigma_seed, 2),
            "sigma_fold":     round(sigma_fold, 2),
            "ratio_fold_seed": round(ratio, 1),
            "IC95_lo":        round(ic_lo, 2),
            "IC95_hi":        round(ic_hi, 2),
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_DIR / "incertidumbre_sb.csv", index=False)

# ── Imprimir tabla ────────────────────────────────────────────────────────────
print("\n--- SpatialBlock ---")
sb = df_out[df_out["estrategia"] == "SpatialBlock"].sort_values("RMSE_media")
print(sb[["modelo","n_seeds","RMSE_media","sigma_seed","sigma_fold",
          "ratio_fold_seed","IC95_lo","IC95_hi"]].to_string(index=False))

print("\n--- RandomKFold ---")
rkf = df_out[df_out["estrategia"] == "RandomKFold"].sort_values("RMSE_media")
print(rkf[["modelo","n_seeds","RMSE_media","sigma_seed","sigma_fold",
           "ratio_fold_seed","IC95_lo","IC95_hi"]].to_string(index=False))

# Mensaje de interpretacion
print("\n--- Interpretacion ---")
for _, row in sb.iterrows():
    afirmacion = (
        "diferencia entre modelos posiblemente real (ratio < 5)"
        if row["ratio_fold_seed"] < 5 else
        f"variabilidad geografica domina {row['ratio_fold_seed']:.0f}x sobre la de inicializacion"
    )
    print(f"  {row['modelo']:8s}: RMSE={row['RMSE_media']:.1f}  "
          f"IC95=[{row['IC95_lo']:.1f}, {row['IC95_hi']:.1f}]  "
          f"sigma_seed={row['sigma_seed']:.1f}  sigma_fold={row['sigma_fold']:.1f}  "
          f"ratio={row['ratio_fold_seed']:.0f}x  → {afirmacion}")

# ── Figura: comparacion grafica sigma_seed vs sigma_fold ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
modelos_sb = sb["modelo"].tolist()
x = np.arange(len(modelos_sb))
width = 0.35

for ax, est in zip(axes, ["SpatialBlock", "RandomKFold"]):
    sub = df_out[df_out["estrategia"] == est].sort_values("RMSE_media")
    xs  = sub["sigma_seed"].values
    xf  = sub["sigma_fold"].values
    mods = sub["modelo"].tolist()
    xx = np.arange(len(mods))
    bars1 = ax.bar(xx - width/2, xs, width, label="σ_seed (inicializ.)", color="#1565C0", alpha=0.85)
    bars2 = ax.bar(xx + width/2, xf, width, label="σ_fold (geográfica)", color="#E53935", alpha=0.85)
    ax.set_xticks(xx); ax.set_xticklabels(mods, fontsize=10)
    ax.set_ylabel("Desviación estándar del RMSE (USD/m²)", fontsize=10)
    ax.set_title(f"Fuentes de variabilidad — {est}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, linestyle="--")
    # Anotar ratios
    for i, (xi, xfi) in enumerate(zip(xs, xf)):
        if xi > 0:
            ax.text(i, max(xi, xfi) + 1, f"{xfi/xi:.0f}×", ha="center", fontsize=9, color="black")

fig.suptitle(
    "Descomposición de incertidumbre del RMSE en SpatialBlock\n"
    "σ_seed = variabilidad entre inicializaciones (pequeña)  |  "
    "σ_fold = variabilidad geográfica (dominante)\n"
    "El número sobre cada grupo es el ratio σ_fold / σ_seed",
    fontsize=10, fontweight="bold", y=1.02,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "incertidumbre_sb.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "incertidumbre_sb.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\n[OK] incertidumbre_sb.csv / .png / .pdf guardados en {OUT_DIR}")

# ── Conclusion para la tesis ──────────────────────────────────────────────────
print("\n=== CONCLUSION PARA LA TESIS ===")
print("La variabilidad fold-a-fold (sigma_fold, geografica) domina sobre la variabilidad")
print("de inicializacion (sigma_seed) en todos los modelos bajo SpatialBlock.")
print("Implicacion metodologica: el IC construido a partir de sigma_seed (e.g., ttest")
print("entre medias de RMSE sobre 5 seeds) mide correctamente la reproducibilidad")
print("del entrenamiento pero NO es un IC para el RMSE esperado en una nueva region.")
print("La afirmacion defensible es: 'la CLASE de modelos con pesos espaciales")
print("generaliza mejor en extrapolacion', no 'GNNWR supera a SANNWR por N USD/m²'.")
