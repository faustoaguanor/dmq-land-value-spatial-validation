# -*- coding: utf-8 -*-
"""
validacion_ordenanza.py — Validación de las predicciones contra la ordenanza vigente.
======================================================================================
Compara el valor del suelo predicho por los modelos y el valor de oferta observado
(`valor_m2`) contra el VALOR OFICIAL de la ordenanza bienal de valoración del suelo del
DMQ vigente (ANIO=2025), a partir de la tabla catastral `datos/val_catas.csv`.

Valor oficial por m² (según la lógica catastral confirmada por la unidad responsable):
  - UNIPROPIEDAD:        oficial_m2 = VALORACION_TERRENO / AREA_ESCRITURA_LOTE
  - PROPIEDAD HORIZONTAL: oficial_m2 = VALORACION_TERRENO / AREA_TERRENO_PREDIO,
        donde AREA_TERRENO_PREDIO = AREA_ESCRITURA_LOTE * ALICUOTA_PARCIAL/100 cuando el
        sistema no lo calculó (AREA_TERRENO_PREDIO<=0).
Unión por número de predio (NUMERO_PREDIO == predio_join).

NO reentrena modelos: usa las predicciones de holdout ya generadas.

Salidas: analisis/output_ordenanza/
  val_ordenanza_predio.csv        (por predio: oferta, oficial, predicciones, discrepancias)
  val_ordenanza_metricas.csv      (MAE/RMSE/R2 de cada serie vs oficial)
  val_ordenanza_por_zona.csv      (discrepancia y sesgo por zona)
  val_ordenanza_equidad.csv       (sesgo por estrato NBI y por decil de valor)
  fig_obs_vs_oficial.(png/pdf)     dispersión oferta vs oficial
  fig_pred_vs_oficial.(png/pdf)    dispersión predicho vs oficial (RF/HGB/GNNWR/SANNWR)
  fig_discrepancia_mapa.(png/pdf)  mapa de discrepancia RF-oficial
  fig_equidad_estrato.(png/pdf)    sesgo (oferta-oficial) por estrato NBI y decil
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analisis" / "output_ordenanza"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "datos" / "dataset.gpkg"
VALCAT = ROOT / "datos" / "val_catas.csv"
MAPDATA = ROOT / "figures" / "aplicacion_valoracion" / "map_data_holdout.csv"
MODELS = ["RF", "HGB", "GNNWR", "SANNWR"]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 11})


def load_oficial():
    """Devuelve DataFrame [predio_join, oficial_m2, tipo] con el valor oficial por m²."""
    g = gpd.read_file(DATA, layer="puntos_mercado"); g["predio_join"] = g["predio_join"].astype(str)
    ids = set(g["predio_join"])
    cols = ["NUMERO_PREDIO", "VALOR_UNITARIO_SUELO", "ALICUOTA_PARCIAL",
            "AREA_TERRENO_PREDIO", "AREA_ESCRITURA_LOTE", "VALORACION_TERRENO",
            "PROPIEDAD_NOMBRE", "CAT_AIVA_VALOR", "ANIO"]
    vc = pd.read_csv(VALCAT, sep=";", usecols=cols, dtype=str)
    vc["NUMERO_PREDIO"] = vc["NUMERO_PREDIO"].astype(str).str.strip()
    vc = vc[vc["ANIO"].astype(str).str.strip() == "2025"].copy()
    if vc.empty:
        raise ValueError("No existen registros ANIO=2025 en val_catas.csv")
    for c in ["VALOR_UNITARIO_SUELO", "ALICUOTA_PARCIAL", "AREA_TERRENO_PREDIO",
              "AREA_ESCRITURA_LOTE", "VALORACION_TERRENO", "CAT_AIVA_VALOR"]:
        vc[c] = pd.to_numeric(vc[c].str.replace(",", ".", regex=False), errors="coerce")
    vc = vc[vc["NUMERO_PREDIO"].isin(ids)].drop_duplicates("NUMERO_PREDIO").copy()
    es_ph = vc["PROPIEDAD_NOMBRE"].str.contains("HORIZONTAL", na=False)
    # En PH la valoración registrada pertenece a la unidad según su alícuota. El
    # denominador compatible es el área de terreno imputable a esa unidad; usar el
    # lote bruto aplicaría la alícuota por segunda vez y sesgaría el USD/m² a la baja.
    area_ph = np.where(vc["AREA_TERRENO_PREDIO"] > 0,
                       vc["AREA_TERRENO_PREDIO"],
                       vc["AREA_ESCRITURA_LOTE"] * vc["ALICUOTA_PARCIAL"] / 100.0)
    ofi = np.where(es_ph, vc["VALORACION_TERRENO"] / area_ph,
                   vc["VALORACION_TERRENO"] / vc["AREA_ESCRITURA_LOTE"])
    vc["oficial_m2"] = ofi
    vc["tipo"] = np.where(es_ph, "PH", "Unipropiedad")
    vc["area_oficial_denominador"] = np.where(es_ph, area_ph, vc["AREA_ESCRITURA_LOTE"])
    n_fallback = int((es_ph & (vc["AREA_TERRENO_PREDIO"] <= 0)).sum())
    vc = vc[np.isfinite(vc["oficial_m2"]) & (vc["oficial_m2"] > 0)]
    print(f"[oficial] predios con valor oficial: {len(vc)} | PH recalculados (area=0): {n_fallback}")
    return g, vc[["NUMERO_PREDIO", "oficial_m2", "tipo", "VALOR_UNITARIO_SUELO",
                  "CAT_AIVA_VALOR", "area_oficial_denominador"]].rename(
                      columns={"NUMERO_PREDIO": "predio_join"})


def metrics(pred, ref):
    e = pred - ref
    return (float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))),
            float(1 - np.sum(e ** 2) / np.sum((ref - ref.mean()) ** 2)))


def main():
    g, ofi = load_oficial()
    # tabla completa (5.051): oferta observada vs oficial
    base = g[["predio_join", "valor_m2", "pc_pnbi"]].merge(ofi, on="predio_join", how="inner")
    base["ratio_oferta_oficial"] = base["valor_m2"] / base["oficial_m2"]

    # predicciones holdout
    md = pd.read_csv(MAPDATA); md["predio_join"] = md["predio_join"].astype(str)
    hold = md.merge(ofi, on="predio_join", how="inner")
    hold = hold.merge(g[["predio_join", "pc_pnbi"]], on="predio_join", how="left")

    # ---- métricas vs oficial ----
    rows = []
    # Comparación principal homogénea: todas las series sobre los mismos predios.
    mae, rmse, r2 = metrics(hold["valor_m2"].values, hold["oficial_m2"].values)
    rows.append({"serie": "Oferta observada (holdout común)", "n": len(hold),
                 "MAE": mae, "RMSE": rmse, "R2": r2})
    for m in MODELS:
        mae, rmse, r2 = metrics(hold[f"pred_{m}"].values, hold["oficial_m2"].values)
        rows.append({"serie": f"{m} predicho", "n": len(hold),
                     "MAE": mae, "RMSE": rmse, "R2": r2})
    met = pd.DataFrame(rows).round(3)
    met.to_csv(OUT / "val_ordenanza_metricas.csv", index=False)
    mae, rmse, r2 = metrics(base["valor_m2"].values, base["oficial_m2"].values)
    pd.DataFrame([{"serie": "Oferta observada (cobertura completa)", "n": len(base),
                   "MAE": mae, "RMSE": rmse, "R2": r2}]).round(3).to_csv(
                       OUT / "val_ordenanza_referencia_completa.csv", index=False)
    print("\n=== Métricas vs valor oficial de la ordenanza (2025) ===")
    print(met.to_string(index=False))
    print(f"\nRatio oferta/oficial: mediana={base['ratio_oferta_oficial'].median():.3f} "
          f"IQR=[{base['ratio_oferta_oficial'].quantile(.25):.2f}, {base['ratio_oferta_oficial'].quantile(.75):.2f}] "
          f"corr(log)={np.corrcoef(np.log(base['valor_m2']), np.log(base['oficial_m2']))[0,1]:.3f}")

    # ---- métricas por tipo de propiedad (misma muestra holdout) ----
    by_type = []
    groups = [("Todos", hold)] + list(hold.groupby("tipo"))
    for tipo, q in groups:
        for label, col in [("Oferta", "valor_m2")] + [(m, f"pred_{m}") for m in MODELS]:
            mae, rmse, r2 = metrics(q[col].values, q["oficial_m2"].values)
            by_type.append({"tipo": tipo, "serie": label, "n": len(q),
                            "MAE": mae, "RMSE": rmse, "R2": r2})
    pd.DataFrame(by_type).round(3).to_csv(OUT / "val_ordenanza_por_tipo.csv", index=False)

    # ---- discrepancia por zona (holdout, RF y GNNWR) ----
    for m in ["RF", "GNNWR"]:
        hold[f"disc_{m}"] = hold[f"pred_{m}"] - hold["oficial_m2"]
    zona = hold.groupby("zona").agg(
        n=("oficial_m2", "size"),
        disc_RF_media=("disc_RF", "mean"), disc_RF_mediana=("disc_RF", "median"),
        disc_GNNWR_media=("disc_GNNWR", "mean"), disc_GNNWR_mediana=("disc_GNNWR", "median"),
    ).round(1).reset_index()
    zona.to_csv(OUT / "val_ordenanza_por_zona.csv", index=False)

    # ---- análisis distributivo preliminar (NO auditoría IAAO) ----
    base["gap_oferta_oficial"] = base["valor_m2"] - base["oficial_m2"]
    base["gap_rel"] = base["gap_oferta_oficial"] / base["oficial_m2"]
    base["quintil_NBI"] = pd.qcut(base["pc_pnbi"], 5, labels=["NBI muy bajo", "bajo", "medio", "alto", "muy alto"], duplicates="drop")
    base["decil_oferta"] = pd.qcut(base["valor_m2"], 10, labels=False, duplicates="drop") + 1
    base["ratio_oficial_oferta"] = base["oficial_m2"] / base["valor_m2"]
    eq_nbi = base.groupby("quintil_NBI", observed=True).agg(
        n=("gap_oferta_oficial", "size"),
        gap_mediano=("gap_oferta_oficial", "median"),
        gap_rel_mediano=("gap_rel", "median"),
        ratio_oficial_oferta_mediano=("ratio_oficial_oferta", "median"),
    ).round(3).reset_index()
    eq_dec = base.groupby("decil_oferta").agg(
        n=("gap_oferta_oficial", "size"),
        oferta_mediana=("valor_m2", "median"),
        gap_mediano=("gap_oferta_oficial", "median"),
        ratio_oficial_oferta_mediano=("ratio_oficial_oferta", "median"),
    ).round(2).reset_index()
    eq_nbi.to_csv(OUT / "val_ordenanza_equidad.csv", index=False)
    eq_dec.to_csv(OUT / "val_ordenanza_equidad_decil.csv", index=False)
    print("\n=== Análisis distributivo preliminar por quintil NBI ===")
    print(eq_nbi.to_string(index=False))

    # tabla por predio
    keep = base[["predio_join", "tipo", "valor_m2", "oficial_m2", "ratio_oferta_oficial",
                 "ratio_oficial_oferta", "gap_oferta_oficial", "pc_pnbi",
                 "quintil_NBI", "decil_oferta"]].merge(
                     hold[["predio_join", *[f"pred_{m}" for m in MODELS]]],
                     on="predio_join", how="left")
    keep.to_csv(OUT / "val_ordenanza_predio.csv", index=False)

    # ==================== FIGURAS ====================
    # 1. Oferta observada vs oficial (todos)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lim = [5, 2500]
    ax.scatter(base["oficial_m2"], base["valor_m2"], s=6, alpha=0.35, edgecolors="none", color="#2c7fb8")
    ax.plot(lim, lim, "k--", lw=1, label="identidad (oferta = oficial)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Valor oficial ordenanza 2025 (USD/m²)"); ax.set_ylabel("Valor de oferta observado (USD/m²)")
    mae, rmse, r2 = metrics(base["valor_m2"].values, base["oficial_m2"].values)
    ax.set_title(f"Oferta observada vs. valor oficial de la ordenanza\n(n={len(base)}, R²={r2:.3f}, "
                 f"ratio mediano={base['ratio_oferta_oficial'].median():.2f})", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    fig.text(0.5, 0.005, "Fuente: elaboración propia (datos/val_catas.csv, ANIO=2025; datos/dataset.gpkg).",
             ha="center", fontsize=6)
    for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_obs_vs_oficial.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Predicho vs oficial (4 paneles)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    for ax, m in zip(axes, MODELS):
        ax.scatter(hold["oficial_m2"], hold[f"pred_{m}"], s=7, alpha=0.4, edgecolors="none", color="#238b45")
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        mae, rmse, r2 = metrics(hold[f"pred_{m}"].values, hold["oficial_m2"].values)
        ax.set_title(f"{m}\nMAE={mae:.0f}  RMSE={rmse:.0f}  R²={r2:.2f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Valor oficial 2025 (USD/m²)")
        if m == MODELS[0]: ax.set_ylabel("Valor predicho (USD/m²)")
    fig.suptitle("Predicción de modelos vs. valor oficial de la ordenanza vigente (holdout, n=%d)" % len(hold),
                 fontweight="bold", y=1.02)
    fig.text(0.5, -0.03, "Comparación descriptiva; el objetivo de entrenamiento (oferta) ≈ valor oficial (R²=0.96), "
             "por lo que reproduce la ordenanza, no la valida de forma independiente. Fuente: elaboración propia.",
             ha="center", fontsize=7)
    for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_pred_vs_oficial.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Mapa de discrepancia RF - oficial
    gg = g.to_crs(32717); gg["predio_join"] = gg["predio_join"].astype(str)
    hold_geo = gg.merge(hold[["predio_join", "disc_RF"]], on="predio_join", how="inner")
    fig, ax = plt.subplots(figsize=(7.5, 8))
    vmax = float(np.nanpercentile(np.abs(hold_geo["disc_RF"]), 95))
    sc = ax.scatter(hold_geo.geometry.x, hold_geo.geometry.y, c=hold_geo["disc_RF"].clip(-vmax, vmax),
                    s=12, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), alpha=0.9, edgecolors="none")
    ax.set_aspect("equal"); ax.set_xlabel("Este (m, UTM 17S)"); ax.set_ylabel("Norte (m, UTM 17S)"); ax.tick_params(labelsize=7)
    ax.set_title("Discrepancia RF − valor oficial de la ordenanza (holdout)\nrojo: RF>oficial · azul: RF<oficial",
                 fontsize=10, fontweight="bold")
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02); cb.set_label("RF − oficial (USD/m²)", fontsize=8)
    fig.text(0.5, 0.02, "Color recortado al percentil 95 de |RF-oficial|; los extremos se conservan en CSV. "
             "Las discrepancias no implican error causal. Fuente: elaboración propia.", ha="center", fontsize=6)
    for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_discrepancia_mapa.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Análisis distributivo preliminar (no auditoría IAAO)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    d1 = base.dropna(subset=["quintil_NBI"])
    order = ["NBI muy bajo", "bajo", "medio", "alto", "muy alto"]
    data = [d1[d1["quintil_NBI"] == s]["ratio_oficial_oferta"].clip(0, 3) for s in order]
    axes[0].boxplot(data, labels=[s.replace("NBI ", "") for s in order], showfliers=False)
    axes[0].axhline(1, color="r", ls="--", lw=1, label="oficial = oferta")
    axes[0].set_ylabel("ratio oficial / oferta"); axes[0].set_xlabel("Quintil muestral de NBI")
    axes[0].set_title("Concordancia por quintil muestral de NBI", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].tick_params(axis="x", rotation=20)
    ax2 = axes[1]
    ax2.plot(eq_dec["decil_oferta"], eq_dec["ratio_oficial_oferta_mediano"], "o-", color="#238b45")
    ax2.axhline(1, color="r", ls="--", lw=1)
    ax2.set_xlabel("Decil de precio de oferta (1=más bajo)"); ax2.set_ylabel("ratio mediano oficial/oferta")
    ax2.set_title("Concordancia por decil de precio de oferta", fontsize=10, fontweight="bold")
    fig.suptitle("Análisis distributivo preliminar de concordancia oferta–valor oficial",
                 fontweight="bold", y=1.02)
    fig.text(0.5, 0.01, "Quintiles internos; no constituye auditoría IAAO ni evaluación causal de equidad. Fuente: elaboración propia.",
             ha="center", fontsize=7)
    fig.subplots_adjust(bottom=0.25, top=0.80, wspace=0.28)
    for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_equidad_estrato.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] artefactos en {OUT}")


if __name__ == "__main__":
    main()
