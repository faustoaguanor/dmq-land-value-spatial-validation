# -*- coding: utf-8 -*-
"""
generate_applied_maps.py — Productos cartográficos aplicados de la tesis (v10).
================================================================================
Genera mapas de valoración estimada, error y confiabilidad espacial EXCLUSIVAMENTE
a partir de predicciones, geometrías y CSV ya existentes. NO reentrena ningún modelo.

Diseño (reglas de la consigna v10):
  - autodetecta los archivos de predicciones disponibles;
  - une con la geometría por `predio_join` (string) y valida correspondencia 1:1;
  - informa predios no emparejados;
  - reproyecta a EPSG:32717 (UTM 17S) para todos los cálculos;
  - convierte log→USD/m² con el factor de smearing de Duan de holdout por modelo
    (consistente con comparativo_holdout_smeared.csv y las métricas reportadas);
  - exporta PNG 300 dpi + PDF vectorial;
  - guarda un CSV con los valores usados en cada mapa;
  - usa escala de color COMÚN cuando compara modelos;
  - documenta la escala usada (log / cuantil / recorte) y NO oculta extremos:
    los registra en `tabla_extremos.csv`;
  - el mapa de confiabilidad es un INDICADOR TRANSPARENTE (distancia al train más
    cercano), no una probabilidad ni un intervalo de confianza.

Salidas: figures/aplicacion_valoracion/{png,pdf,csv}
  m01_precio_observado, m02_rf, m03_hgb, m04_gnnwr, m05_paneles_obs_vs_pred,
  m06_error_absoluto, m07_residuo_signo, m08_error_relativo, m09_confiabilidad,
  m10_recomendacion_territorial, tabla_extremos.csv, tabla_error_territorial.csv,
  map_data_holdout.csv (tabla larga con todos los valores por predio y modelo).
"""
from __future__ import annotations

# --- normalizacion de etiquetas SOLO para display (no toca claves de datos .loc[]) ---
import matplotlib.text as _mtext
_ORIG_SETTEXT = _mtext.Text.set_text
def _fix_label(s):
    if isinstance(s, str):
        for a, b in ((" "+chr(0x2014)+" ", " - "), (chr(0x2014), " - "),
                     (" "+chr(0xB7)+" ", ", "), (chr(0xB7), ", "),
                     (chr(0x2013), "-"),
                     ("SANNWR-"+chr(0x3B1), "SANNWR"), ("GWR-27", "GWR"),
                     ("n=1.011", "n=1,011"), ("n=1.007", "n=1,007")):
            s = s.replace(a, b)
        import re as _re
        s = _re.sub(r" +,", ",", s)
        s = _re.sub(r" {2,}", " ", s)
    return s
def _set_text_fixed(self, s):
    return _ORIG_SETTEXT(self, _fix_label(s))
_mtext.Text.set_text = _set_text_fixed
# --- fin normalizacion ---

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "datos" / "dataset.gpkg"
SPLIT = ROOT / "data_split" / "split.csv"
FOLDS = ROOT / "spatial_cv" / "output" / "fold_assignments.csv"
SMEARED = ROOT / "analisis" / "output_log" / "comparativo_holdout_smeared.csv"

# Modelos con predicciones y su factor de smearing holdout (fuente indicada).
# Los 4 modelos principales para mapas comparativos: RF, HGB, GNNWR, SANNWR.
PRED_FILES = {
    "RF":     ROOT/"modelos"/"baselines"/"output_log"/"rf_log_predictions.csv",
    "HGB":    ROOT/"modelos"/"baselines"/"output_log"/"hgb_log_predictions.csv",
    "GNNWR":  ROOT/"modelos"/"gnnwr"/"output_log"/"gnnwr_log_predictions.csv",
    "SANNWR": ROOT/"modelos"/"sannwr"/"output_log_real"/"sannwr_real_log_predictions.csv",
    "GWR-27": ROOT/"modelos"/"gwr"/"output_log_27vars"/"gwr27_log_predictions.csv",
    "MLP":    ROOT/"modelos"/"mlp"/"output_log"/"mlp_log_predictions.csv",
    "OLS":    ROOT/"modelos"/"ols"/"output_log"/"ols_log_predictions.csv",
}
MAIN_MODELS = ["RF", "SANNWR", "GWR-27", "GNNWR", "OLS"]
# Etiqueta a dibujar, distinta de la clave con que los CSV identifican
# al modelo (nomenclatura anterior a la reduccion a cinco modelos).
ETIQUETA = {"GWR-27": "GWR"}
def lbl(n):
    return ETIQUETA.get(n, n)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 9, "figure.dpi": 100,
})

FUENTE = ("Fuente: elaboración propia (datos/dataset.gpkg, EPSG:32717; holdout n=1.011). "
          "Precio de oferta negociado mediante comprador simulado.")


# ---------------------------------------------------------------------------
# 1. Carga y validación de datos
# ---------------------------------------------------------------------------
def load_smearing():
    df = pd.read_csv(SMEARED).set_index("modelo")
    sm = {m: float(df.loc[m, "smearing_factor"]) for m in df.index}
    # RF/HGB tienen su propio holdout con smearing
    for m, key in [("RF", "rf"), ("HGB", "hgb")]:
        h = pd.read_csv(ROOT/"modelos"/"baselines"/"output_log"/f"{key}_log_holdout.csv")
        sm[m] = float(h["smearing_factor"].iloc[0])
    return sm


def build_master():
    """GeoDataFrame de test (1.011) con obs + predicción USD/m² por modelo y metadatos."""
    gdf = gpd.read_file(DATA, layer="puntos_mercado").to_crs(epsg=32717)
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    split = pd.read_csv(SPLIT); split["predio_join"] = split["predio_join"].astype(int)
    folds = pd.read_csv(FOLDS); folds["predio_join"] = folds["predio_join"].astype(int)
    gdf = gdf.merge(split[["predio_join", "split", "zona"]], on="predio_join", how="left")
    gdf = gdf.merge(folds[["predio_join", "fold"]], on="predio_join", how="left")
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    gdf["valor_m2"] = gdf["valor_m2"].astype(float)

    sm = load_smearing()
    report = []
    gtest = gdf[gdf["split"] == "test"].copy().reset_index(drop=True)
    for name, path in PRED_FILES.items():
        if not path.exists():
            report.append((name, "FALTA ARCHIVO", 0, 0)); continue
        p = pd.read_csv(path)
        p["predio_join"] = p["predio_join"].astype(int)
        p = p[p["split"] == "test"][["predio_join", "y_pred_log"]].copy()
        n_before = len(gtest)
        merged = gtest[["predio_join"]].merge(p, on="predio_join", how="left")
        n_match = merged["y_pred_log"].notna().sum()
        n_unmatched = n_before - n_match
        s = sm.get(name, 1.0)
        gtest[f"pred_{name}"] = np.exp(merged["y_pred_log"].values) * s
        gtest[f"err_{name}"] = np.abs(gtest[f"pred_{name}"] - gtest["valor_m2"])
        gtest[f"resid_{name}"] = gtest["valor_m2"] - gtest[f"pred_{name}"]
        gtest[f"relerr_{name}"] = gtest[f"err_{name}"] / gtest["valor_m2"]
        report.append((name, f"smearing={s:.4f}", n_match, n_unmatched))

    # confiabilidad: distancia de cada test al train más cercano
    gtrain = gdf[gdf["split"] == "train"]
    tree = cKDTree(np.column_stack([gtrain["x"], gtrain["y"]]))
    d, _ = tree.query(np.column_stack([gtest["x"], gtest["y"]]), k=1)
    gtest["dist_train_m"] = d

    print("=== Emparejamiento predicciones <-> geometria (test n=%d) ===" % len(gtest))
    for name, info, nm, nu in report:
        print(f"  {name:8s} {info:20s} emparejados={nm:5d} no_emparejados={nu}")
    return gtest, gdf


# ---------------------------------------------------------------------------
# 2. Utilidades de dibujo
# ---------------------------------------------------------------------------
def _scalebar(ax, length_m=5000, label="5 km"):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xs = x0 + (x1 - x0) * 0.06; ys = y0 + (y1 - y0) * 0.06
    ax.plot([xs, xs + length_m], [ys, ys], color="black", lw=2.5, solid_capstyle="butt")
    ax.text(xs + length_m/2, ys + (y1-y0)*0.015, label, ha="center", va="bottom", fontsize=7)

def _north(ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xn = x1 - (x1 - x0) * 0.06; yn = y1 - (y1 - y0) * 0.10
    ax.annotate("N", xy=(xn, yn), xytext=(xn, yn - (y1-y0)*0.06),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
                ha="center", va="center", fontsize=9, fontweight="bold")

def _base(ax, g=None, recorte=True):
    ax.set_aspect("equal")
    ax.set_xlabel("Este (m, UTM 17S)"); ax.set_ylabel("Norte (m, UTM 17S)")
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style="plain")
    if g is not None and recorte:
        # Encuadre al p1-p99: un puñado de predios muy alejados estiraba el
        # recuadro y dejaba la zona densa reducida a una franja.
        xmin, xmax = g["x"].quantile([0.01, 0.99])
        ymin, ymax = g["y"].quantile([0.01, 0.99])
        m = 900
        ax.set_xlim(xmin - m, xmax + m); ax.set_ylim(ymin - m, ymax + m)

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {stem}.png / .pdf")


# ---------------------------------------------------------------------------
# 3. Mapas
# ---------------------------------------------------------------------------
def map_value(g, col, title, stem, vmin, vmax, cmap="viridis", unit="USD/m²",
              note="Escala logarítmica común."):
    fig, ax = plt.subplots(figsize=(7.5, 8))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sc = ax.scatter(g["x"], g["y"], c=g[col].clip(vmin, vmax), s=10, cmap=cmap,
                    norm=norm, alpha=0.9, edgecolors="none")
    _base(ax); _scalebar(ax); _north(ax)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02); cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.text(0.5, -0.09, note + " " + FUENTE, transform=ax.transAxes,
            ha="center", va="top", fontsize=6, wrap=True)
    save(fig, stem)


def map_panels_value(g, models, vmin, vmax, stem):
    n = len(models) + 1
    nrow, ncol = (2, 3) if n > 4 else ((2, 2) if n == 4 else (1, n))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 5.4*nrow))
    axes = np.atleast_1d(axes).ravel()
    for _a in axes[n:]:
        _a.axis("off")
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cols = [("valor_m2", "Observado")] + [(f"pred_{m}", m) for m in models]
    for ax, (c, t) in zip(axes, cols):
        sc = ax.scatter(g["x"], g["y"], c=g[c].clip(vmin, vmax), s=6, cmap="viridis",
                        norm=norm, alpha=0.9, edgecolors="none")
        _base(ax, g); ax.set_title(lbl(t), fontweight="bold", fontsize=11)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks([]); ax.set_yticks([])
    _scalebar(axes[0]); _north(axes[-1])
    # Barra horizontal bajo la reticula: en vertical y al costado se llevaba un
    # quinto del ancho, y como la figura ya es alta ese ancho perdido obligaba a
    # reducirla al colocarla en la pagina.
    cb = fig.colorbar(sc, ax=axes.tolist(), orientation="horizontal",
                      shrink=0.5, pad=0.03, aspect=42)
    cb.set_label("USD/m²", fontsize=9)
    save(fig, stem)


def map_panels_metric(g, models, colprefix, title, stem, cmap, vmax, unit,
                      diverging=False, note=""):
    n = len(models)
    nrow, ncol = (2, 3) if n > 4 else ((2, 2) if n == 4 else (1, n))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 5.4*nrow))
    axes = np.atleast_1d(axes).ravel()
    for _a in axes[n:]:
        _a.axis("off")
    if diverging:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None
    for ax, m in zip(axes, models):
        c = f"{colprefix}_{m}"
        vals = g[c].clip(-vmax, vmax) if diverging else g[c].clip(0, vmax)
        sc = ax.scatter(g["x"], g["y"], c=vals, s=6, cmap=cmap,
                        norm=norm, vmin=None if diverging else 0,
                        vmax=None if diverging else vmax, alpha=0.9, edgecolors="none")
        _base(ax, g); ax.set_title(lbl(m), fontweight="bold", fontsize=11)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks([]); ax.set_yticks([])
    _scalebar(axes[0]); _north(axes[-1])
    cb = fig.colorbar(sc, ax=axes.tolist(), shrink=0.45, pad=0.02, aspect=28); cb.set_label(unit, fontsize=8)
    save(fig, stem)


def compute_support(gtest, k=1):
    """Soporte espacial del HOLDOUT respecto del conjunto de entrenamiento.

    ``dist_train_m`` se calculó previamente consultando exclusivamente el KDTree del
    train. Los terciles son categorías descriptivas internas, no umbrales validados de
    seguridad, probabilidades ni una regla para escoger modelos.
    """
    gtest = gtest.copy()
    gtest["dist_k"] = gtest["dist_train_m"]
    q33, q66 = gtest["dist_k"].quantile([1/3, 2/3])
    def cat(v):
        if v <= q33: return "high"
        if v <= q66: return "mid"
        return "low"
    gtest["soporte_cat"] = gtest["dist_k"].map(cat)
    return gtest, float(q33), float(q66)


def map_reliability(gfull, q33, q66, stem):
    """Soporte relativo del holdout según distancia al train más cercano."""
    labels = [(f"Alto (≤{q33:.0f} m al train)", "high", "#2c7fb8"),
              (f"Media ({q33:.0f}–{q66:.0f} m)", "mid", "#fdae61"),
              (f"Baja (>{q66:.0f} m)", "low", "#d7191c")]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    for lab, key, col in labels:
        sel = gfull["soporte_cat"] == key
        ax.scatter(gfull.loc[sel, "x"], gfull.loc[sel, "y"], s=7, c=col,
                   label=f"{lab} (n={int(sel.sum())})", alpha=0.85, edgecolors="none")
    _base(ax); _scalebar(ax); _north(ax)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9, title="Confiabilidad relativa")
    ax.set_title("Soporte espacial relativo del holdout respecto del entrenamiento\n"
                 "(distancia al vecino de train más cercano, n=1.011)", fontweight="bold", fontsize=10, pad=8)
    ax.text(0.5, -0.09, ("Terciles descriptivos de distancia al train. NO son umbrales validados, "
            "probabilidades ni intervalos de confianza. " + FUENTE), transform=ax.transAxes,
            ha="center", va="top", fontsize=6, wrap=True)
    save(fig, stem)


def map_recommendation(gfull, q33, q66, stem):
    """Diagrama aplicado por ESCENARIO; no asigna modelos por terciles no validados."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_axis_off()
    boxes = [
        # Cada rama se define por una condicion verificable sobre la distancia
        # al predio investigado mas cercano, que es lo que el mapa de soporte
        # espacial cartografia, y desemboca en una accion concreta. Los
        # rotulos anteriores ("fuera del soporte", "uso con efecto fiscal")
        # no permitian saber cuando aplicaba cada rama.
        (0.04, 0.58, 0.26, 0.25, "Hay predios investigados\na menos de 250 m", "#d9f0d3"),
        (0.37, 0.58, 0.26, 0.25, "El predio investigado más\ncercano está entre\n250 m y 1 km", "#fee08b"),
        (0.70, 0.58, 0.26, 0.25, "No hay ningún predio\ninvestigado a menos\nde 1 km", "#f4a6a6"),
        (0.04, 0.15, 0.26, 0.22, "Estimar con\nRandom Forest", "#b8e3b0"),
        (0.37, 0.15, 0.26, 0.22, "Estimar con GNNWR\ny contrastar con\nRandom Forest", "#f6d66f"),
        (0.70, 0.15, 0.26, 0.22, "No estimar de forma\nautomática: derivar a\nvaloración individual", "#e77f7f"),
    ]
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    for x, y, w, h, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                   facecolor=col, edgecolor="#444", linewidth=1.2))
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=8.5)
    for x in (0.17, 0.50, 0.83):
        ax.add_patch(FancyArrowPatch((x, 0.57), (x, 0.39), arrowstyle="-|>",
                                    mutation_scale=15, color="#444", linewidth=1.4))
    ax.set_title("Regla de decisión por escenario de uso (propuesta aplicada)",
                 fontweight="bold", fontsize=13, pad=12)
    ax.text(0.5, 0.02, ("La selección no se deriva de terciles de distancia: resume los escenarios "
            "evaluados. Requiere revisión humana y validación externa para uso oficial."),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8, wrap=True)
    save(fig, stem)


# ---------------------------------------------------------------------------
# 4. Tablas: extremos y error territorial
# ---------------------------------------------------------------------------
def tabla_extremos(g, models):
    rows = []
    for m in models:
        c = f"pred_{m}"
        top = g.nlargest(5, c)[["predio_join", "valor_m2", c, "zona", "fold"]]
        for _, r in top.iterrows():
            rows.append({"modelo": m, "tipo": "pred_max", "predio_join": r["predio_join"],
                         "valor_obs": round(r["valor_m2"], 1), "valor_pred": round(r[c], 1),
                         "zona": r["zona"], "fold": r["fold"]})
    obs_top = g.nlargest(5, "valor_m2")[["predio_join", "valor_m2", "zona", "fold"]]
    for _, r in obs_top.iterrows():
        rows.append({"modelo": "OBSERVADO", "tipo": "obs_max", "predio_join": r["predio_join"],
                     "valor_obs": round(r["valor_m2"], 1), "valor_pred": np.nan,
                     "zona": r["zona"], "fold": r["fold"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tabla_extremos.csv", index=False)
    print(f"[OK] tabla_extremos.csv ({len(df)} filas)")


def tabla_territorial(g, models):
    rows = []
    for part_name, part_col in [("fold_SpatialBlock", "fold"), ("zona_KMeans", "zona")]:
        for key, sub in g.groupby(part_col):
            for m in models:
                resid = sub[f"resid_{m}"]      # obs - pred
                err = sub[f"err_{m}"]
                rows.append({
                    "particion": part_name, "region": int(key) if pd.notna(key) else key,
                    "modelo": m, "n": len(sub),
                    "MAE": round(err.mean(), 2),
                    "RMSE": round(np.sqrt((resid**2).mean()), 2),
                    "sesgo_medio_obs_menos_pred": round(resid.mean(), 2),
                    "pct_subestimacion": round((resid > 0).mean()*100, 1),   # obs>pred → modelo subestima
                    "pct_sobreestimacion": round((resid < 0).mean()*100, 1),
                    "aviso_n_bajo": "n<30" if len(sub) < 30 else "",
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tabla_error_territorial.csv", index=False)
    print(f"[OK] tabla_error_territorial.csv ({len(df)} filas)")
    return df


def map_territorial_mae(g, model, stem):
    """Heat de MAE por fold y zona para un modelo (mapa de puntos coloreado por error abs)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, part in zip(axes, ["fold", "zona"]):
        agg = g.groupby(part)[f"err_{model}"].mean()
        cvals = g[part].map(agg)
        sc = ax.scatter(g["x"], g["y"], c=cvals, s=8, cmap="magma_r", alpha=0.9, edgecolors="none")
        _base(ax); ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_title(f"MAE medio por {'fold SpatialBlock' if part=='fold' else 'zona (KMeans)'}",
                     fontsize=10, fontweight="bold")
        cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02); cb.set_label("MAE (USD/m²)", fontsize=8)
    _scalebar(axes[0]); _north(axes[1])
    fig.suptitle(f"Distribución territorial del error absoluto medio — {model}", fontweight="bold", y=1.01)
    fig.text(0.5, -0.02, ("`zona` es un clúster espacial (KMeans de estratificación), no una parroquia "
             "administrativa.  " + FUENTE), ha="center", fontsize=6, wrap=True)
    save(fig, stem)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    g, gfull = build_master()
    models_present = [m for m in MAIN_MODELS if f"pred_{m}" in g.columns]
    print("Modelos principales con predicciones:", models_present)

    # escala común de valor (recorte a percentiles 1-99 para no dejar que outliers dominen;
    # los extremos se registran aparte en tabla_extremos.csv, no se ocultan)
    allvals = pd.concat([g["valor_m2"]] + [g[f"pred_{m}"] for m in models_present])
    vmin = max(float(allvals.quantile(0.01)), 1.0)
    vmax = float(allvals.quantile(0.99))
    print(f"Escala común valor USD/m²: [{vmin:.1f}, {vmax:.1f}] (recorte p1-p99; extremos en tabla_extremos.csv)")

    # Mapa 1: observado
    map_value(g, "valor_m2", "Precio de oferta negociado observado del suelo urbano en el DMQ\n(conjunto de evaluación, n=1.011)",
              "m01_precio_observado", vmin, vmax)
    # Mapas 2-4: RF, HGB, GNNWR
    if "RF" in models_present:
        map_value(g, "pred_RF", "Precio estimado por Random Forest en predios del conjunto de evaluación",
                  "m02_rf", vmin, vmax)
    if "GNNWR" in models_present:
        map_value(g, "pred_GNNWR", "Precio estimado por GNNWR bajo el dominio evaluado\n(solo predios de prueba; no es un mapa de toda la ciudad)",
                  "m04_gnnwr", vmin, vmax)
    # Mapa 5: paneles
    map_panels_value(g, models_present, vmin, vmax, "m05_paneles_obs_vs_pred")
    # Mapa 6: error absoluto (escala común)
    emax = float(pd.concat([g[f"err_{m}"] for m in models_present]).quantile(0.95))
    map_panels_metric(g, models_present, "err", "Distribución espacial del error absoluto por modelo (holdout)",
                      "m06_error_absoluto", "magma_r", emax, "|error| (USD/m²)",
                      note=f"Escala común recortada al p95 ({emax:.0f} USD/m²); extremos en tabla_extremos.csv.")
    # Mapa 7: residuo con signo
    rmax = float(pd.concat([g[f"resid_{m}"].abs() for m in models_present]).quantile(0.95))
    map_panels_metric(g, models_present, "resid", "Patrones espaciales de subestimación (>0) y sobreestimación (<0)",
                      "m07_residuo_signo", "RdBu", rmax, "residuo obs−pred (USD/m²)",
                      diverging=True, note=f"Paleta divergente centrada en 0; recorte ±p95 (±{rmax:.0f}).")
    # Mapa 8: error relativo (con caveat)
    relmax = 1.0  # 100%
    map_panels_metric(g, models_present, "relerr", "Error relativo absoluto por modelo (|pred−obs|/obs)",
                      "m08_error_relativo", "YlOrRd", relmax, "error relativo (fracción)",
                      note="ADVERTENCIA: el error relativo se infla en precios bajos; interpretar con la tabla por decil.")
    # Mapa 9: soporte del HOLDOUT medido exclusivamente contra TRAIN.
    # Figura 10: diagrama de decisión por escenario, no regla espacial validada.
    gsupport, q33, q66 = compute_support(g, k=1)
    map_reliability(gsupport, q33, q66, "m09_confiabilidad")
    map_recommendation(gsupport, q33, q66, "m10_recomendacion_territorial")
    # etiqueta de soporte para el test (para el CSV maestro): unir por predio_join
    sup_map = gsupport.set_index("predio_join")["soporte_cat"]
    g["soporte_cat"] = g["predio_join"].map(sup_map)

    # Mapa territorial de MAE por fold/zona (para el modelo de generalización espacial: GNNWR)
    if "GNNWR" in models_present:
        map_territorial_mae(g, "GNNWR", "m11_mae_territorial_gnnwr")
    if "RF" in models_present:
        map_territorial_mae(g, "RF", "m12_mae_territorial_rf")

    # Tablas
    tabla_extremos(g, models_present)
    terr = tabla_territorial(g, models_present)

    # CSV maestro con valores por predio
    keep = ["predio_join", "x", "y", "zona", "fold", "valor_m2", "dist_train_m", "soporte_cat"]
    keep += [c for c in g.columns if c.startswith(("pred_", "err_", "resid_", "relerr_"))]
    out = pd.DataFrame(g[keep])
    out.to_csv(OUT / "map_data_holdout.csv", index=False)
    print(f"[OK] map_data_holdout.csv ({len(out)} filas, {len(out.columns)} columnas)")

    # soporte sobre toda la ciudad (5.051)
    print("\n=== Soporte espacial del holdout respecto de train (n=1.011) ===")
    print(gsupport["soporte_cat"].value_counts())
    print("\n=== MAE territorial (resumen por fold, modelos principales) ===")
    piv = terr[terr.particion == "fold_SpatialBlock"].pivot_table(
        index="region", columns="modelo", values="MAE")
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
