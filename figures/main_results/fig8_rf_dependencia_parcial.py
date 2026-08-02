# -*- coding: utf-8 -*-
"""
fig8_rf_dependencia_parcial.py
==============================
Dependencia parcial de Random Forest sobre sus variables mas influyentes.

Por que esta figura. La revision pidio explicar POR QUE Random Forest gana en
interpolacion, y propuso dibujar un arbol. Un arbol de los 300 no explica el
modelo: cada arbol individual es un predictor debil y la capacidad del bosque
reside justamente en promediarlos, de modo que mostrar uno daria una idea
falsa de como decide. La dependencia parcial si describe el comportamiento
del bosque completo: para cada valor de una variable, promedia la prediccion
sobre todos los predios manteniendo el resto de sus caracteristicas, y revela
la forma de la relacion que el modelo aprendio.

Lo que se busca ver es si esa forma es no lineal, porque ahi esta la
explicacion de la ventaja frente a OLS: un modelo lineal solo puede trazar
una recta donde el bosque traza una curva con mesetas y quiebres.

Salida: figures/main_results/fig8_rf_dependencia_parcial.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_RF = "#1B7837"

# Las cinco variables de mayor importancia por permutacion (Tabla 5.6), con su
# nombre legible y la unidad en que se expresan.
VARS = [
    ("dist_cc",            "Distancia a centros comerciales", "m"),
    ("dist_parque_metro",  "Distancia a parques metropolitanos", "m"),
    ("dist_centr_metro",   "Distancia a la centralidad metropolitana", "m"),
    ("dist_metro",         "Distancia al Metro", "m"),
    ("log_area",           "Superficie del lote (logaritmo)", "log m²"),
    ("pc_pnbi",            "Necesidades básicas insatisfechas", "%"),
]


def main():
    gdf = gpd.read_file(ROOT / "datos" / "dataset.gpkg")
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    split = pd.read_csv(ROOT / "data_split" / "split.csv")
    split["predio_join"] = split["predio_join"].astype(int)
    df = gdf.merge(split[["predio_join", "split"]], on="predio_join", how="inner")

    # Mismo vector de entrada que el modelo del cuerpo: 27 variables, sin
    # coordenadas, con la categorica de uso de suelo en columnas binarias.
    excl = {"predio_join", "valor_m2", "split", "geometry"}
    num = [c for c in df.columns
           if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
    X = pd.get_dummies(df[num].copy(), drop_first=False)
    y = np.log(df["valor_m2"].values)

    tr = (df["split"] == "train").values
    modelo = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    modelo.fit(X[tr], y[tr])
    print(f"  bosque ajustado sobre {tr.sum()} predios, {X.shape[1]} columnas")

    presentes = [(c, n, u) for c, n, u in VARS if c in X.columns]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))
    axes = axes.ravel()

    reales = []
    for ax, (col, nombre, unidad) in zip(axes, presentes):
        disp = PartialDependenceDisplay.from_estimator(
            modelo, X[tr], [col], ax=ax, line_kw={"color": C_RF, "linewidth": 2.0},
            percentiles=(0.02, 0.98), grid_resolution=60)
        # from_estimator crea sus propios ejes dentro del hueco recibido: hay
        # que rotular y dibujar sobre esos, no sobre el contenedor.
        a = np.ravel(disp.axes_)[0]
        reales.append(a)

        # Recta entre los extremos de la curva: lo unico que un modelo lineal
        # podria representar de esa misma relacion. La distancia entre ambas
        # es, visualmente, lo que Random Forest captura y OLS no.
        xs = np.ravel(disp.pd_results[0]["grid_values"][0])
        ys = np.ravel(disp.pd_results[0]["average"])
        a.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color="#B0413E",
               lw=1.2, ls="--", zorder=1, label="recta equivalente")

        a.set_title(nombre, fontsize=10.5, fontweight="bold", pad=6)
        a.set_xlabel(unidad, fontsize=9)
        a.set_ylabel("log(precio) estimado", fontsize=9)
        a.grid(alpha=0.25, ls="--")
        a.set_axisbelow(True)
        a.tick_params(labelsize=8.5)

    reales[0].legend(fontsize=8.5, frameon=False, loc="upper right")
    for ax in axes[len(presentes):]:
        ax.axis("off")

    fig.tight_layout(h_pad=2.0, w_pad=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig8_rf_dependencia_parcial.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig8_rf_dependencia_parcial.png / .pdf")


if __name__ == "__main__":
    main()
