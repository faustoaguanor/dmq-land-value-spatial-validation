# -*- coding: utf-8 -*-
"""
mapa_soporte_superficie.py
==========================
Superficie continua de soporte espacial: a que distancia esta el predio de
entrenamiento mas cercano, en cada punto del territorio muestreado.

Por que una superficie y no puntos. La version anterior pintaba los 1,011
predios de evaluacion en tres colores segun su tercil de distancia. Con esa
densidad los tres colores se superponen y no se distingue ningun patron: el
mapa no comunica nada. Una superficie continua si lo hace.

Y aqui la superficie es legitima, a diferencia de un mapa de error. La
distancia al predio de entrenamiento mas cercano es una funcion definida en
CUALQUIER punto del espacio, no una cantidad medida solo donde hay
observaciones: se puede calcular exactamente en cada celda de una malla sin
estimar ni interpolar nada. Lo que se dibuja son valores calculados, no
inferidos.

La malla se recorta a 2 km alrededor de los predios muestreados, porque mas
alla de esa distancia el estudio no tiene nada que decir, y se superponen los
limites parroquiales para poder ubicarse.

Salida: figures/aplicacion_valoracion/mapa_soporte.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).parent

plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})

MAP_OUTER = "#5b6b76"
MAP_FRAME = "#3d4852"
PASO = 150          # tamaño de celda de la malla, en metros
ALCANCE = 2000      # hasta donde se dibuja alrededor de los predios muestreados
CORTES = [0, 100, 250, 500, 1000, 2000, 4000]   # metros


def main():
    g = gpd.read_file(ROOT / "datos" / "dataset.gpkg").to_crs(32717)
    g["predio_join"] = g["predio_join"].astype(int)
    sp = pd.read_csv(ROOT / "data_split" / "split.csv")
    sp["predio_join"] = sp["predio_join"].astype(int)
    g = g.merge(sp[["predio_join", "split"]], on="predio_join")
    tr = g[g["split"] == "train"]
    te = g[g["split"] == "test"]
    print(f"  entrenamiento {len(tr)}, evaluacion {len(te)}")

    # Malla regular sobre la extension muestreada
    x0, y0, x1, y1 = g.total_bounds
    xs = np.arange(x0 - ALCANCE, x1 + ALCANCE, PASO)
    ys = np.arange(y0 - ALCANCE, y1 + ALCANCE, PASO)
    XX, YY = np.meshgrid(xs, ys)
    puntos = np.c_[XX.ravel(), YY.ravel()]
    print(f"  malla de {len(xs)}x{len(ys)} = {len(puntos):,} celdas de {PASO} m")

    # Distancia exacta al predio de entrenamiento mas cercano
    arbol = cKDTree(np.c_[tr.geometry.x, tr.geometry.y])
    dist, _ = arbol.query(puntos, k=1)
    dist = dist.reshape(XX.shape)

    # Fuera del alcance del muestreo no se dibuja: el estudio no cubre esa zona
    arbol_todo = cKDTree(np.c_[g.geometry.x, g.geometry.y])
    d_muestra, _ = arbol_todo.query(puntos, k=1)
    dist = np.where(d_muestra.reshape(XX.shape) <= ALCANCE, dist, np.nan)

    parroquias = gpd.read_file(ROOT.parent / "capas" / "PARROQUIAS_F.shp").to_crs(32717)

    fig, ax = plt.subplots(figsize=(8.2, 9.0))
    cmap = ListedColormap(plt.cm.YlOrRd(np.linspace(0.06, 0.92, len(CORTES) - 1)))
    cmap.set_bad("#ffffff")
    im = ax.pcolormesh(XX, YY, dist, cmap=cmap, norm=BoundaryNorm(CORTES, cmap.N),
                       shading="auto", zorder=1)

    parroquias.boundary.plot(ax=ax, color=MAP_OUTER, linewidth=0.45, alpha=0.85, zorder=3)
    ax.scatter(te.geometry.x, te.geometry.y, s=1.6, c="#1a1a1a", alpha=0.55,
               linewidths=0, zorder=4, label="predios evaluados")

    xmin, xmax = np.percentile(g.geometry.x, [1, 99])
    ymin, ymax = np.percentile(g.geometry.y, [1, 99])
    ax.set_xlim(xmin - 1500, xmax + 1500)
    ax.set_ylim(ymin - 1500, ymax + 1500)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MAP_FRAME); s.set_linewidth(0.9)

    cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02, aspect=26,
                      ticks=CORTES, spacing="uniform")
    cb.set_label("Distancia al predio de entrenamiento más cercano (m)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9,
              markerscale=4)

    # escala grafica
    xs_, xe = ax.get_xlim(); ys_, ye = ax.get_ylim()
    bx = xs_ + (xe - xs_) * 0.06; by = ys_ + (ye - ys_) * 0.12
    ax.plot([bx, bx + 5000], [by, by], color=MAP_FRAME, lw=2.2, solid_capstyle="butt", zorder=6)
    ax.text(bx + 2500, by + (ye - ys_) * 0.012, "5 km", ha="center", va="bottom",
            fontsize=8.5, color=MAP_FRAME, zorder=6)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"mapa_soporte.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> mapa_soporte.png / .pdf")

    val = dist[~np.isnan(dist)]
    print("\n  reparto del territorio muestreado por distancia al entrenamiento:")
    for a, b in zip(CORTES[:-1], CORTES[1:]):
        pct = 100 * ((val >= a) & (val < b)).mean()
        print(f"    {a:>5}-{b:<5} m: {pct:5.1f}%")


if __name__ == "__main__":
    main()
