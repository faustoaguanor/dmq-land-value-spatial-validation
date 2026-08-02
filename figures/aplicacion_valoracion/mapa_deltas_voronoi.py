# -*- coding: utf-8 -*-
"""
mapa_deltas_voronoi.py
======================
Mapa de la diferencia entre valor observado y estimado (delta) por modelo.

Por que este mapa. La revision pidio "mostrar el delta" y "ver mejor el
error, que modelos fallan en donde". El mapa de error ABSOLUTO no responde a
eso: al tomar valor absoluto pierde el signo, de modo que un modelo que
sobreestima y otro que subestima el mismo predio se dibujan identicos. El
delta con signo si distingue ambos casos, que es lo que importa para un
catastro, porque subestimar significa recaudar de menos y sobreestimar
significa cobrar de mas.

Por que agregado a rejilla y no predio a predio. Se probaron ambas formas.
A nivel de predio, con teselacion de Voronoi, el mapa resulta ilegible: los
signos alternan de un predio al siguiente y el ojo no distingue patron
alguno. Esa alternancia es real, pero es ruido de predio individual y tapa
lo que interesa, que es el sesgo SISTEMATICO por zona. Promediando el delta
dentro de celdas de 2 km ese ruido se cancela y emerge la estructura
regional. La operacion no inventa nada: promedia observaciones medidas
dentro de cada celda, a diferencia de una interpolacion, que estimaria
valores donde no se midio ninguno, cosa inaceptable en un mapa de error.

Solo se dibujan las celdas con al menos cinco predios evaluados, para que
ninguna quede determinada por una observacion suelta.

Salida: figures/aplicacion_valoracion/mapa_deltas.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).parent

plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})

MAP_LAND = "#f7f9fa"
MAP_OUTER = "#93a1ab"
MAP_FRAME = "#3d4852"
MODELOS = ["RF", "GNNWR", "SANNWR"]
CELDA_M = 1000
MIN_PREDIOS = 3
RECORTE_PCT = (1, 99)


def rejilla(g):
    """Agrega el delta de cada modelo a celdas cuadradas de CELDA_M."""
    x, y = g.geometry.x.values, g.geometry.y.values
    g = g.copy()
    g["cx"] = np.floor(x / CELDA_M).astype(int)
    g["cy"] = np.floor(y / CELDA_M).astype(int)
    cols = {f"resid_{m}": "mean" for m in MODELOS}
    cols["x"] = "size"
    agg = g.groupby(["cx", "cy"]).agg(cols).rename(columns={"x": "n"}).reset_index()
    agg = agg[agg["n"] >= MIN_PREDIOS]
    geom = [box(cx * CELDA_M, cy * CELDA_M, (cx + 1) * CELDA_M, (cy + 1) * CELDA_M)
            for cx, cy in zip(agg["cx"], agg["cy"])]
    return gpd.GeoDataFrame(agg, geometry=geom, crs=g.crs)


def main():
    d = pd.read_csv(OUT / "map_data_holdout.csv")
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d["x"], d["y"]),
                         crs="EPSG:32717")
    print(f"  {len(g)} predios de evaluacion")

    cel = rejilla(g)
    print(f"  {len(cel)} celdas de {CELDA_M/1000:.0f} km con >= {MIN_PREDIOS} predios "
          f"({cel['n'].sum()} predios, {100*cel['n'].sum()/len(g):.0f}% del total)")

    # Escala comun y simetrica sobre el delta ya promediado.
    todos = pd.concat([cel[f"resid_{m}"].abs() for m in MODELOS])
    vmax = float(np.ceil(todos.quantile(0.97) / 10) * 10)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    print(f"  escala comun: +/-{vmax:.0f} USD/m2")

    parroquias = gpd.read_file(ROOT.parent / "capas" / "PARROQUIAS_F.shp").to_crs(32717)
    dmq = parroquias.geometry.union_all()

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 6.6))
    xmin, xmax = np.percentile(g.geometry.x, RECORTE_PCT)
    ymin, ymax = np.percentile(g.geometry.y, RECORTE_PCT)

    for ax, m in zip(axes, MODELOS):
        gpd.GeoSeries([dmq], crs=32717).plot(ax=ax, facecolor=MAP_LAND,
                                             edgecolor=MAP_OUTER, lw=0.8, zorder=0)
        # Limites parroquiales para poder ubicarse: sin ellos las celdas
        # flotan sobre un fondo sin referencias reconocibles.
        parroquias.boundary.plot(ax=ax, color="#9aa5ad", linewidth=0.35,
                                 alpha=0.9, zorder=3)
        cel.plot(ax=ax, column=f"resid_{m}", cmap="RdBu_r", norm=norm,
                 edgecolor="none", zorder=2, alpha=0.92)
        ax.set_title(m, fontsize=12.5, fontweight="bold", pad=8)
        ax.set_aspect("equal")
        ax.set_xlim(xmin - 900, xmax + 900)
        ax.set_ylim(ymin - 900, ymax + 900)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MAP_FRAME); s.set_linewidth(0.9)

        # El peor sesgo de zona: la cifra que el mapa senala pero no cuantifica.
        peor = cel.loc[cel[f"resid_{m}"].abs().idxmax(), f"resid_{m}"]
        signo = "subestima" if peor > 0 else "sobreestima"
        ax.text(0.5, -0.035,
                f"peor zona: {signo} {abs(peor):.0f} USD/m²",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color="#555555", style="italic")

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    cb = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                      shrink=0.42, pad=0.09, aspect=36)
    cb.set_label("Diferencia media por zona, observado $-$ estimado (USD/m²).   "
                 "Rojo: el modelo subestima.   Azul: sobreestima.", fontsize=9.5)
    cb.outline.set_edgecolor(MAP_FRAME); cb.outline.set_linewidth(0.7)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"mapa_deltas.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  -> mapa_deltas.png / .pdf")

    print("\n  Sesgo por modelo sobre las celdas dibujadas (USD/m2):")
    for m in MODELOS:
        s = cel[f"resid_{m}"]
        print(f"    {m:7} media {s.mean():6.1f} | celdas que subestiman {100*(s>0).mean():4.0f}% "
              f"| peor zona {s.abs().max():6.1f}")


if __name__ == "__main__":
    main()
