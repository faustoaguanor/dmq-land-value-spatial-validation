# -*- coding: utf-8 -*-
"""
mapas_resultados.py
===================
Rehace los cuatro mapas del capitulo de resultados con una base cartografica
comun. Sustituye a fig7_all_models_maps.py, mapa_deltas_voronoi.py y
mapa_soporte_superficie.py, cuyas salidas la revision rechazo por ilegibles.

Que fallaba en cada uno y como se corrige.

Precio observado y estimado. La barra de color era logaritmica y solo tenia
una marca, "10^2", de modo que el lector no podia traducir ningun color a un
precio. Ahora la escala es de clases discretas por cuantiles con los cortes
rotulados en USD/m2, que es lo que se lee de un vistazo.

Error absoluto. Eran 1,011 puntos superpuestos sobre fondo blanco: en el
nucleo consolidado los puntos se tapan unos a otros y el orden de dibujo
decide que color queda visible, de modo que no se distingue donde el error
es mayor. Se agrega a celdas, que es lo unico que elimina el solapamiento
sin inventar datos, y las cinco clases son las mismas en todos los paneles
para que los modelos sean comparables entre si.

Sesgo. Faltaban OLS y GWR, con lo que la figura contradecia al resto del
capitulo, y la rampa continua dejaba casi todas las celdas en blanco porque
unas pocas extremas se llevaban todo el rango. Entran los cinco modelos y la
escala pasa a clases discretas simetricas.

Soporte. Ocupaba una pagina entera para decir una sola cosa, y la rampa de
siete clases en rojo hacia que lo mas visible fuera la periferia sin datos,
justo lo que menos importa. Se reduce a las tres clases que corresponden a
una decision, dentro, margen y fuera, y ocupa un tercio del espacio.

Criterios comunes: clasificacion por cuantiles porque el precio y el error
son muy asimetricos y una rampa lineal deja el 80% de los predios en la
primera clase; limites parroquiales siempre visibles, porque sin ellos las
celdas flotan sobre un fondo sin referencias; y recorte al area con datos,
que evita dedicar dos tercios del papel al DMQ rural vacio.

Salidas: mapa_precio.png, mapa_error.png, mapa_sesgo.png, mapa_soporte.png
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).parent
DATOS = ROOT / "figures" / "aplicacion_valoracion" / "map_data_holdout.csv"
CAPAS = ROOT.parent / "capas" / "PARROQUIAS_F.shp"

plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 150})

# Nomenclatura vigente: la columna del gpkg conserva el nombre historico.
MODELOS = [("RF", "Random Forest"), ("SANNWR", "SANNWR"), ("GWR-27", "GWR"),
           ("GNNWR", "GNNWR"), ("OLS", "OLS")]

TIERRA = "#f2f5f7"
BORDE_DMQ = "#8d9aa4"
BORDE_PARR = "#d5dde2"
MARCO = "#3d4852"
CELDA_M = 1200  # ancho entre lados opuestos del hexagono
MIN_PREDIOS = 3


# ───────────────────────────── base cartografica ──────────────────────────

def cargar():
    d = pd.read_csv(DATOS)
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.x, d.y), crs="EPSG:32717")
    parr = gpd.read_file(CAPAS).to_crs(32717)
    return g, parr, parr.geometry.union_all()


def marco(ax, dmq, parr, extension):
    """Fondo comun: silueta del DMQ, parroquias y recorte al area con datos."""
    gpd.GeoSeries([dmq], crs=32717).plot(ax=ax, facecolor=TIERRA,
                                         edgecolor=BORDE_DMQ, lw=0.7, zorder=0)
    parr.boundary.plot(ax=ax, color=BORDE_PARR, linewidth=0.25, alpha=0.75,
                       zorder=1)
    ax.set_xlim(*extension[0])
    ax.set_ylim(*extension[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MARCO)
        s.set_linewidth(0.8)


def extension_datos(g, margen=1200):
    x, y = g.geometry.x, g.geometry.y
    xmin, xmax = np.percentile(x, [0.5, 99.5])
    ymin, ymax = np.percentile(y, [0.5, 99.5])
    return (xmin - margen, xmax + margen), (ymin - margen, ymax + margen)


def escala(ax, largo_m=5000):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xi = x0 + 0.06 * (x1 - x0)
    yi = y0 + 0.05 * (y1 - y0)
    ax.plot([xi, xi + largo_m], [yi, yi], color="#2b2b2b", lw=2.6,
            solid_capstyle="butt", zorder=6)
    ax.text(xi + largo_m / 2, yi + 0.016 * (y1 - y0), f"{largo_m // 1000} km",
            ha="center", va="bottom", fontsize=8, color="#2b2b2b", zorder=6)


def norte(ax):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x = x0 + 0.90 * (x1 - x0)
    y = y0 + 0.95 * (y1 - y0)
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.075 * (y1 - y0)),
                ha="center", va="center", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#2b2b2b", lw=0.9),
                zorder=6)


# Cortes en dolares por metro cuadrado, comunes al mapa de error y al de
# sesgo. Redondos a proposito: los cuantiles reparten las celdas en cinco
# grupos iguales, pero estiran el ultimo hasta el maximo, con lo que sale un
# "68 a 652" que no informa, y dan cortes distintos en cada figura, de modo
# que las dos dejan de ser comparables. Con estos el reparto sigue siendo
# razonable, 9, 31, 32, 19 y 9 por ciento de las celdas.
CORTES_ERROR = [0, 10, 25, 50, 100, np.inf]
CORTES_SESGO = [-np.inf, -50, -10, 10, 50, np.inf]


def cortes_cuantiles(valores, k=5):
    """Clases por cuantiles. Se conserva para el mapa de precio, donde no hay
    una escala de referencia externa y el reparto por igual numero de predios
    es lo que permite distinguir el gradiente urbano."""
    q = np.nanquantile(valores, np.linspace(0, 1, k + 1))
    q[0], q[-1] = np.nanmin(valores), np.nanmax(valores)
    return np.unique(q)


def rotulos_error():
    return ["menos de 10", "10 a 25", "25 a 50", "50 a 100", "más de 100"]


def rotulos_sesgo():
    return ["sobreestima más de 50", "sobreestima 10 a 50", "ajustada (± 10)",
            "subestima 10 a 50", "subestima más de 50"]


def rotulo(a, b, sufijo=""):
    fmt = (lambda v: f"{v:,.0f}") if abs(b) >= 10 else (lambda v: f"{v:.1f}")
    return f"{fmt(a)} – {fmt(b)}{sufijo}"


def leyenda_clases(fig, cortes, colores, titulo, ncol=None, y=0.055):
    parches = [Patch(facecolor=c, edgecolor="#ffffff", linewidth=0.4,
                     label=rotulo(cortes[i], cortes[i + 1]))
               for i, c in enumerate(colores)]
    leg = fig.legend(handles=parches, loc="lower center",
                     ncol=ncol or len(parches), frameon=False,
                     bbox_to_anchor=(0.5, y), fontsize=9,
                     handlelength=1.5, handleheight=1.0,
                     columnspacing=1.4, title=titulo)
    leg.get_title().set_fontsize(9.5)
    return leg


def leyenda_hueco(ax, cortes, colores, titulo):
    """Coloca la leyenda en el panel que sobra de la retilla, en lugar de
    debajo: asi no roba altura a los mapas."""
    parches = [Patch(facecolor=c, edgecolor="#b9c2c8", linewidth=0.3,
                     label=rotulo(cortes[i], cortes[i + 1]))
               for i, c in enumerate(colores)]
    leg = ax.legend(handles=parches, loc="center", ncol=1, frameon=False,
                    fontsize=8.5, handlelength=1.5, handleheight=1.0,
                    title=titulo)
    leg.get_title().set_fontsize(9)
    return leg


def leyenda_rotulos(ax, rotulos, colores, titulo):
    """Como leyenda_hueco, pero con los rotulos escritos en vez de deducidos
    de los cortes: con clases abiertas por un extremo, "mas de 100" se lee
    mejor que "100 a 652"."""
    parches = [Patch(facecolor=c, edgecolor="#b9c2c8", linewidth=0.3, label=r)
               for c, r in zip(colores, rotulos)]
    leg = ax.legend(handles=parches, loc="center", ncol=1, frameon=False,
                    fontsize=8.5, handlelength=1.5, handleheight=1.0,
                    title=titulo)
    leg.get_title().set_fontsize(9)
    return leg


def rejilla(g, columnas, resumen="mean"):
    """Agrega a celdas hexagonales. Promedia observaciones medidas dentro de
    cada celda; no interpola, de modo que no aparece ningun valor donde no se
    evaluo ningun predio.

    El hexagono tesela mejor que el cuadrado: sus seis vecinos equidistan,
    de modo que no privilegia las direcciones norte-sur y este-oeste, y el
    conjunto se lee como superficie en vez de como mosaico."""
    r_hex = CELDA_M / np.sqrt(3)          # circunradio
    ancho, alto = np.sqrt(3) * r_hex, 1.5 * r_hex

    h = g.copy()
    fila = np.round(h.geometry.y / alto).astype(int)
    desfase = (fila % 2) * ancho / 2      # filas impares medio paso a la derecha
    col = np.round((h.geometry.x - desfase) / ancho).astype(int)
    h["fila"], h["col"] = fila, col

    agg = {c: resumen for c in columnas}
    agg["predio_join"] = "size"
    d = h.groupby(["fila", "col"]).agg(agg).rename(
        columns={"predio_join": "n"}).reset_index()
    d = d[d.n >= MIN_PREDIOS]

    ang = np.pi / 180 * np.array([90, 150, 210, 270, 330, 30])
    geom = []
    for f, c in zip(d.fila, d.col):
        cy = f * alto
        cx = c * ancho + (f % 2) * ancho / 2
        geom.append(Polygon(np.c_[cx + r_hex * np.cos(ang),
                                  cy + r_hex * np.sin(ang)]))
    return gpd.GeoDataFrame(d, geometry=geom, crs=g.crs)


def paneles(n, ext, col_in=2.55):
    """Dos filas de tres. En una sola fila de cinco cada mapa queda demasiado
    estrecho para distinguir las celdas; repartidos en dos filas cada panel
    triplica su area y el hueco sobrante aloja la leyenda.

    El alto se deriva de la proporcion del propio mapa, que es vertical
    (1:1.45). Fijandolo a ojo, la celda de la retilla sale apaisada y, al
    imponer la proporcion real, cada panel deja huecos a los lados."""
    razon = (ext[1][1] - ext[1][0]) / (ext[0][1] - ext[0][0])
    fig, axes = plt.subplots(2, 3, figsize=(3 * col_in,
                                            2 * col_in * razon + 0.55))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()


def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{nombre}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}.png / .pdf")


# ───────────────────────────────── mapas ──────────────────────────────────

def mapa_precio(g, parr, dmq, ext):
    """Precio observado y estimado por cada modelo, en clases de cuantiles."""
    cortes = cortes_cuantiles(g.valor_m2, k=5)
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0.08, 0.96, len(cortes) - 1)))
    norm = BoundaryNorm(cortes, cmap.N)

    capas = [("valor_m2", "Observado")] + \
            [(f"pred_{c}", n) for c, n in MODELOS]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.6))
    for ax, (col, nom) in zip(axes.ravel(), capas):
        marco(ax, dmq, parr, ext)
        g.plot(ax=ax, column=col, cmap=cmap, norm=norm, markersize=5.5,
               linewidth=0, zorder=3)
        ax.set_title(nom, fontsize=11.5, fontweight="bold", pad=6)
    escala(axes[0, 0])
    norte(axes[0, 2])
    fig.subplots_adjust(bottom=0.09, wspace=0.03, hspace=0.08)
    leyenda_clases(fig, cortes, [cmap(i) for i in range(cmap.N)],
                   "Precio del suelo (USD/m²), clases de igual número de predios",
                   y=0.005)
    guardar(fig, "mapa_precio")


def mapa_error(g, parr, dmq, ext):
    """Error absoluto agregado a celdas, con clases comunes a los cinco."""
    cols = [f"err_{c}" for c, _ in MODELOS]
    cel = rejilla(g, cols)
    print(f"  error: {len(cel)} celdas de {CELDA_M/1000:.1f} km "
          f"({cel.n.sum()} predios, {100*cel.n.sum()/len(g):.0f}%)")

    cmap = ListedColormap(plt.cm.YlOrRd(np.linspace(0.06, 0.92, 5)))
    norm = BoundaryNorm(CORTES_ERROR[:-1] + [1e9], cmap.N)

    fig, axes = paneles(5, ext)
    for ax, (col, nom) in zip(axes, MODELOS):
        marco(ax, dmq, parr, ext)
        cel.plot(ax=ax, column=f"err_{col}", cmap=cmap, norm=norm,
                 edgecolor="none", zorder=2)
        ax.set_title(nom, fontsize=11.5, fontweight="bold", pad=6)
        ax.text(0.035, 0.965, f"peor zona\n{cel[f'err_{col}'].max():.0f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="#3d4852", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff",
                          edgecolor="#d3dade", linewidth=0.5, alpha=0.9))
    escala(axes[0])
    norte(axes[2])
    fig.subplots_adjust(wspace=0.02, hspace=0.10, left=0.02, right=0.98,
                        top=0.95, bottom=0.02)
    leyenda_rotulos(axes[5], rotulos_error(), [cmap(i) for i in range(cmap.N)],
                    "Error absoluto medio\nde la zona (USD/m²)")
    guardar(fig, "mapa_error")


def mapa_sesgo(g, parr, dmq, ext):
    """Sesgo con signo, agregado a celdas, en clases divergentes simetricas."""
    cols = [f"resid_{c}" for c, _ in MODELOS]
    cel = rejilla(g, cols)
    cmap = ListedColormap(["#2c6fad", "#93bcda", "#f5e4c8", "#e79c86", "#b3402f"])
    norm = BoundaryNorm([-1e9] + CORTES_SESGO[1:-1] + [1e9], cmap.N)

    fig, axes = paneles(5, ext)
    for ax, (col, nom) in zip(axes, MODELOS):
        marco(ax, dmq, parr, ext)
        # Borde en todas las celdas: sin el, las de la clase central se
        # confunden con el fondo y los paneles parecen tener distinto numero
        # de celdas cuando en realidad son las mismas en los cinco.
        cel.plot(ax=ax, column=f"resid_{col}", cmap=cmap, norm=norm,
                 edgecolor="#b9c2c8", linewidth=0.25, zorder=2)
        ax.set_title(nom, fontsize=11.5, fontweight="bold", pad=6)
        s = cel[f"resid_{col}"]
        ax.text(0.035, 0.965, f"{100*(s > 0).mean():.0f}% de zonas\nsubestimadas",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="#3d4852", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff",
                          edgecolor="#d3dade", linewidth=0.5, alpha=0.9))
    escala(axes[0])
    norte(axes[2])

    parches = [Patch(facecolor=cmap(i), edgecolor="#b9c2c8", linewidth=0.3,
                     label=e) for i, e in enumerate(rotulos_sesgo())]
    fig.subplots_adjust(wspace=0.02, hspace=0.10, left=0.02, right=0.98,
                        top=0.95, bottom=0.02)
    leg = axes[5].legend(handles=parches, loc="center", ncol=1, frameon=False,
                         fontsize=8.5, handlelength=1.5, handleheight=1.0,
                         title="Diferencia media por zona\nentre el precio "
                               "observado y el estimado\n(USD/m²)")
    leg.get_title().set_fontsize(9)
    guardar(fig, "mapa_sesgo")


def mapa_soporte(parr, dmq):
    """Hasta donde llega el respaldo de la muestra sobre el territorio.

    Es una superficie y no una nube de puntos, y la distincion importa: la
    distancia al predio de entrenamiento mas cercano esta definida en
    cualquier punto del territorio y se calcula de forma exacta, de modo que
    dibujarla en una malla no interpola nada. Sobre los predios de prueba, en
    cambio, el mapa no dice nada util, porque el reparto los dejo intercalados
    con los de entrenamiento y el 90% cae dentro del soporte por construccion.
    La pregunta pertinente no es donde se evaluo sino donde podria valorarse
    un predio nuevo.

    Dos zonas y no tres. Se probo separar ademas la franja mas densa, la que
    queda a menos de la separacion tipica entre predios muestreados, pero
    cubre el 2% del distrito y se dibuja como un moteado ilegible mientras la
    clase exterior se come el 76% restante. Con una sola frontera, la del
    alcance de la dependencia espacial estimado en la Seccion 4.3.2, queda un
    limite que si se puede interpretar, y la densidad de la muestra se ve
    donde debe verse, en los propios predios dibujados encima.
    """
    from scipy.spatial import cKDTree
    from shapely import contains_xy

    PASO = 120
    ALCANCE_DEP = 2237    # alcance de la autocorrelacion residual, Seccion 4.3.2
    # El tono de "fuera" debe distinguirse del papel: si se acerca al
    # blanco, tres cuartas partes del distrito se leen como vacio.
    DENTRO, FUERA = "#4a7d9b", "#e4e0d6"

    d = gpd.read_file(ROOT / "datos" / "dataset.gpkg").to_crs(32717)
    d["predio_join"] = d["predio_join"].astype(int)
    sp = pd.read_csv(ROOT / "data_split" / "split.csv")
    sp["predio_join"] = sp["predio_join"].astype(int)
    d = d.merge(sp[["predio_join", "split"]], on="predio_join")
    tr = d[d.split == "train"]

    x0, y0, x1, y1 = dmq.bounds
    xs = np.arange(x0, x1 + PASO, PASO)
    ys = np.arange(y0, y1 + PASO, PASO)
    XX, YY = np.meshgrid(xs, ys)
    dist = cKDTree(np.c_[tr.geometry.x, tr.geometry.y]).query(
        np.c_[XX.ravel(), YY.ravel()])[0].reshape(XX.shape)

    # Solo cuenta el territorio del distrito, no el rectangulo que lo encierra.
    territorio = contains_xy(dmq, XX, YY)
    cubierto = float((dist[territorio] < ALCANCE_DEP).mean())

    razon = (y1 - y0) / (x1 - x0)
    fig, ax = plt.subplots(figsize=(5.2, 5.2 * razon + 0.95))
    # Sin el fondo comun: aqui las dos clases cubren el distrito entero y
    # una silueta debajo solo anadiria un tercer tono.
    ax.set_xlim(x0 - 900, x1 + 900)
    ax.set_ylim(y0 - 900, y1 + 900)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # Relleno por curvas de nivel: la distancia al predio mas cercano es una
    # funcion continua, de modo que la frontera puede trazarse interpolando a
    # lo largo de la malla en vez de escalonarse en pixeles.
    ax.contourf(xs, ys, np.where(territorio, dist, np.nan),
                levels=[0, ALCANCE_DEP, 1e9], colors=[DENTRO, FUERA],
                zorder=2, antialiased=True)
    parr.boundary.plot(ax=ax, color="#9aa5ad", linewidth=0.4, zorder=4)
    gpd.GeoSeries([dmq], crs=32717).boundary.plot(ax=ax, color="#2b3840",
                                                  linewidth=1.0, zorder=5)
    # Puntos muy pequenos y claros: son cuatro mil y en el corredor
    # consolidado se funden en una mancha si se agrandan.
    ax.scatter(tr.geometry.x, tr.geometry.y, s=0.35, c="#eaf1f5",
               linewidths=0, alpha=0.5, zorder=6)
    escala(ax)
    norte(ax)

    fig.subplots_adjust(bottom=0.125, top=0.99, left=0.02, right=0.98)
    leg = fig.legend(
        handles=[
            Patch(facecolor=DENTRO, edgecolor="#b9c2c8", linewidth=0.3,
                  label=f"Con respaldo: a menos de {ALCANCE_DEP:,} m de un "
                        f"predio observado   {100*cubierto:.0f}%"),
            Patch(facecolor=FUERA, edgecolor="#b9c2c8", linewidth=0.3,
                  label=f"Sin respaldo   {100*(1-cubierto):.0f}%"),
            # El punto de la leyenda debe verse como en el mapa: claro sobre
            # el azul de la zona con respaldo, no oscuro sobre el papel.
            Line2D([], [], marker="o", linestyle="", markersize=3.5,
                   markerfacecolor="#eaf1f5", markeredgecolor="#4a7d9b",
                   markeredgewidth=0.8,
                   label=f"predios observados ({len(tr):,})")],
        loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.005),
        fontsize=7.5, handlelength=1.4, handleheight=0.95,
        title="Respaldo muestral del territorio")
    leg.get_title().set_fontsize(8)
    guardar(fig, "mapa_soporte")
    print(f"    con respaldo {100*cubierto:.1f}%, sin respaldo "
          f"{100*(1-cubierto):.1f}%")


def main():
    g, parr, dmq = cargar()
    ext = extension_datos(g)
    print(f"  {len(g)} predios evaluados")
    mapa_precio(g, parr, dmq, ext)
    mapa_error(g, parr, dmq, ext)
    mapa_sesgo(g, parr, dmq, ext)
    mapa_soporte(parr, dmq)


if __name__ == "__main__":
    main()
