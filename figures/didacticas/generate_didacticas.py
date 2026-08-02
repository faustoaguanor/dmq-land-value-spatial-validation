"""
generate_didacticas.py
=======================
Figuras didacticas para audiencia NO geografa (comite/jurado de Ciencia de Datos).
Explican, con esquemas y mapas reales, los conceptos geoespaciales centrales de la
tesis: division de muestras, semivariograma, como funciona GWR y su evolucion, y
que miden las variables predictoras.

Estas figuras son COMPLEMENTARIAS a figures/main_results/ (que reporta resultados).
Aqui no se reporta ningun resultado nuevo: D1, D3, D4 (panel 1), D6 usan datos reales
del repositorio (dataset.gpkg, split.csv, fold_assignments.csv, variograma en vivo).
D2 y el panel A de D5 son esquemas puramente conceptuales (datos sinteticos), y se
etiquetan como tales en el titulo/caption.

Salidas: figures/didacticas/*.png (+ .pdf)
  D1 — area_estudio
  D2 — leakage_esquema
  D3 — division_muestras
  D4 — variograma_anotado
  D5 — modelos_geograficos
  D6 — variables_mapeadas
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Wedge
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap,
                               ListedColormap, LogNorm)
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "modelos"))

OUT_DIR = Path(__file__).parent
CRS_UTM = "EPSG:32717"

# ── Estilo compartido (coherente con figures/main_results/) ──────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    10.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colores de esquemas de validacion (fresco->calido = interpolacion->extrapolacion)
# Validado con scripts/validate_palette.js (dataviz skill): PASS todos los checks.
C_HOLDOUT = "#1B7837"   # verde
C_RKF     = "#2166AC"   # azul
C_SB      = "#EF6C00"   # naranja
C_SBBUF   = "#C62828"   # rojo
C_TRAIN   = "#B8C4CC"   # gris-azulado claro (puntos de entrenamiento no resaltados)
C_EXCL    = "#757575"   # gris (puntos excluidos por buffer)

# Colores de modelos (reutilizados de figures/main_results/generate_figures.py
# para consistencia visual con las figuras de resultados ya existentes)
COLOR_MODEL = {
    "OLS":    "#9E9E9E",
    "MLP":    "#00897B",
    "GWR":    "#607D8B",
    "GWR-27": "#607D8B",
    "GNNWR":  "#2196F3",
    "SANNWR": "#4CAF50",
    "GSAWR":  "#9C27B0",
}

# Rampa secuencial azul (steps del palette.md validado del skill de dataviz).
# Se conserva para las figuras esquematicas (D5), donde el azul codifica un peso
# y no compite con un mapa base.
BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
)

# Rampa de los MAPAS coropleticos (D1, D6). Cambiada por observacion del tutor:
# el azul secuencial se confundia con los tonos grises del mapa base y sus
# valores altos quedaban indistinguibles entre si. 'viridis' es perceptualmente
# uniforme (pasos iguales de color = pasos iguales de valor), legible en escala
# de grises al imprimir y segura para daltonismo, y su extremo amarillo destaca
# los predios de mayor valor sobre el fondo neutro.
MAP_SEQ = plt.cm.viridis


def load_dataset_utm() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(ROOT / "datos" / "dataset.gpkg")
    return gdf.to_crs(CRS_UTM)


PARROQUIAS_CANDIDATES = [
    Path(os.environ["DMQ_PARROQUIAS_SHP"]) if os.environ.get("DMQ_PARROQUIAS_SHP") else None,
    ROOT / "capas" / "PARROQUIAS_F.shp",
    ROOT.parent / "capas" / "PARROQUIAS_F.shp",
]


# ── Paleta cartografica (armoniza con el azul de datos: gris-pizarra calido) ──
# Aclarados respecto de la version anterior: el gris del mapa base competia
# con los datos. El fondo debe leerse como contexto, no como informacion.
MAP_LAND    = "#f7f9fa"   # relleno de la masa "terrestre" (DMQ disuelto)
MAP_VOID    = "#ffffff"   # fondo fuera del DMQ (vacio, no es un dato)
MAP_OUTER   = "#93a1ab"   # contorno exterior del DMQ (jerarquia: el mas fuerte)
MAP_INNER   = "#dde3e7"   # limites internos de parroquia (subordinados, tenues)
MAP_FRAME   = "#3d4852"   # marco (neatline) que encuadra el mapa


def load_parroquias_utm():
    """Limites de TODAS las parroquias del DMQ (contexto geografico de fondo).

    No se recorta la geometria: cualquier parroquia que quede fuera del area
    visible simplemente se corta en la VISUALIZACION (via set_view_extent),
    nunca se elimina del GeoDataFrame. Evita el efecto de "parroquias
    desaparecidas" que deja huecos o bordes abruptos en el mapa.
    """
    shp = next((p for p in PARROQUIAS_CANDIDATES if p is not None and p.exists()), None)
    if shp is None:
        print("[WARN] no se encontro PARROQUIAS_F.shp; se omite el mapa base de parroquias. "
              "Defina DMQ_PARROQUIAS_SHP o ubique capas/PARROQUIAS_F.shp junto al proyecto.")
        return None
    try:
        g = gpd.read_file(shp).to_crs(CRS_UTM)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] no se pudo cargar parroquias desde {shp}: {e}")
        return None
    return g


def dissolve_dmq(parroquias):
    """Disuelve las parroquias en un unico poligono: la 'masa continental'
    del DMQ, usada como relleno de fondo (figura vs. fondo)."""
    if parroquias is None:
        return None
    return parroquias.geometry.union_all()


def draw_basemap(ax, parroquias, dmq_union=None, lw_inner=0.45, lw_outer=1.1):
    """Dibuja un mapa base cartografico de verdad: relleno de tierra (para que
    el DMQ se lea como una masa solida, no como lineas flotando en blanco),
    contorno exterior mas fuerte (jerarquia visual) y limites internos de
    parroquia tenues y subordinados. Todo en tonos neutros para no competir
    con los datos (puntos azules) que van encima."""
    if parroquias is None:
        return
    if dmq_union is not None:
        gpd.GeoSeries([dmq_union], crs=parroquias.crs).plot(
            ax=ax, facecolor=MAP_LAND, edgecolor="none", zorder=0)
        gpd.GeoSeries([dmq_union], crs=parroquias.crs).boundary.plot(
            ax=ax, color=MAP_OUTER, linewidth=lw_outer, zorder=1.6)
    parroquias.boundary.plot(ax=ax, color=MAP_INNER, linewidth=lw_inner, zorder=1)


def set_view_extent(ax, gdf, margin_m=2000, recorte_pct=None):
    """Acota la VISTA (xlim/ylim) a la extension de los predios + margen,
    sin tocar los datos de fondo (parroquias). Esto es lo que produce el
    'corte en la visualizacion' en vez de eliminar geometria.

    Con recorte_pct=(lo, hi) la vista se ajusta al percentil indicado de las
    coordenadas en vez de al minimo y maximo. Sirve cuando un puñado de
    predios muy alejados estira el recuadro y empequeñece a los demas: se
    prefiere ampliar la zona densa aunque unos pocos puntos queden fuera del
    encuadre. Devuelve cuantos predios quedaron fuera, para declararlo.
    """
    x, y = gdf.geometry.x.values, gdf.geometry.y.values
    if recorte_pct is None:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        fuera = 0
    else:
        lo, hi = recorte_pct
        xmin, xmax = np.percentile(x, [lo, hi])
        ymin, ymax = np.percentile(y, [lo, hi])
        fuera = int(((x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)).sum())
    ax.set_xlim(xmin - margin_m, xmax + margin_m)
    ax.set_ylim(ymin - margin_m, ymax + margin_m)
    return fuera


def clases_por_cuantiles(v, n=5):
    """Cortes de clase por cuantiles para un mapa coropletico.

    Con escala lineal continua, una variable sesgada deja casi todos los
    predios en el extremo oscuro de la rampa y el mapa se ve de un solo
    color: en este conjunto, el 88.6% de los predios cae en el cuartil
    inferior de la escala de NBI y el 76.6% en la de pendiente. Repartiendo
    las observaciones en clases de igual frecuencia, cada color agrupa
    aproximadamente el mismo numero de predios y el contraste geografico
    vuelve a ser visible. Es el criterio estandar de clasificacion en
    cartografia tematica.
    """
    cortes = np.unique(np.percentile(v, np.linspace(0, 100, n + 1)))
    if len(cortes) < 3:            # variable casi constante: sin clasificar
        return None
    return cortes


def fmt_corte(c):
    """Etiqueta de un corte de clase, sin decimales inutiles."""
    if abs(c) >= 100:
        return f"{c:,.0f}"
    if abs(c) >= 10:
        return f"{c:.0f}"
    return f"{c:.2f}".rstrip("0").rstrip(".")


def style_map_frame(ax, show_frame=True):
    """Trata los ejes como un MAPA, no como un grafico de dispersion: sin
    ticks ni coordenadas numericas (la escala grafica ya cumple esa funcion),
    sin grilla, con un marco (neatline) fino y uniforme en las 4 caras."""
    ax.set_facecolor(MAP_VOID)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.grid(False)
    for s in ax.spines.values():
        if show_frame:
            s.set_visible(True)
            s.set_color(MAP_FRAME)
            s.set_linewidth(0.9)
        else:
            s.set_visible(False)


def add_scalebar(ax, length_m, label, loc=(0.04, 0.045)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    tick_h = 0.008 * (y1 - y0)
    # linea principal + ticks verticales en los extremos (estilo topografico)
    ax.plot([bx, bx + length_m], [by, by], color=MAP_FRAME, lw=1.3,
            solid_capstyle="butt", zorder=10)
    for xv in (bx, bx + length_m / 2, bx + length_m):
        ax.plot([xv, xv], [by - tick_h, by + tick_h], color=MAP_FRAME, lw=1.1, zorder=10)
    ax.text(bx + length_m / 2, by + 0.016 * (y1 - y0), label,
            ha="center", va="bottom", fontsize=7.8, color=MAP_FRAME, zorder=10)


def add_north_arrow(ax, loc=(0.945, 0.88)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax_x = x0 + loc[0] * (x1 - x0)
    ax_y = y0 + loc[1] * (y1 - y0)
    dy = 0.042 * (y1 - y0)
    ax.annotate("", xy=(ax_x, ax_y + dy), xytext=(ax_x, ax_y),
                arrowprops=dict(arrowstyle="-|>", color=MAP_FRAME, lw=1.3,
                                mutation_scale=14), zorder=10)
    ax.text(ax_x, ax_y - 0.014 * (y1 - y0), "N", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=MAP_FRAME, zorder=10)


# =============================================================================
# D1 — Area de estudio
# =============================================================================
def fig_d1_area_estudio(gdf: gpd.GeoDataFrame, parroquias=None, dmq_union=None):
    fig, ax = plt.subplots(figsize=(7.6, 8.2))

    # Mapa base: relleno de tierra + jerarquia de contornos (nunca recorta geometria).
    draw_basemap(ax, parroquias, dmq_union)

    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    v = gdf["valor_m2"].values

    # Clases de igual frecuencia en vez de rampa continua: el precio esta muy
    # concentrado en valores bajos (el 94% de los predios cae en el cuartil
    # inferior de una escala lineal) y la rampa continua dejaba indistinguibles
    # entre si a casi todos los predios de la periferia.
    cortes = clases_por_cuantiles(v, 6)
    k = len(cortes) - 1
    cmap = ListedColormap(MAP_SEQ(np.linspace(0.06, 0.96, k)))
    orden = np.random.default_rng(42).permutation(len(v))   # evita sesgo de dibujo
    sc = ax.scatter(x[orden], y[orden], c=v[orden], cmap=cmap,
                    norm=BoundaryNorm(cortes, k), s=7, alpha=0.9,
                    linewidths=0, zorder=2)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.025, aspect=22,
                        boundaries=cortes, ticks=cortes, spacing="uniform")
    cbar.set_ticklabels([fmt_corte(c) for c in cortes])
    cbar.set_label(f"Precio de oferta (USD/m²)\n{k} clases de ~{len(v)//k} predios", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_edgecolor(MAP_FRAME)
    cbar.outline.set_linewidth(0.7)

    ax.set_title("Área de estudio: 5,051 predios en el Distrito Metropolitano de Quito",
                  fontsize=12.5, fontweight="bold", color=MAP_FRAME, pad=12)
    ax.set_aspect("equal")

    # Encuadre al percentil 1-99 y no al minimo-maximo: un puñado de predios
    # muy al norte estiraba el recuadro y dejaba el 37% del alto en blanco,
    # con la zona densa reducida a una franja. Se prefiere ampliar donde estan
    # los datos aunque unos pocos puntos queden fuera del encuadre, y se
    # declara cuantos son.
    fuera = set_view_extent(ax, gdf, margin_m=900, recorte_pct=(1, 99))
    style_map_frame(ax)

    add_scalebar(ax, 5000, "5 km")
    add_north_arrow(ax)

    fig.text(0.5, 0.005,
              f"Elaboración propia (n=5,051 predios, EPSG:32717). Límites de parroquias del DMQ como referencia "
              f"geográfica. El encuadre\nprioriza la zona muestreada: {fuera} predios dispersos en el extremo norte y "
              f"sur quedan fuera del recuadro, y el territorio\ncontinúa más allá de él.",
              ha="center", fontsize=7.5, color="#666666", style="italic")
    print(f"     (D1: {fuera} predios fuera del encuadre p1-p99)")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d1_area_estudio.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("[OK] D1 area_estudio")


# =============================================================================
# D2 — El problema: spatial data leakage (esquema conceptual, datos sinteticos)
# =============================================================================
def fig_d2_leakage_esquema():
    rng = np.random.default_rng(7)
    n = 160
    pts = rng.uniform(0, 10, size=(n, 2))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    note_bbox = dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="#666", linewidth=0.7, alpha=0.9)

    # --- Panel A: particion aleatoria ---
    ax = axes[0]
    test_idx = rng.choice(n, size=int(n * 0.2), replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False

    ax.scatter(pts[train_mask, 0], pts[train_mask, 1], s=26, color=C_RKF,
               alpha=0.7, label="Entrenamiento", zorder=2)
    ax.scatter(pts[~train_mask, 0], pts[~train_mask, 1], s=26, color=C_SB,
               alpha=0.9, label="Prueba", zorder=3)

    # Punto de test de foco: el de test mas cercano al centro (zona con espacio arriba).
    focus = pts[test_idx][np.argmin(np.linalg.norm(pts[test_idx] - np.array([5.0, 5.5]), axis=1))]
    d = np.linalg.norm(pts[train_mask] - focus, axis=1)
    nearest3 = pts[train_mask][np.argsort(d)[:3]]
    ax.scatter(*focus, s=180, facecolor="none", edgecolor="black", linewidths=1.8, zorder=5)
    for p in nearest3:
        ax.plot([focus[0], p[0]], [focus[1], p[1]], color="black", lw=1.2,
                linestyle=":", zorder=4)
    # Anotacion LOCAL: offset en puntos desde el foco -> flecha corta, no cruza la figura.
    ax.annotate("El punto de prueba tiene\nvecinos de entrenamiento\na muy corta distancia\n→ fuga de información",
                xy=focus, xytext=(38, 40), textcoords="offset points",
                fontsize=8.5, color="#222", ha="left", va="bottom", bbox=note_bbox,
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.0,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_title("(a) Partición aleatoria (RandomKFold)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_xlim(-0.4, 10.4); ax.set_ylim(-0.4, 10.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # --- Panel B: particion por bloques ---
    ax = axes[1]
    block_id = (pts[:, 0] // 3.4).astype(int) + (pts[:, 1] // 3.4).astype(int) * 3
    test_block = 4  # bloque central
    is_test = block_id == test_block

    # sombrear el bloque de prueba (bloque central: x,y en [3.4, 6.8)) exacto
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((3.4, 3.4), 3.4, 3.4, facecolor=C_SB, alpha=0.08,
                           edgecolor="none", zorder=0))
    ax.scatter(pts[~is_test, 0], pts[~is_test, 1], s=26, color=C_RKF, alpha=0.7,
               label="Entrenamiento", zorder=2)
    ax.scatter(pts[is_test, 0], pts[is_test, 1], s=26, color=C_SB, alpha=0.9,
               label="Prueba (un bloque)", zorder=3)

    for gx in np.arange(0, 10.1, 3.4):
        ax.axvline(gx, color="#999", lw=0.9, linestyle="-", zorder=1)
    for gy in np.arange(0, 10.1, 3.4):
        ax.axhline(gy, color="#999", lw=0.9, linestyle="-", zorder=1)

    # Foco: punto de test cerca del borde derecho del bloque, cuyos vecinos de train
    # quedan al otro lado del borde.
    focus2 = pts[is_test][np.argmin(np.linalg.norm(pts[is_test] - np.array([6.4, 5.2]), axis=1))]
    d2 = np.linalg.norm(pts[~is_test] - focus2, axis=1)
    nearest3b = pts[~is_test][np.argsort(d2)[:3]]
    ax.scatter(*focus2, s=180, facecolor="none", edgecolor="black", linewidths=1.8, zorder=5)
    for p in nearest3b:
        ax.plot([focus2[0], p[0]], [focus2[1], p[1]], color="black", lw=1.2,
                linestyle=":", zorder=4)
    ax.annotate("Sus vecinos de\nentrenamiento quedan\nen otro bloque\n→ separación geográfica",
                xy=focus2, xytext=(30, -70), textcoords="offset points",
                fontsize=8.5, color="#222", ha="left", va="top", bbox=note_bbox,
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.0,
                                connectionstyle="arc3,rad=-0.15"))

    ax.set_title("(b) Partición por bloques (SpatialBlock)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_xlim(-0.4, 10.4); ax.set_ylim(-0.4, 10.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig.suptitle("El problema: cómo la partición afecta la cercanía entre entrenamiento y prueba",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03,
              "Esquema ilustrativo con puntos sintéticos (n=160), no son datos del estudio. Objetivo: mostrar por qué la "
              "autocorrelación espacial\ncontamina la validación aleatoria (a) y cómo los bloques geográficos la mitigan (b).",
              ha="center", fontsize=8, color="#555555", style="italic")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d2_leakage_esquema.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("[OK] D2 leakage_esquema")


# =============================================================================
# D3 — Division de muestras: panel 2x2 sobre el mapa real
# =============================================================================
def fig_d3_division_muestras(gdf: gpd.GeoDataFrame, parroquias=None, dmq_union=None):
    split = pd.read_csv(ROOT / "data_split" / "split.csv")
    folds = pd.read_csv(ROOT / "spatial_cv" / "output" / "fold_assignments.csv")

    gdf = gdf.copy()
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    split["predio_join"] = split["predio_join"].astype(int)
    folds["predio_join"] = folds["predio_join"].astype(int)

    df = gdf.merge(split[["predio_join", "split"]], on="predio_join", how="left")
    df = df.merge(folds[["predio_join", "fold"]], on="predio_join", how="left")

    x = df.geometry.x.values
    y = df.geometry.y.values

    # SIMBOLOGIA (rehecha por observacion del tutor). El diseño anterior daba a
    # los predios de prueba un tamaño cuatro veces mayor que a los de
    # entrenamiento: como resultado el 20% de prueba tapaba al 80% restante y
    # los paneles (a) y (b) parecian decir que TODOS los predios son de prueba,
    # justo lo contrario del dato. Ahora ambos grupos comparten tamaño y la
    # distincion la lleva solo el color: el entrenamiento en un gris muy claro
    # que actua de fondo, la prueba en color saturado. Asi la proporcion real
    # entre ambos se lee de un vistazo.
    C_TR = "#d3d9dd"          # entrenamiento: presente pero subordinado
    S_PT = 5.5                # mismo tamaño para todos los grupos
    TITLE_C = MAP_FRAME

    # Ancho ajustado a la proporcion real de la nube de predios (0.67) mas el
    # espacio de las leyendas: con el ancho anterior cada panel quedaba con
    # franjas blancas a los lados y los mapas se veian mas pequeños de lo
    # necesario en la pagina impresa.
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 6.4))

    def base(ax, titulo, subtitulo):
        draw_basemap(ax, parroquias, dmq_union, lw_inner=0.3, lw_outer=0.85)
        ax.set_aspect("equal")
        set_view_extent(ax, gdf, margin_m=900, recorte_pct=(1, 99))
        style_map_frame(ax)
        ax.set_title(titulo, fontsize=11.5, fontweight="bold", color=TITLE_C, pad=22)
        ax.text(0.5, 1.035, subtitulo, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.8, color="#5f7280")

    def leg(ax, handles):
        """Leyenda FUERA del mapa. Antes iba dentro, abajo a la izquierda, y su
        recuadro tapaba predios del sur: en un mapa, la leyenda nunca debe
        ocultar el dato que explica."""
        lg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015),
                       frameon=False, fontsize=8.6, ncol=1, markerscale=1.5,
                       handletextpad=0.6, labelspacing=0.35)
        return lg

    def punto(color, label, **kw):
        return Line2D([], [], marker="o", linestyle="none", markersize=5.5,
                      markerfacecolor=color, markeredgecolor="none", label=label, **kw)

    def plot_te(ax, mask_te, color, label_tr, label_te):
        ax.scatter(x[~mask_te], y[~mask_te], s=S_PT, color=C_TR, alpha=1.0,
                   zorder=2, edgecolors="none")
        ax.scatter(x[mask_te], y[mask_te], s=S_PT, color=color, alpha=0.95, zorder=3,
                   edgecolors="none")
        return [punto(C_TR, label_tr), punto(color, label_te)]

    # (a) Particion reservada 80/20
    ax = axes[0]
    is_test = (df["split"] == "test").values
    h = plot_te(ax, is_test, C_HOLDOUT,
                f"Entrenamiento  ({(~is_test).sum():,})", f"Prueba  ({is_test.sum():,})")
    base(ax, "(a) Partición reservada", "reparto estratificado: la prueba se dispersa por todo el mapa")
    leg(ax, h)

    # (b) Reparto aleatorio — ilustrativo (K=5, para contraste visual)
    ax = axes[1]
    rng = np.random.default_rng(42)
    rkf_fold = rng.permutation(np.arange(len(df)) % 5)
    is_test_rkf = rkf_fold == 0
    h = plot_te(ax, is_test_rkf, C_RKF,
                f"Entrenamiento  ({(~is_test_rkf).sum():,})",
                f"Prueba, 1 de 5 grupos  ({is_test_rkf.sum():,})")
    base(ax, "(b) Reparto aleatorio, K=5", "sin criterio geográfico: cada predio se sortea por separado")
    leg(ax, h)

    # (c) Bloques espaciales — fold real 1 de 5
    TEST_FOLD = 1
    ax = axes[2]
    is_test_sb = (df["fold"] == TEST_FOLD).values
    h = plot_te(ax, is_test_sb, C_SB,
                f"Entrenamiento  ({(~is_test_sb).sum():,})",
                f"Prueba, bloque 1 de 5  ({is_test_sb.sum():,})")
    base(ax, "(c) Bloques espaciales, K=5", "zonas completas van juntas a prueba (bloques de 5.62 km)")
    leg(ax, h)

    # Sin titulo general ni nota al pie dentro de la imagen: los repetia el pie
    # de figura del documento y restaban alto a los mapas.
    fig.tight_layout(h_pad=1.4, w_pad=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d3_division_muestras.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("[OK] D3 division_muestras (3 esquemas)")


# =============================================================================
# D4 — Semivariograma anotado (recalculado en vivo, misma metodologia que
#      spatial_cv/estrategias_cv.py::estimate_autocorrelation_range)
# =============================================================================
def fig_d4_variograma_anotado(gdf: gpd.GeoDataFrame):
    try:
        import skgstat as skg
    except ImportError:
        print("[SKIP] D4: scikit-gstat no instalado")
        return

    coords_all = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    valor = gdf["valor_m2"].values

    # Panel 1: valor_m2 crudo (serie con tendencia -> rango 5,621.8 m, el usado
    # para SpatialBlock). Mismos parametros que estrategias_cv.py.
    V1 = skg.Variogram(coordinates=coords_all, values=valor, model="exponential",
                        maxlag="median", n_lags=25)
    range1 = V1.parameters[0]
    sill1 = V1.parameters[1]
    nugget1 = V1.parameters[2] if len(V1.parameters) > 2 else 0.0

    bins1 = V1.bins
    exp1 = V1.experimental

    # Panel 2: residuales OLS (serie estacionaria -> rango residual 1.5-2.3 km,
    # el usado para calibrar el buffer). Se recalcula OLS en vivo.
    from features import build_feature_matrix
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    Xf, _ = build_feature_matrix(gdf, one_hot=True)
    y_log = np.log(valor)
    Xs = StandardScaler().fit_transform(Xf)
    ols = LinearRegression().fit(Xs, y_log)
    resid = y_log - ols.predict(Xs)

    V2 = skg.Variogram(coordinates=coords_all, values=resid, model="exponential",
                        maxlag="median", n_lags=25)
    range2 = V2.parameters[0]
    bins2 = V2.bins
    exp2 = V2.experimental

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    def plot_variogram(ax, bins, exp, rango, sill, color, title, xlabel_extra=""):
        ax.scatter(bins, exp, s=22, color=color, alpha=0.75, zorder=3, label="Semivarianza empírica")
        # curva teorica exponencial suavizada
        xx = np.linspace(0, bins.max(), 200)
        if sill is not None:
            yy = sill * (1 - np.exp(-xx / (rango / 3)))
            ax.plot(xx, yy, color=color, lw=2, alpha=0.9, zorder=2, label="Modelo exponencial ajustado")
        ax.axvline(rango, color="#333", linestyle="--", lw=1.3, zorder=1)
        ax.annotate(f"rango ≈ {rango:,.0f} m", xy=(rango, ax.get_ylim()[1] * 0.05),
                    xytext=(rango * 1.05, ax.get_ylim()[1] * 0.15 if ax.get_ylim()[1] else 0),
                    fontsize=9, fontweight="bold", color="#222")
        ax.set_xlabel(f"Distancia (m){xlabel_extra}")
        ax.set_ylabel("Semivarianza γ(h)")
        ax.set_title(title, fontsize=10.8)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.grid(alpha=0.2, linestyle="--")

    # Los rotulos nombraban los esquemas de validacion de la version anterior,
    # incluido el de la franja de exclusion, retirado de la tesis. Cada panel
    # indica ahora para que sirve realmente el rango que estima.
    plot_variogram(axes[0], bins1, exp1, range1, sill1, C_SB,
                    "(1) Sobre el precio observado\nFija el tamaño de los bloques de validación")
    plot_variogram(axes[1], bins2, exp2, range2, np.nanmax(exp2) if len(exp2) else None, C_SBBUF,
                    "(2) Sobre los residuos de OLS\nDimensiona el remuestreo de los intervalos")

    fig.text(0.5, -0.05,
              f"Panel (1): rango de {range1:,.0f} m sobre el precio observado, que incluye la tendencia urbana "
              f"además del contagio entre vecinos.\n"
              f"Panel (2): rango de {range2:,.0f} m sobre los residuos de OLS, que aíslan el contagio local "
              f"porque ya tienen descontada esa tendencia.",
              ha="center", fontsize=8.5, color="#555555", style="italic")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d4_variograma_anotado.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] D4 variograma_anotado (rango1={range1:.1f}m, rango2={range2:.1f}m)")


# =============================================================================
# D5 — Los modelos geograficos: como funciona GWR + evolucion GWR->GSAWR
# =============================================================================
def fig_d5_modelos_geograficos():
    fig = plt.figure(figsize=(11.5, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[0.09, 1.15, 0.09, 0.62, 0.07])

    axLabelA = fig.add_subplot(gs[0])
    axLabelA.axis("off")
    axLabelA.text(0.5, 0.5, "(a) Cómo pondera GWR sus vecinos: cada predio vecino aporta según su distancia al punto que se estima",
                  ha="center", va="center", fontsize=11.5, fontweight="bold", transform=axLabelA.transAxes)

    # --- Panel A: como funciona GWR (kernel bisquare) ---
    gsA = gs[1].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)
    axA1 = fig.add_subplot(gsA[0])
    axA2 = fig.add_subplot(gsA[1])

    rng = np.random.default_rng(3)
    n = 70
    pts = rng.uniform(-1, 1, size=(n, 2)) * 6
    x0 = np.array([0, 0])
    d = np.linalg.norm(pts - x0, axis=1)
    bandwidth = 3.2

    def bisquare(d, bw):
        w = np.zeros_like(d)
        inside = d < bw
        w[inside] = (1 - (d[inside] / bw) ** 2) ** 2
        return w

    w = bisquare(d, bandwidth)
    sizes = 18 + 220 * w
    axA1.scatter(pts[:, 0], pts[:, 1], s=sizes, c=w, cmap=BLUE_SEQ, vmin=0, vmax=1,
                 edgecolor="#555", linewidths=0.3, zorder=3)
    circle = Circle(x0, bandwidth, facecolor="none", edgecolor="#C62828", lw=1.8,
                     linestyle="--", zorder=4)
    axA1.add_patch(circle)
    axA1.scatter(*x0, marker="X", s=160, color="#C62828", zorder=5,
                 label="Punto de estimación $x_0$")
    axA1.annotate("bandwidth", xy=(bandwidth * 0.72, bandwidth * 0.72), fontsize=8.5,
                  color="#C62828", fontweight="bold")
    axA1.set_xlim(-6.5, 6.5); axA1.set_ylim(-6.5, 6.5)
    axA1.set_aspect("equal")
    axA1.set_xticks([]); axA1.set_yticks([])
    for s in axA1.spines.values():
        s.set_visible(False)
    axA1.set_title("Vecindad ponderada:\npuntos más grandes/oscuros pesan más", fontsize=10)
    axA1.legend(loc="upper left", frameon=False, fontsize=8)

    dd = np.linspace(0, 5, 200)
    ww = bisquare(dd, bandwidth)
    axA2.plot(dd, ww, color=COLOR_MODEL["GWR"], lw=2.4)
    axA2.axvline(bandwidth, color="#C62828", linestyle="--", lw=1.3)
    axA2.annotate("bandwidth", xy=(bandwidth, 0.5), xytext=(bandwidth + 0.3, 0.6),
                  fontsize=9, color="#C62828", fontweight="bold")
    axA2.fill_between(dd, ww, alpha=0.12, color=COLOR_MODEL["GWR"])
    axA2.set_xlabel("Distancia al punto de estimación (km)")
    axA2.set_ylabel("Peso del kernel bisquare  $w(d)$")
    axA2.set_title("El peso decae suavemente con la distancia\ny se anula fuera del bandwidth", fontsize=10)
    axA2.grid(alpha=0.2, linestyle="--")

    axLabelB = fig.add_subplot(gs[2])
    axLabelB.axis("off")
    axLabelB.text(0.5, 0.5, "(b) Qué añade cada extensión respecto a la anterior (2020 → 2025)",
                  ha="center", va="center", fontsize=11.5, fontweight="bold", transform=axLabelB.transAxes)

    # --- Panel B: evolucion GWR -> GNNWR -> SANNWR -> GSAWR ---
    axB = fig.add_subplot(gs[3])
    axB.set_xlim(0, 10)
    axB.set_ylim(0.2, 2.75)
    axB.axis("off")

    boxes = [
        ("GWR", "Kernel bisquare fijo.\nEl peso se calcula con\nuna fórmula matemática\n(distancia → peso).", COLOR_MODEL["GWR"]),
        ("GNNWR", "+ Una red neuronal (SWNN)\naprende la función de\npesos directamente de\nlos datos.", COLOR_MODEL["GNNWR"]),
        ("SANNWR", "+ Combina la distancia\nespacial con la distancia\nen atributos (uso de suelo,\nCOS, pendiente…).", COLOR_MODEL["SANNWR"]),
        ("GSAWR", "+ Usa una grilla de\nreferencia y una red\nconvolucional (CNN) para\npatrones 2D de vecindad.", COLOR_MODEL["GSAWR"]),
    ]
    box_w, box_h = 2.05, 2.1
    xs = [0.35, 2.85, 5.35, 7.85]
    y0 = 0.45

    for (name, desc, color), bx in zip(boxes, xs):
        box = FancyBboxPatch((bx, y0), box_w, box_h,
                              boxstyle="round,pad=0.06,rounding_size=0.12",
                              facecolor=color, edgecolor="none", alpha=0.18, zorder=1)
        axB.add_patch(box)
        box_edge = FancyBboxPatch((bx, y0), box_w, box_h,
                                   boxstyle="round,pad=0.06,rounding_size=0.12",
                                   facecolor="none", edgecolor=color, lw=2.0, zorder=2)
        axB.add_patch(box_edge)
        axB.text(bx + box_w / 2, y0 + box_h - 0.32, name, ha="center", va="top",
                 fontsize=13, fontweight="bold", color=color, zorder=3)
        axB.text(bx + box_w / 2, y0 + box_h - 0.62, desc, ha="center", va="top",
                 fontsize=8.3, color="#222", zorder=3, linespacing=1.35)

    for i in range(3):
        arrow = FancyArrowPatch((xs[i] + box_w + 0.03, y0 + box_h / 2),
                                 (xs[i + 1] - 0.03, y0 + box_h / 2),
                                 arrowstyle="-|>", mutation_scale=22, color="#555", lw=1.8, zorder=1)
        axB.add_patch(arrow)

    axCaption = fig.add_subplot(gs[4])
    axCaption.axis("off")
    axCaption.text(0.5, 0.5,
              "Panel (a): esquema conceptual del mecanismo de ponderación de GWR (kernel bisquare, datos sintéticos). "
              "Panel (b): línea evolutiva de la familia de modelos evaluada en esta tesis.",
              ha="center", va="center", fontsize=8, color="#555555", style="italic",
              transform=axCaption.transAxes)

    fig.suptitle("Los modelos geográficamente ponderados: de un kernel fijo a redes que aprenden los pesos",
                 fontsize=13.5, fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d5_modelos_geograficos.{ext}")
    plt.close(fig)
    print("[OK] D5 modelos_geograficos")


# =============================================================================
# D6 — Las variables mapeadas (small multiples)
# =============================================================================
def fig_d6_variables_mapeadas(gdf: gpd.GeoDataFrame, parroquias=None, dmq_union=None):
    # Reducido de 8 paneles a 4 por observacion del tutor ("solo dejar 3
    # variables"): la variable objetivo y las TRES predictoras que el texto
    # discute, cada una ilustrando un patron espacial distinto (social,
    # topografico y normativo). Menos paneles = paneles mas grandes y legibles
    # en una sola pagina. Los titulos desarrollan la sigla, no la abrevian.
    variables = [
        ("valor_m2", "Precio de oferta\n(USD/m²)",
         "Lo que se quiere predecir: alto en el centro-norte"),
        ("pc_pnbi", "Necesidades Básicas\nInsatisfechas, NBI (%)",
         "Contraste social: alto en el sur, bajo en el centro-norte"),
        ("pendiente_grados", "Pendiente del terreno\n(grados)",
         "Relieve: dibuja la cordillera y las quebradas"),
        ("cos_num", "Coeficiente de Ocupación\ndel Suelo, COS (fracción)",
         "Norma: delimita el núcleo consolidado"),
    ]

    x = gdf.geometry.x.values
    y = gdf.geometry.y.values

    # La nube de predios es alta y estrecha (ancho/alto = 0.67). Si la figura no
    # respeta esa proporcion, matplotlib deja franjas blancas a los lados de cada
    # panel y los mapas se ven pequeños. El ancho se calcula desde el alto para
    # que cada panel quede lleno: 2 columnas x (alto_panel * 0.67 + barra de color).
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 10.0))
    axes = axes.ravel()

    # Con 5,051 puntos superpuestos, dibujarlos en el orden del archivo deja
    # arriba siempre a los ultimos registros y puede inventar un patron. Se
    # baraja el orden una sola vez, igual para los cuatro paneles.
    orden = np.random.default_rng(42).permutation(len(gdf))

    N_CLASES = 5
    for ax, (col, label, lectura) in zip(axes, variables):
        draw_basemap(ax, parroquias, dmq_union, lw_inner=0.3, lw_outer=0.85)
        v = gdf[col].values

        # Clasificacion por cuantiles en LOS CUATRO paneles: con escala lineal
        # continua, NBI y pendiente salian practicamente monocromas porque su
        # distribucion esta concentrada en valores bajos. Las clases de igual
        # frecuencia devuelven el contraste sin alterar el dato.
        cortes = clases_por_cuantiles(v, N_CLASES)
        if cortes is None:
            sc = ax.scatter(x[orden], y[orden], c=v[orden], cmap=MAP_SEQ, s=4.5,
                            alpha=0.9, linewidths=0, zorder=2)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02, aspect=20)
        else:
            k = len(cortes) - 1
            cmap = ListedColormap(MAP_SEQ(np.linspace(0.06, 0.96, k)))
            norm = BoundaryNorm(cortes, k)
            sc = ax.scatter(x[orden], y[orden], c=v[orden], cmap=cmap, norm=norm,
                            s=4.5, alpha=0.9, linewidths=0, zorder=2)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02, aspect=20,
                                boundaries=cortes, ticks=cortes, spacing="uniform")
            cbar.set_ticklabels([fmt_corte(c) for c in cortes])

        ax.set_aspect("equal")
        set_view_extent(ax, gdf, margin_m=900, recorte_pct=(1, 99))
        style_map_frame(ax)
        ax.set_title(label, fontsize=10.5, fontweight="bold", color=MAP_FRAME, pad=6)
        # Que debe verse en el panel, bajo el titulo: el mapa se lee solo.
        ax.text(0.5, -0.045, lectura, transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#555555", style="italic")
        cbar.ax.tick_params(labelsize=7.5, length=2)
        cbar.outline.set_edgecolor(MAP_FRAME)
        cbar.outline.set_linewidth(0.6)

    # Escala y norte una sola vez: los cuatro paneles cubren el mismo territorio.
    # Se separan del borde mas que en un mapa a pagina completa, porque en un
    # panel pequeño el rotulo quedaba pisando el marco.
    add_scalebar(axes[0], 5000, "5 km", loc=(0.07, 0.075))
    add_north_arrow(axes[0], loc=(0.90, 0.87))

    # Sin titulo general ni nota al pie dentro de la imagen: ambos repetian
    # literalmente el pie de figura del documento. Suprimirlos libera alto para
    # los mapas, que es lo que el lector necesita ver.
    fig.tight_layout(h_pad=0.9, w_pad=0.7)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"d6_variables_mapeadas.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("[OK] D6 variables_mapeadas")


# =============================================================================
if __name__ == "__main__":
    print("Cargando dataset...")
    gdf = load_dataset_utm()
    print(f"  {len(gdf)} predios, CRS={gdf.crs}")
    parroquias = load_parroquias_utm()
    print(f"  parroquias: {'cargadas (' + str(len(parroquias)) + ', completas, sin recortar)' if parroquias is not None else 'NO disponibles'}")
    dmq_union = dissolve_dmq(parroquias)

    fig_d1_area_estudio(gdf, parroquias, dmq_union)
    fig_d2_leakage_esquema()
    fig_d3_division_muestras(gdf, parroquias, dmq_union)
    fig_d4_variograma_anotado(gdf)
    fig_d5_modelos_geograficos()
    fig_d6_variables_mapeadas(gdf, parroquias, dmq_union)

    print("\nTodas las figuras generadas en", OUT_DIR)
