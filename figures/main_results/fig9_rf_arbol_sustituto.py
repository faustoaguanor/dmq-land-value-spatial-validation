# -*- coding: utf-8 -*-
"""
fig9_rf_arbol_sustituto.py
==========================
Arbol sustituto de Random Forest: las reglas de decision que el bosque aplica,
expresadas en una estructura legible.

Por que un arbol SUSTITUTO y no uno de los 300. La revision pidio ver las
reglas de decision sobre las variables. Dibujar un arbol cualquiera del
bosque no responde a eso: cada arbol individual se ajusta a una submuestra
distinta y es, por diseño, un predictor debil cuyas reglas no representan al
conjunto. Un arbol sustituto si: se ajusta un arbol poco profundo a las
PREDICCIONES del bosque, de modo que aprende a imitarlo, y su R2 frente al
bosque mide exactamente cuanta de su logica queda recogida. Esa cifra se
reporta en el pie, de manera que el lector sabe hasta donde puede fiarse de
la lectura.

Se limita a profundidad 3 porque el objetivo es la legibilidad: con mas
niveles el arbol gana fidelidad pero deja de poder leerse en una pagina.

Salida: figures/main_results/fig9_rf_arbol.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).parent

plt.rcParams.update({"font.family": "serif", "figure.dpi": 150})

PROFUNDIDAD = 3
# Sin el "Distancia a", que se repite en casi todos los nodos y ensancha las
# cajas; el pie de la figura lo dice una sola vez.
LEGIBLE = {
    "dist_cc": "Centros comerciales",
    "dist_parque_metro": "Parques metropolitanos",
    "dist_centr_metro": "Centralidad metropolitana",
    "dist_metro": "Metro",
    "dist_universidad": "Universidades",
    "dist_mercado_mayorista": "Mercado mayorista",
    "dist_industrial": "Zonas industriales",
    "dist_hospital": "Hospitales",
    "dist_quebrada": "Quebradas",
    "dist_plataforma_gub": "Plataforma Gubernamental",
    "log_area": "Superficie del lote (log)",
    "pc_pnbi": "NBI del sector",
    "cos_num": "Coef. de ocupación",
    "pendiente_grados": "Pendiente",
    "frente_m": "Frente del lote",
    "num_pisos": "Número de pisos",
    "suscept_codigo": "Susceptibilidad",
    "uso_suelo_cod": "Uso de suelo",
    "area_const_m2": "Área construida",
    "dist_centr_zonal": "Centralidad zonal",
    "dist_parque_zonal": "Parques zonales",
}


def recortar(ruta, margen=12):
    """Deja la imagen ajustada a lo que no es blanco.

    plot_tree dibuja dentro de unos ejes cuyo alto no se adapta al contenido,
    y bbox_inches="tight" no quita esas franjas porque para matplotlib son
    parte de los ejes. Recortarlas aqui es lo que permite que el arbol ocupe
    de verdad el ancho de la caja de texto.
    """
    from PIL import Image, ImageChops

    im = Image.open(ruta).convert("RGB")
    fondo = Image.new("RGB", im.size, (255, 255, 255))
    caja = ImageChops.difference(im, fondo).getbbox()
    if caja:
        x0, y0, x1, y1 = caja
        im.crop((max(0, x0 - margen), max(0, y0 - margen),
                 min(im.width, x1 + margen),
                 min(im.height, y1 + margen))).save(ruta)


def main():
    gdf = gpd.read_file(ROOT / "datos" / "dataset.gpkg")
    gdf["predio_join"] = gdf["predio_join"].astype(int)
    split = pd.read_csv(ROOT / "data_split" / "split.csv")
    split["predio_join"] = split["predio_join"].astype(int)
    df = gdf.merge(split[["predio_join", "split"]], on="predio_join", how="inner")

    excl = {"predio_join", "valor_m2", "split", "geometry"}
    num = [c for c in df.columns
           if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
    X = pd.get_dummies(df[num].copy(), drop_first=False)
    y = np.log(df["valor_m2"].values)
    tr = (df["split"] == "train").values

    bosque = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    bosque.fit(X[tr], y[tr])
    pred_bosque = bosque.predict(X[tr])
    print(f"  bosque de 300 arboles ajustado sobre {tr.sum()} predios")

    # El arbol aprende a imitar al bosque, no a los datos.
    arbol = DecisionTreeRegressor(max_depth=PROFUNDIDAD, random_state=42)
    arbol.fit(X[tr], pred_bosque)
    fidelidad = r2_score(pred_bosque, arbol.predict(X[tr]))
    print(f"  arbol sustituto de profundidad {PROFUNDIDAD}: "
          f"R2 frente al bosque = {fidelidad:.3f}")

    nombres = [LEGIBLE.get(c, c.replace("_", " ")) for c in X.columns]

    # Se dibuja ya al ancho con que se imprime: hacerlo mayor y reducirlo
    # despues encoge el tipo hasta volverlo ilegible.
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    plot_tree(arbol, feature_names=nombres, filled=True, rounded=True,
              impurity=False, proportion=True, precision=2, fontsize=8, ax=ax,
              node_ids=False)

    RAMAS = {"true": "sí", "false": "no"}
    for txt in ax.texts:
        crudo = txt.get_text()
        if crudo.strip().lower() in RAMAS:      # rotulos de rama, en ingles
            txt.set_text(RAMAS[crudo.strip().lower()])
            continue
        lineas = []
        for s in crudo.split("\n"):
            # El arbol predice en logaritmo; se traduce a dolares. Se parte la
            # linea en vez de reconstruirla, porque scikit-learn no siempre
            # escribe el mismo numero de decimales y una sustitucion literal
            # falla en silencio.
            if s.startswith("value = "):
                v = float(s.split("=", 1)[1].strip("[] "))
                s = f"{np.exp(v):,.0f} USD/m²"   # sin "valor =": el pie lo dice
            elif s.startswith("samples = "):
                s = s.split("=", 1)[1].strip() + " de predios"
            elif " <= " in s:
                # El umbral viene con dos decimales, que son centimetros.
                var, corte = s.rsplit(" <= ", 1)
                s = f"{var} \u2264 {float(corte):,.0f}"
            lineas.append(s)
        txt.set_text("\n".join(lineas))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig9_rf_arbol.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    recortar(OUT / "fig9_rf_arbol.png")
    print("  -> fig9_rf_arbol.png / .pdf")

    print("\n  Variables que el arbol usa para partir, por nivel:")
    t = arbol.tree_
    for i in range(t.node_count):
        if t.children_left[i] != -1:
            print(f"    nodo {i}: {nombres[t.feature[i]]} <= {t.threshold[i]:,.1f}")


if __name__ == "__main__":
    main()
