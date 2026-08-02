# Resultados publicados

Los CSV del nivel superior son las tablas canónicas de la tesis definitiva. `raw/` contiene métricas agregadas que respaldan esos resúmenes; se excluyen todas las predicciones, coordenadas, claves y errores individuales.

La desviación de SpatialBlock en `spatial_block.csv` mide heterogeneidad entre cinco regiones. La estabilidad entre semillas se informa por separado en `stability_holdout.csv`.

Las figuras públicas se regeneran exclusivamente desde estas tablas mediante `figures/publication/generate_summary_figures.py`. No contienen geometrías ni resultados por observación.
