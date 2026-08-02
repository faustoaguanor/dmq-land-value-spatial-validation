# Resultados publicados

Los CSV del nivel superior son las tablas canónicas de la tesis definitiva. `raw/` contiene métricas agregadas que respaldan esos resúmenes; se excluyen todas las predicciones, coordenadas, claves y errores individuales.

La desviación de la validación por bloques espaciales en `validacion_bloques_espaciales.csv` mide la heterogeneidad entre cinco regiones. La estabilidad entre semillas del conjunto de prueba se informa por separado en `estabilidad_conjunto_prueba.csv`.

Las demás tablas canónicas son `conjunto_prueba.csv`, `validacion_cruzada_aleatoria.csv` y `moran_conjunto_prueba.csv`.

Las figuras públicas se regeneran exclusivamente desde estas tablas mediante `figures/publication/generate_summary_figures.py`. No contienen geometrías ni resultados por observación.
