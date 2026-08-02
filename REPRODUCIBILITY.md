# Guía de reproducibilidad

## Alcance

El repositorio permite auditar el código, el protocolo experimental, las semillas y los resultados agregados sin acceso a información restringida. La reproducción numérica exacta requiere obtener de las instituciones competentes las mismas versiones de las fuentes descritas en `DATA_AVAILABILITY.md`.

## Entorno de referencia

- Python 3.11
- CPU para OLS, GWR, Random Forest y análisis
- NVIDIA RTX 4090 de 24 GB para las ejecuciones neuronales de referencia
- PyTorch 2.5.1
- CRS de modelado: EPSG:32717 (UTM zona 17S)

Las operaciones CUDA pueden producir pequeñas diferencias entre hardware aun con la misma semilla.

## Estructura esperada de fuentes

Defina `DMQ_DATA_DIR` apuntando a un directorio externo con esta estructura:

```text
DMQ_DATA_DIR/
├── procesados/inf_procesada.gdb
├── PUGS_DMOT/DMOT_1.gdb
├── PUGS_DMOT/DMOT_2.gdb
├── PUG_Ciudad_Linea/
│   └── plan_de_uso_y_gestión_del_suelo_2024.gdb/PUGS_2024.gdb
├── catastro/stp_sector_a.shp
├── Predio_variables.csv
└── CATASTRO_PREDIAL_diciembre/predio_pend.csv
```

`predio_pend.csv` es el producto tabular derivado del MDT institucional de 2010. Si se recibe el TIF original, primero debe reproducirse el cálculo zonal de pendiente media por predio.

## Pipeline

1. `data_pipeline/pipeline.py`: integra las fuentes y escribe `datos/dataset.gpkg`.
2. `data_split/create_split.py`: genera el conjunto de prueba fijo 80/20.
3. `spatial_cv/pipeline_bloques.py`: construye las particiones espaciales.
4. Los scripts de `modelos/` entrenan y generan métricas.
5. `analisis/` consolida inferencia, sensibilidad e interpretabilidad.
6. `figures/` reconstruye las figuras a partir de las salidas.

Los archivos derivados sensibles permanecen ignorados por Git.

## Esquemas principales de evaluación

- Conjunto de prueba del 20 %: 4.040 observaciones de entrenamiento y 1.011 de prueba; interpolación interna.
- Validación cruzada aleatoria: cinco particiones sobre el conjunto de entrenamiento.
- Validación por bloques espaciales: cinco bloques de 5.621,8 m; separación geográfica parcial.

### Análisis exploratorio de separación estricta

El código conserva un análisis complementario con una zona de exclusión mínima de 2.530 m alrededor de cada región de prueba. Este escenario no constituye un cuarto esquema principal ni alimenta las tablas finales de comparación. En la tesis definitiva, 2.530 m se utiliza como tamaño de celda del remuestreo espacial empleado para estimar intervalos de confianza.

## Semillas

- Conjunto de prueba, diez réplicas: `42, 2011, 456, 777, 2026, 99, 1234, 888, 314, 7`.
- Validación cruzada neuronal, cinco réplicas: `42, 2011, 456, 777, 2026`.
- Random Forest canónico: `42`.

## Retransformación

Los modelos predicen `log(valor_m2)`. Para volver a USD/m² se aplica el estimador de *smearing* de Duan calculado exclusivamente sobre residuos de entrenamiento:

```text
y_hat_usd = exp(y_hat_log) * mean(exp(residual_train))
```

## Resultados esperados

Las tablas de referencia están en `results/*.csv`. Los archivos de `results/raw/` conservan métricas agregadas de las ejecuciones, nunca predicciones individuales.

## Validación sin datos

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

Estos controles verifican sintaxis, ausencia de artefactos restringidos, rutas portables y coherencia de los resultados canónicos.
