# Construcción del conjunto integrado

El pipeline integra secuencialmente investigación de mercado, susceptibilidad, NBI, infraestructura urbana, PUGS, catastro y pendiente. Las fuentes no están incluidas.

Configure su ubicación mediante `DMQ_DATA_DIR` y ejecute:

```bash
python data_pipeline/pipeline.py
```

Opciones disponibles:

```bash
python data_pipeline/pipeline.py --force
python data_pipeline/pipeline.py --step 8 9
python data_pipeline/pipeline.py --from 4
```

Los productos intermedios se escriben en `data_pipeline/output/` y el resultado final en `datos/`; ambas rutas están excluidas de Git.

La procedencia y las condiciones de acceso se documentan en [`../DATA_AVAILABILITY.md`](../DATA_AVAILABILITY.md).
