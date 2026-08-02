# Ficha de modelos

## Propósito

Comparar modelos de predicción del precio de oferta del suelo urbano bajo interpolación y separación espacial en el Distrito Metropolitano de Quito.

## Modelos focales

- OLS: referencia lineal global.
- GWR: regresión local con ancho de banda seleccionado dentro de cada partición e intercepto no penalizado.
- Random Forest: control tabular no espacial fuerte.
- GNNWR: red que aprende pesos a partir de la proximidad geográfica.
- SANNWR: ponderación neuronal espacial y atributiva con `alpha=0.5`.

## Usos previstos

- Investigación sobre validación espacial y fuga de información geográfica.
- Comparación reproducible de arquitecturas bajo un protocolo común.
- Apoyo metodológico preliminar para diseñar evaluaciones catastrales.

## Usos no autorizados por la evidencia

- Avalúo catastral automático u oficial.
- Determinación de impuestos u obligaciones individuales.
- Predicción fuera del soporte geográfico observado sin revisión humana.
- Interpretación causal de la importancia de variables.

## Limitaciones principales

- Precios de oferta, no transacciones consumadas.
- Un solo período de mercado, sin validación temporal independiente.
- Cinco regiones espaciales y una única geometría de bloques.
- Multicolinealidad severa entre variables de accesibilidad.
- Autocorrelación residual significativa en todos los modelos focales.
- Tendencia a subestimar sectores de mayor valor.
- El conjunto de prueba fue reutilizado durante el desarrollo y se interpreta como evidencia descriptiva interna.

## Revisión humana

Ninguna predicción debe convertirse directamente en una decisión catastral. Cualquier aplicación institucional requiere validación externa, auditoría territorial, análisis de equidad, documentación normativa y revisión por especialistas del Municipio.
