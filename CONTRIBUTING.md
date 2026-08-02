# Contribuir

Se aceptan correcciones de documentación, pruebas y mejoras metodológicas que no requieran publicar datos restringidos.

Antes de proponer cambios:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

No incluya datos prediales, coordenadas individuales, claves catastrales, predicciones por observación, archivos institucionales ni rutas locales. Los resultados nuevos deben ser agregados y describir claramente su esquema de validación, unidad de dispersión, semillas y transformación de la variable objetivo.
