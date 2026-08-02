# Disponibilidad y procedencia de los datos

## Restricción de acceso

El conjunto integrado utilizado en la tesis no se distribuye en este repositorio. Contiene información predial georreferenciada, identificadores catastrales y observaciones de investigación de mercado del bienio 2024–2025.

El acceso a los datos originales debe solicitarse formalmente a la **Dirección Metropolitana de Catastro (DMC) del Municipio del Distrito Metropolitano de Quito**. La disponibilidad, alcance y condiciones de uso dependen exclusivamente de la autorización de esa institución y de las demás entidades productoras.

## Fuentes efectivamente utilizadas

| Fuente | Producto empleado | Uso en el estudio | Acceso |
|---|---|---|---|
| Dirección Metropolitana de Catastro | Investigación de mercado 2024–2025 | Precio de oferta observado por m² | Solicitud y autorización institucional |
| SIREC-Q | Base alfanumérica catastral, corte 2025 | Atributos físicos y constructivos | Solicitud y autorización institucional |
| DMOT / Municipio de Quito | `DMOT_1.gdb` y `DMOT_2.gdb` | Infraestructura, servicios, quebradas y susceptibilidad | Consultar disponibilidad municipal |
| Municipio del DMQ | `PUGS_2024.gdb` | Uso del suelo, COS y centralidades | Consultar PUGS y portales municipales |
| Atlas Socioeconómico DMQ / INEC | `stp_sector_a.shp`, basado en el Censo 2010 | Necesidades básicas insatisfechas | Consultar Atlas, INEC y Municipio |
| Dirección Metropolitana de Catastro | Modelo Digital del Terreno, corte 2010, resolución 5 m, SIRES-DMQ, formato TIF | Pendiente media del terreno por predio | Solicitud y autorización institucional |

El Modelo Digital del Terreno fue transformado al sistema de trabajo EPSG:32717 antes de derivar la pendiente media en grados por predio.

## Portales públicos de consulta

Algunas capas actuales o equivalentes pueden consultarse en:

- Gobierno Abierto Quito: <https://gobiernoabierto.quito.gob.ec/>
- Geoportal del Municipio de Quito: <https://geoquito.quito.gob.ec/portal/home/>
- Portal de Servicios Municipales: <https://servicios.quito.gob.ec/>
- Instituto de Investigaciones de la Ciudad: <https://investigaciones.quito.gob.ec/>
- Atlas Socioeconómico del DMQ: <https://geoquito.quito.gob.ec/portal/apps/storymaps/stories/5ac26be6de5e4400bf0573759c46865b>
- Portal Nacional de Datos Abiertos: <https://www.datosabiertos.gob.ec/>

Las versiones públicas pueden diferir en fecha de corte, cobertura, resolución, atributos y geometría. No se garantiza que permitan reconstruir exactamente las 5.051 observaciones utilizadas.

## Integridad y privacidad

Los archivos excluidos incluyen:

- `dataset.csv`, `dataset.gpkg` y cualquier GeoPackage derivado;
- claves catastrales y archivos de partición asociados a ellas;
- predicciones, residuos y errores por predio;
- `val_catas.csv` y la base alfanumérica completa;
- mapas o tablas que permitan reconstruir ubicaciones individuales;
- capas fuente GDB, SHP, TIF y productos intermedios.

Los CSV publicados en `results/` contienen únicamente métricas agregadas por modelo, estrategia, semilla o fold.
