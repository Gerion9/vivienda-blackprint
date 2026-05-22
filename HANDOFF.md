# Pase de estafeta — Módulo de Vivienda CDMX

## TL;DR

`fact_avaluos_cdmx.py` es el pipeline de producción que materializa un reporte HTML editorial sobre el mercado de vivienda hipotecada en CDMX. Lee solo de `staging.stg_avaluos` (read-only), corre K-Means sobre atributos físicos + geográficos + POIs + NSE, calcula plusvalía nominal por cluster y entrena una regresión log-lineal para el estimador de precios.

`fact_avaluos_cdmx_audit.py` es la versión espejo con 60 checkpoints `audit_log(...)` inyectados antes/después de cada operación importante. Misma lógica, salida a `eda_cdmx_audit.html`. Úsalo para verificar que los datos que ves en la plataforma son fidedignos antes de un deploy o tras cualquier cambio.

## Cómo correrlo (Windows / PowerShell)

```powershell
cd C:\Users\ricar\Documents\BlackPrint\Vivienda
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Producción → eda_cdmx.html (~31 MB)
& "C:\Users\ricar\anaconda3\python.exe" fact_avaluos_cdmx.py

# Auditoría → eda_cdmx_audit.html + audit.log con 60 checkpoints
& "C:\Users\ricar\anaconda3\python.exe" fact_avaluos_cdmx_audit.py | Out-File -Encoding utf8 audit.log

# DDL sugerido (no se ejecuta; solo imprime)
& "C:\Users\ricar\anaconda3\python.exe" fact_avaluos_cdmx.py --dump-ddl
```

Credenciales de DB están **hardcoded** en el dict `DB` (~línea 80). El docstring dice que respeta `BLACKPRINT_DB_PASSWORD` pero en realidad no — moverlo a env var es uno de los primeros TODO.

## Arquitectura del pipeline (orden de ejecución en `main()`)

1. **connect()** — sesión read-only a Postgres prod.
2. **load_data** → `SELECT FROM staging.stg_avaluos WHERE cve_ent='09' AND proposito='1'`. ~189k filas.
3. **winsorize_block** → recorta a `[p1, p99]` columnas monetarias y de superficie. Conservador, preserva mediana.
4. **load_pois / load_nse_muni / load_muni_geometries** → tres `SELECT` independientes a tablas dim.
5. **attach_pois** → `cKDTree` sobre POIs en XY (proyección equirectangular); conteo en buffer de 500 m por top-6 `main_category`. Añade columnas `poi_<categoria>`.
6. **attach_nse** → merge LEFT por `cve_mun` (zfill 3). Aporta `nse_score`, `nse_label`, conteos por tier.
7. **describe** → KPIs globales + por año + histogramas precomputados (alimenta el modal KPI con comparación de años).
8. **cluster_kmeans** → `StandardScaler` + `KMeans(k auto, random_state=42, n_init=10)`. `k = max(3, min(6, #clases con n≥200))`. Remap de IDs `0=más barato → k-1=más caro` para que el orden visual del reporte sea estable.
9. **timeseries_plusvalia** → mediana de `m2_sv` por `(cluster, año)`, referida al primer año con `n ≥ N_MIN_TIMESERIES (50)`.
10. **compute_cluster_data / compute_alcaldia_data / compute_absorcion_data** → precomputa histogramas y stats por cluster, por alcaldía y por desarrollador para que los modales del HTML sean instantáneos.
11. **train_price_regression** → `LinearRegression` sobre `log(valor_concluido) ~ lat + lon + sup + log(sup) + rec + ban + edad + sqrt(edad) + dummies_de_cluster`. Sin train/test split a propósito (queremos coeficientes estables, no diagnosticar overfit).
12. **prepare_map_points / prepare_phys_attr_points** → arrays compactos en formato `[lat, lon, cluster, año, ...]` para que el JS de Leaflet los lea por índice posicional.
13. **render_html** → escribe `eda_cdmx.html` (~31 MB con todo el payload + logos base64).

## Bugs conocidos / TODO

### Crítico

- **`Exportar Vista PNG` no funciona bien (botones en §02 y §05).**
  - Síntoma observado: la imagen exportada no muestra el buffer porque el mapa no está centrado en el punto antes de la exportación, y a veces el buffer/círculo aparece cortado o ausente.
  - Causa: `html2canvas` captura el viewport actual del `<div id="map-canvas">` o `#phys-map-canvas`. Si el usuario hizo pan o el buffer está fuera del centro, Leaflet muestra tiles diferentes a los que se quieren exportar.
  - Fix sugerido: justo antes de `html2canvas(node, ...)` hacer
    ```js
    map.setView([state.centerLat, state.centerLon], map.getZoom(), {animate: false});
    map.invalidateSize();
    map.once('moveend', () => html2canvas(node, opts).then(...));
    ```
    y asegurarse de que `bufCircle` esté en el centro del viewport.
  - Archivos: `MAP_JS_TEMPLATE` (~línea 2200, search `m-export-png`) y `PHYS_MAP_JS_TEMPLATE` (~línea 2600, search `phys-export-pdf`).
  - Bonus: dar al usuario un toggle entre exportar "vista actual" vs "buffer centrado" para flexibilidad.

### Importante

- **Credenciales DB hardcoded** en `DB` dict (~línea 80). Mover a `os.environ["BLACKPRINT_DB_PASSWORD"]`.
- **`PARTIAL_YEAR = 2025` hardcoded.** Reemplazar por `datetime.now().year` o lógica que detecte el último año con menos del 50% de los meses cubiertos.
- **Margen del Desarrollador (§05) — sesgo de Jensen.** La fórmula actual usa medianas de tres distribuciones distintas: `(med(valor_concluido) − med(terreno) − med(construcción)) / med(valor_concluido)`. La mediana del cociente NO es el cociente de las medianas. Para producción, calcula el margen vivienda-por-vivienda cuando los tres campos están en la misma fila, y luego toma la mediana de esos cocientes.
- **POI categorías** (`POI_TOP_N=6`) se eligen por volumen total — sesga hacia categorías ubicuas (e.g. "restaurant", "store"). Considera ponderar por relevancia urbana o usar tf-idf por buffer.
- **NSE solo a nivel alcaldía**. La tabla `dim_socioeconomic_level_ageb_locality` tiene granularidad AGEB. Un spatial join AGEB-vivienda da NSE por inmueble y debería mejorar el clustering y la regresión.
- **Sample estratificado en `prepare_map_points`** (max 250k pts): si en el futuro CDMX crece más allá, ajustar.

### Polish

- El payload `__CDMX` del modal KPI incluye histogramas completos en JSON inline. Si quieres reducir el tamaño del HTML, considera servir el JSON aparte (pero rompes el self-contained).
- `setupPredictor` corre en `DOMContentLoaded` porque el `<script>` de §05 va antes en el body que la sección §06. Si reordenas las secciones, ese hack se vuelve innecesario.
- Las funciones internas en MAP_JS reusan el nombre `fmtM2` con redefinición; ya está limpio en PHYS_MAP_JS (renombrado a `paFmt*`) pero MAP_JS no.

## Interpretabilidad pendiente (para el siguiente LLM)

El reporte actual es **exploratorio y descriptivo**. Lo que falta es acoplarle un marco económico que justifique los hallazgos y permita responder "por qué" se observan los patrones que el cliente ve. Estas son las referencias canónicas para hacerlo:

### 1. Modelos hedónicos (Rosen 1974)

El precio de la vivienda se descompone como suma de los **precios implícitos** de sus atributos (sup, recámaras, edad, ubicación, amenidades). La `train_price_regression` actual es un proto-modelo hedónico log-lineal; falta:

- Reportar **elasticidades** (`∂log(P)/∂log(X)` para continuas, semi-elasticidades para dummies).
- Mostrar **CIs por coeficiente** (bootstrap o errores Huber-White).
- Contraste con literatura: una elasticidad típica de `log(sup)` está en `[0.6, 0.9]`; baño y recámara aportan ~5–12% cada uno.

**Referencia base:** Rosen, S. (1974). *Hedonic Prices and Implicit Markets: Product Differentiation in Pure Competition*. JPE 82(1).

### 2. Modelo monocéntrico / bid-rent (Alonso–Mills–Muth)

El gradiente precio–distancia al centro de empleo es la primera dimensión a explicar para CDMX. Falta:

- Calcular **distancia al CBD** (e.g. Reforma–Centro, o Polanco como CBD secundario) y graficar `med(m2_sv) ~ distancia`.
- Estimar **elasticidad precio-distancia**: `log(P) ~ α + β·dist + controles`. Pendiente esperada: negativa y ~-0.05 a -0.15 por km.
- Discutir policentrismo (CDMX tiene Reforma, Polanco, Santa Fe, Insurgentes Sur).

**Referencias:**
- Alonso, W. (1964). *Location and Land Use*.
- Mills, E. (1967). *An Aggregative Model of Resource Allocation in a Metropolitan Area*. AER 57(2).
- Muth, R. (1969). *Cities and Housing*.
- Anas, A., Arnott, R., Small, K. (1998). *Urban Spatial Structure*. JEL 36(3) — para policentrismo.

### 3. Economía de aglomeración y amenidades urbanas

Los conteos de POIs son un proxy de amenidades de consumo. Para interpretarlos:

- Regresión auxiliar `log(P) ~ Σ poi_<categoría>` por cluster, reportando qué amenidades pagan prima en cada submercado.
- Diferenciar amenidades de **producción** (proximidad a empleo) vs **consumo** (restaurantes, parques, cultura).

**Referencias:**
- Glaeser, E., Kolko, J., Saiz, A. (2001). *Consumer City*. JEG 1(1).
- Couture, V., Handbury, J. (2020). *Urban Revival in America*. JUE 119.
- Combes, P-P., Duranton, G., Gobillon, L. — varios papers sobre wage premium urbano que aplican a precios de vivienda.

### 4. Oferta de vivienda y restricciones regulatorias

Para entender **por qué unas zonas son caras** (no solo demanda):

- Cruzar con uso de suelo de SEDUVI (PDDU) y altura máxima permitida.
- El §07 (absorción por desarrollador) ya muestra dónde están construyendo; combinar con regulación → mapa de oferta elástica vs inelástica.

**Referencias:**
- Saiz, A. (2010). *The Geographic Determinants of Housing Supply*. QJE 125(3).
- Glaeser, E., Gyourko, J. (2018). *The Economic Implications of Housing Supply*. JEP 32(1).
- Para México: Monkkonen, P. (2011, 2014). Trabajo sobre INFONAVIT y expansión periférica.

### 5. Índices de precios robustos

`timeseries_plusvalia` actual = `med(P_t) / med(P_0) - 1`. Es sensible a cambios en la composición de productos vendidos cada año. Mejores prácticas:

- **Repeat sales** (Bailey-Muth-Nourse 1963, Case-Shiller 1989): requiere mismo inmueble vendido varias veces — probablemente no aplica a avalúos hipotecarios.
- **Quality-adjusted hedonic index**: predice un "avalúo estándar" cada año con los coeficientes hedónicos del modelo del paso 1. **Esta es la opción más viable con los datos actuales.**
- Reportar en paralelo la plusvalía nominal-mediana (actual) y la quality-adjusted.

**Referencias:**
- Bailey, M., Muth, R., Nourse, H. (1963). *A Regression Method for Real Estate Price Index Construction*. JASA 58(304).
- Case, K., Shiller, R. (1989). *The Efficiency of the Market for Single-Family Homes*. AER 79(1).
- Diewert, W. E. (2003). *Hedonic regressions: A consumer theory approach*. NBER chapter.

### 6. Spatial econometrics

K-Means agrupa por features incluyendo lat/lon, pero ignora **dependencia espacial residual** (autocorrelación en los errores de la regresión hedónica).

- Test rápido: **Moran's I** sobre residuales de `train_price_regression`. Si `p < 0.01`, hay autocorrelación.
- Modelos: SAR (Spatial Autoregressive), SEM (Spatial Error Model), o GWR (Geographically Weighted Regression).
- En Python: `pysal`, `spreg`, `mgwr`.

**Referencias:**
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
- LeSage, J., Pace, R. K. (2009). *Introduction to Spatial Econometrics*.
- Fotheringham, A. S., Brunsdon, C., Charlton, M. (2002). *Geographically Weighted Regression*.

### 7. Submercados endógenos y validación de clusters

Los clusters actuales son K-Means sobre features pre-elegidas. Para defender que son submercados "reales":

- Goodman, A., Thibodeau, T. (1998). *Housing Market Segmentation*. JUE 7(2).
- Bourassa, S., Hoesli, M., Peng, V. (2003). *Do housing submarkets really matter?* JHE 12(1).
- Validación: comparar RMSE de un hedónico global vs hedónicos cluster-specific. Si los cluster-specific dominan, los submercados son informativos.

### 8. Contexto mexicano específico

- **Monkkonen, P.** — políticas de vivienda en México, INFONAVIT, suburbanización.
- **CIDOC + SHF** — anuarios con benchmarks de mercado (contrastar nuestras medianas contra estos).
- **CONEVAL + INEGI** — validación cruzada del NSE más allá del proxy a nivel alcaldía.
- **López-Morales, E.** y otros — gentrificación en CDMX y otras metrópolis latinoamericanas.
- **Inurreta-Aguirre y otros** — estudios mexicanos de pricing hedónico (buscar en Estudios Económicos, El Trimestre Económico).

## Upgrades de modelado (prioritarios)

Lo que hoy se llama "Estimador de precio" en §06 es una **regresión lineal en log-precio** (`sklearn.linear_model.LinearRegression`) — eso fue una elección deliberada para arrancar con coeficientes interpretables y con varianza controlada para el IC del 95%. Para producción es claramente insuficiente; las dos líneas de trabajo más importantes para el siguiente equipo son:

### 1. Migrar el predictor de precio: regresión lineal → XGBoost

- **Por qué**: el mercado inmobiliario es altamente no lineal (interacciones entre superficie, ubicación, edad, submercado), tiene heterogeneidad espacial y rendimientos marginales decrecientes que la regresión log-lineal solo captura parcialmente vía `log(sup)` y `sqrt(edad)`. Un gradient-boosted tree captura estas interacciones automáticamente y suele dominar en pricing de inmuebles (RMSE típicamente 30–50% mejor que OLS hedónico).
- **Setup sugerido**:
  - `xgboost.XGBRegressor` con `objective='reg:squarederror'` sobre `log(valor_concluido)` (mantener log para estabilidad y aditividad de errores).
  - **Train/test/val split sí o sí** (la regresión actual no lo necesitaba; XGBoost se sobreajusta fácil).
  - **CV temporal**: corta por año (entrenar 2019–2023, validar 2024, test 2025) — los splits aleatorios filtran información futura y sobreestiman accuracy.
  - **IC 95%** via cuantiles: tres modelos paralelos con `objective='reg:quantileerror'` sobre `α ∈ {0.025, 0.5, 0.975}`, o un solo modelo + bootstrap.
  - **SHAP values** para mantener interpretabilidad. El UI del estimador actual muestra "submercado seleccionado + comparables"; con SHAP puedes añadir una mini-explicación tipo "este precio está +X% por encima del baseline del submercado por: sup (+12%), ubicación (+8%), edad (–3%)".
- **Mantener el predictor actual como baseline** durante la migración — la regresión es defendible como modelo "transparente" y vale para reportar coeficientes hedónicos en un anexo.
- **Referencias**: Chen & Guestrin (2016) *XGBoost*. Para pricing inmobiliario con boosting: Mayer et al. (2022) *Estimation and updating methods for hedonic valuation*; Pace & Hayunga (2020) sobre interacciones espaciales con ML.

### 2. Diseño cuidadoso de variables urbanas (site score)

Las features actuales del clustering y de la regresión son una mezcla razonable pero **no se han diseñado deliberadamente**: son los atributos que vinieron del avalúo más conteos crudos de POIs por top-N categoría. Para producción se necesita una capa de **site scoring** explícita.

- **Versión mínima ("v0")**: empezar con una **parametrización general que capture densidades** alrededor del punto:
  - **Densidad de POIs** total y por familia (consumo, servicios, salud, educación, transporte) — ya tenemos los datos, falta agruparlos en categorías económicas en lugar de las `main_category` crudas de Dataplor.
  - **Densidad de vivienda** (n avalúos vecinos por km², con buffer escalable de 200 m a 2 km).
  - **Densidad de empleo** vía DENUE (INEGI) si se puede traer a `dim_*`.
  - **Accesibilidad a transporte** (distancia mínima a Metro / Metrobús / Tren Suburbano / cablebús).
  - **Distancia a CBD/sub-CBDs** (Reforma–Centro, Polanco, Santa Fe, Insurgentes Sur).
  - **Elevación / pendiente** si está disponible (afecta valor en zonas como Naucalpan, Álvaro Obregón sur).
- Cada componente se normaliza y se combina en un **site score** (e.g. PCA sobre las componentes, o ponderaciones derivadas de la regresión hedónica). Se documenta la transformación para que el cliente la entienda y la puedas auditar.
- **Versión "v1"** (después de v0): seleccionar variables adicionales **cuidadosamente** revisando:
  - La DB completa (`DB_ARCHITECTURE.txt`) — hay tablas con uso de suelo, infraestructura, indicadores socioeconómicos a nivel AGEB que hoy no se usan.
  - La literatura (sección "Interpretabilidad pendiente" abajo) — qué variables están demostradas como significativas en mercados urbanos comparables.
- Mantén el design doc del site score versionado: cada feature debe tener (a) fuente, (b) transformación, (c) razón teórica para incluirla, (d) signo esperado en la regresión.

### 3. Rediseño del clustering

K-Means con `k ∈ [3, 6]` heurística sobre features estandarizadas es un primer corte; **no es un análisis de submercados terminado**. Lo que falta:

- **Exploración previa**: PCA / UMAP / t-SNE sobre el espacio de features ampliado (con site score) para visualizar si hay estructura natural y cuántos grupos.
- **Selección de k formal**: elbow + silhouette + gap statistic (Tibshirani et al. 2001) en lugar de la heurística por #clases del valuador. Probablemente k=5 o k=7 después de meter site score.
- **Algoritmos alternativos**: Gaussian Mixture (clusters con covarianzas no esféricas, más realistas para mezcla de tipologías), HDBSCAN (clusters de densidad variable, captura barrios atípicos), k-medoids con distancia robusta.
- **Validación**: tests de submercados estadísticamente distintos (Goodman-Thibodeau 1998, Bourassa et al. 2003) — comparar RMSE de un hedónico global vs hedónicos cluster-specific. Si los clusters no mejoran la predicción, no son submercados informativos.
- **Estabilidad**: bootstrap los clusters y reporta cuántas viviendas cambian de cluster a través de muestras. Si la asignación es inestable (>20% de churn), el número de clusters o las features están mal.
- **Interpretación**: hoy los nombres salen de medianas + heurísticas; con SHAP sobre un clasificador de "cluster id" puedes mostrar qué atributo es el más discriminante por cluster y construir nombres respaldados estadísticamente.

Estos tres bloques (XGBoost + site score + rediseño de clusters) deberían correr en **paralelo y en este orden de dependencia**: site score se diseña primero, después se rehace clustering con las features mejoradas, y XGBoost finalmente se entrena con el mejor set de features y submercados resultante.

## Decisiones de diseño que el siguiente equipo debe respetar (o revisitar conscientemente)

| Decisión | Justificación actual | Cuándo revisitarla |
|---|---|---|
| **Read-only**, sin escritura a DB. | Seguridad operacional. `--dump-ddl` imprime el DDL pero no lo ejecuta. | Solo si product-owners aprueban materializar la vista. |
| **Filtro `proposito='1'`** (adquisición). | 99.7% de los datos. Excluye refinanciamientos. | Si cambias el filtro, cambia el storytelling completo del reporte. |
| **Año parcial 2025** marcado con `*`. | Captura aún en curso. | Hacer dinámico (`datetime.now().year`). |
| **`random_state=42` en KMeans.** | Reproducibilidad cluster-a-cluster entre runs. | Cambiar el seed mueve los IDs y rompe comparaciones con reportes históricos. |
| **k entre [3, 6]** auto-elegida por #clases con n≥200. | Heurística defendible pero arbitraria. | Validar con elbow / silhouette si quieres formalizar. |
| **Sin train/test split** en la regresión. | Objetivo: coeficientes estables, no predicción out-of-sample. | Si reportas accuracy del predictor en la UI, sí necesitas split o cross-val. |
| **Etiquetas de cluster sin NSE ni alcaldía.** | Pedido del producto (cae siempre en C/C-/C+, no separa). La info sigue en `cluster_data`. | Si cambias las features de clustering y NSE empieza a separar, vuelve a meterlo. |
| **Español MX (tuteo).** | Cliente mexicano. | No hay razón para voseo argentino ni España. |
| **Paleta** gris BlackPrint + orange `#ed6b1f` + rose tenue `#d49ba2` + negro hero/footer. | Branding. | Pedir aprobación de diseño antes de cambiar. |
| **Logos base64 inline.** | HTML self-contained, fácil de compartir. | Si pesa demasiado, servir desde CDN. |
| **Margen del Desarrollador** = `(vivienda − terreno − construcción) / vivienda` con medianas intra-buffer. | El pedido fue restar el terreno; antes solo restaba construcción. | Ver bug "sesgo de Jensen" arriba — la fórmula es defensible como heurística pero formalmente sesgada. |

## Archivos del proyecto

```
Vivienda/
├── fact_avaluos_cdmx.py            # pipeline de producción
├── fact_avaluos_cdmx_audit.py      # mirror con 60 audit_log(); mismo comportamiento funcional
├── eda_cdmx.html                   # salida de producción (regenerar tras cada cambio)
├── eda_cdmx_audit.html             # salida de auditoría
├── HANDOFF.md                      # este documento
├── LogoSimple_Dark.png             # BlackPrint wordmark dark (embed base64)
├── LogoSimple_Light.png            # BlackPrint wordmark white (embed base64)
├── Logo_Orange.png                 # orange mark (embed base64)
├── DB_ARCHITECTURE.txt             # notas de la DB
└── EDA_avaluos.py                  # exploración inicial (legacy, no se ejecuta en prod)
```

## Próximos pasos sugeridos (en orden de impacto/esfuerzo)

**Quick wins (horas a 1 día):**

1. **Fix export PNG** (1–2 h) — centrar mapa antes de `html2canvas`.
2. **Mover credenciales a env vars** (30 min).
3. **`PARTIAL_YEAR` dinámico** (15 min).
4. **Distance-to-CBD** + regresión bid-rent rápida (1 día) — primer paso de interpretabilidad económica.

**Track de modelado (ver sección "Upgrades de modelado" arriba):**

5. **Diseñar site score v0** — densidades + accesibilidad + distancia a CBDs (1 semana).
6. **NSE a nivel AGEB** vía spatial join (3–5 días) — feature crítica para site score y clustering.
7. **Rediseño de clustering** — PCA/UMAP exploratorio + selección formal de k + GMM/HDBSCAN comparativo + validación de submercados (1–2 semanas, tras tener site score).
8. **Migrar predictor a XGBoost** con CV temporal + cuantiles para IC + SHAP para explicabilidad (1 semana, tras tener clusters y site score nuevos).

**Track de interpretabilidad económica:**

9. **Quality-adjusted hedonic index** reemplazando o complementando plusvalía actual (1 semana).
10. **Moran's I + SAR/SEM** sobre residuales de la regresión (1 semana con `pysal`).
11. **Validación con literatura mexicana** — Monkkonen, INEGI, CIDOC — contraste de medianas y elasticidades.
12. **Hedónico cluster-specific** + comparación RMSE vs global — valida que los submercados son informativos (cae como subproducto del paso 7 si se hace bien).

---

*Si llegas hasta aquí: el script de auditoría es tu amigo — antes de cambiar cualquier operación, corre `fact_avaluos_cdmx_audit.py`, captura el log, haz tu cambio, vuelve a correr el audit y diffea los logs. Si los descriptivos antes/después de las operaciones clave (winsorize, K-Means, regresión) cambian más allá de tolerancia numérica, algo no es fidedigno.*
