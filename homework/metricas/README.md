# Tarea 4: Métricas de Clasificación
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción de la tarea

Esta tarea aborda la evaluación de modelos de clasificación en presencia de clases desbalanceadas, utilizando el conjunto de datos de telemarketing bancario *Bank Marketing* (Moro et al., 2014). Se obtienen las métricas de clasificación y se comparan con una línea base ("Siempre No"), se evalúan diferentes umbrales de decisión y se analiza el impacto de la variable `duration`. Se abordan las siguientes operaciones:

1. **Fuga de información (*Data Leakage*):** Identificación y eliminación de la variable `duration`, la cual es desconocida en el momento de decidir a quién contactar y cuya inclusión sesgaría de manera irrealista el rendimiento del modelo.
2. **Evaluación de línea base trivial:** Comparación del rendimiento frente a la linea base ("Siempre No"), demostrando por qué la exactitud (*accuracy*) es una métrica engañosa e inútil cuando la clase minoritaria representa solo el $11.27\,\%$.
3. **Modelado con Regresión Logística en Pipeline:** Construcción de un Pipeline de scikit-learn que incluye: `ColumnTransformer` (`StandardScaler` para 10 variables numéricas y `OneHotEncoder` para 10 categóricas) y un clasificador de regresión logística.
4. **Métricas de evaluación completas (umbral $0.5$):** Cálculo analítico y mediante scikit-learn de la Exactitud, Precisión, Sensibilidad, Especificidad y $F_1$-score, derivando la matriz de confusión.
5. **Curvas ROC vs. Precision-Recall:** Visualización de las curvas ROC ($\text{AUC} = 0.8007$) y Precision-Recall ($\text{AP} = 0.4657$), exponiendo cómo la curva ROC disfraza el impacto de las falsas alarmas debido a un denominador masivo de verdaderos negativos, mientras que la curva PR revela el compromiso operativo real.
6. **Umbral de decisión óptimo y función de costo-beneficio:** Deducción del umbral óptimo $p^* = 0.10$ a partir de los costos operativos ($\$5$ por llamada) y el beneficio neto ($\$50$ por cliente que contrata), evaluando el incremento de ganancia neta en el conjunto de prueba ($\$8,900$ a $\$22,060$, un aumento de $+147.9\,\%$).
7. **Impacto de la variable omitida (`duration`):** Reentrenamiento del modelo incorporando `duration` para contrastar las métricas y la viabilidad de operar dicho modelo en la toma de decisiones.

---

## El Conjunto de Datos

- **Fuente:** [UCI Machine Learning Repository — Bank Marketing (Moro et al., 2014)](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Instancias:** 41,188 registros de llamadas telefónicas de campañas de venta directa realizadas por una institución bancaria portuguesa entre 2008 y 2010.
- **Atributos:** 20 variables predictoras operativas (10 numéricas continuas/discretas y 10 categóricas tras remover `duration`) y 1 variable objetivo binaria (`y`: `"yes"` si contrató un depósito a plazo fijo, `"no"` en caso contrario).
- **Balance de clases:** 36,548 negativos ($88.73\,\%$) frente a 4,640 positivos ($11.27\,\%$), es decir, solo 1 de cada 9 clientes contactados aceptó la oferta.
- **Partición:** 80 % entrenamiento ($n = 32,950$) y 20 % prueba ($n = 8,238$) con muestreo estratificado y semilla fija `random_state=42`.

### Diccionario de variables clave

| Variable | Tipo | Descripción |
|---|---|---|
| `age` | Numérico | Edad del cliente en años |
| `job` | Categórico | Ocupación laboral (12 categorías: admin., blue-collar, technician, etc.) |
| `marital` | Categórico | Estado civil (married, single, divorced, unknown) |
| `education` | Categórico | Nivel de escolaridad (basic.4y, high.school, university.degree, etc.) |
| `default` | Categórico | ¿Tiene crédito en incumplimiento de pago? (no, yes, unknown) |
| `housing` | Categórico | ¿Tiene préstamo para vivienda? (no, yes, unknown) |
| `loan` | Categórico | ¿Tiene préstamo personal? (no, yes, unknown) |
| `contact` | Categórico | Tipo de canal de comunicación (cellular, telephone) |
| `month`, `day_of_week` | Categórico | Último mes y día de la semana en que fue contactado |
| `campaign` | Numérico | Número de contactos realizados durante esta campaña para este cliente |
| `pdays` | Numérico | Días transcurridos desde la última campaña previa (999 = no contactado antes) |
| `previous` | Numérico | Número de contactos realizados antes de esta campaña |
| `poutcome` | Categórico | Resultado de la campaña de marketing previa (nonexistent, failure, success) |
| `emp.var.rate` | Numérico | Tasa de variación del empleo (indicador trimestral) |
| `cons.price.idx` | Numérico | Índice mensual de precios al consumidor |
| `cons.conf.idx` | Numérico | Índice mensual de confianza del consumidor |
| `euribor3m` | Numérico | Tasa Euríbor a 3 meses (indicador diario del mercado interbancario) |
| `nr.employed` | Numérico | Número promedio de empleados en el trimestre (en miles) |
| `duration` | Numérico | Duración del último contacto en segundos (**eliminada por fuga de información**) |
| `y` | Categórico (objetivo) | ¿El cliente contrató el depósito a plazo? (`yes` = 1, `no` = 0) |

---

## Estructura del Directorio

```text
homework/metricas/
├── data/
│   ├── bank-additional-full.csv   # Conjunto de datos completo delimitado por punto y coma (41,188 filas)
│   └── bank-additional-names.txt  # Descripción oficial de variables y metadatos UCI
├── notebooks/
│   └── metricas.ipynb             # Cuaderno Jupyter con el flujo completo, modelos y gráficos
├── requirements.txt               # Dependencias requeridas del proyecto
└── README.md                      # Documentación del proyecto
```

---

## Conclusiones y Recomendaciones

1. **Métrica a priorizar:** En conjuntos de datos desbalanceados, la exactitud no debe emplearse como criterio rector. La institución bancaria debe priorizar la **sensibilidad (*recall*)** monitoreada a través de la curva Precision-Recall, pues dejar escapar clientes interesados genera un costo de oportunidad considerablemente superior al costo operativo de una llamada.
2. **Umbral de decisión óptimo:** Se recomienda adoptar de forma inmediata el **umbral económico $p^* = 0.10$**. Operar bajo el umbral simétrico de $0.50$ destruía más de la mitad de la rentabilidad del negocio; ajustarlo al trade-off real de costos y beneficios más que duplica las ganancias netas de la campaña.

---

## Requisitos y Entorno de Ejecución

Para reproducir el cuaderno se requiere Python 3.10+ y las bibliotecas especificadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

Para abrir y ejecutar el cuaderno:

```bash
jupyter notebook notebooks/metricas.ipynb
```
