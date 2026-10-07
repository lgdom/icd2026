# Tarea 3: Regresión Lineal y Clasificación
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción de la tarea

Esta tarea aborda el modelado y evaluación de modelos lineales aplicados a dos problemas distintos:
1. **Regresión:** Predecir la calificación continua de calidad (`quality`) de un vino tinto a partir de sus propiedades fisicoquímicas, evaluando el desempeño mediante la Raíz del Error Cuadrático Medio (RMSE) y el Coeficiente de Determinación ($R^2$) frente a una línea base trivial (el promedio del conjunto de entrenamiento).
2. **Clasificación mediante Regresión Lineal:** Transformar el problema en una clasificación binaria («bueno» $\ge 7$ vs «no bueno» $< 7$) forzando un modelo de regresión lineal múltiple con umbral de decisión en $0.5$, analizando sus limitaciones, el impacto del desbalance de clases frente a la línea base («siempre no bueno») y la ocurrencia de predicciones fuera del rango válido de probabilidad $[0, 1]$.

---

## El Conjunto de Datos

- **Fuente:** [UCI Machine Learning Repository — Wine Quality Dataset (Red Wine)](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Instancias:** 1,599 muestras de vino tinto portugués *Vinho Verde*.
- **Atributos:** 11 variables fisicoquímicas numéricas continuas y 1 variable objetivo discreta (`quality`, evaluada en escala entera de 3 a 8).
- **Valores faltantes:** 0 valores nulos en todo el conjunto de datos.
- **Partición:** 80 % entrenamiento ($n = 1,279$) y 20 % prueba ($n = 320$) con semilla fija `random_state=42`.

### Diccionario de variables

| Variable | Tipo | Rango | Descripción |
|---|---|:---:|---|
| `fixed acidity` | Numérico | 4.6 – 15.9 | Acidez fija (ácido tartárico en $\text{g/dm}^3$) |
| `volatile acidity` | Numérico | 0.12 – 1.58 | Acidez volátil (ácido acético en $\text{g/dm}^3$) |
| `citric acid` | Numérico | 0.0 – 1.0 | Ácido cítrico en $\text{g/dm}^3$ |
| `residual sugar` | Numérico | 0.9 – 15.5 | Azúcar residual en $\text{g/dm}^3$ |
| `chlorides` | Numérico | 0.012 – 0.611 | Cloruros (cloruro de sodio en $\text{g/dm}^3$) |
| `free sulfur dioxide` | Numérico | 1.0 – 72.0 | Dióxido de azufre libre en $\text{mg/dm}^3$ |
| `total sulfur dioxide` | Numérico | 6.0 – 289.0 | Dióxido de azufre total en $\text{mg/dm}^3$ |
| `density` | Numérico | 0.990 – 1.004 | Densidad en $\text{g/cm}^3$ |
| `pH` | Numérico | 2.74 – 4.01 | Nivel de acidez o basicidad en escala pH |
| `sulphates` | Numérico | 0.33 – 2.0 | Sulfatos (sulfato de potasio en $\text{g/dm}^3$) |
| `alcohol` | Numérico | 8.4 – 14.9 | Porcentaje de alcohol por volumen ($\%$ vol) |
| `quality` | Numérico (entero) | 3 – 8 | Calificación sensorial de calidad (mediana de evaluaciones) |

---

## Estructura del Directorio

```text
homework/regresion/
├── data/
│   ├── winequality-red.csv    # Conjunto de datos original delimitado por punto y coma
│   └── winequality.names      # Descripción de las variables
├── notebooks/
│   └── regresion.ipynb        # Cuaderno Jupyter con el análisis, modelos y visualizaciones
├── requirements.txt           # Dependencias requeridas del proyecto
└── README.md                  # Documentación del proyecto
```

---

## Resultados

### 1. Regresión (Evaluación en el 20 % de Prueba)

| Modelo | RMSE | $R^2$ | Interpretación |
|---|:---:|:---:|---|
| **Línea base (promedio)** | $0.811$ | $-0.006$ | Error de referencia al predecir una constante sin información predictora. |
| **Regresión simple (`alcohol`)** | $0.707$ | $0.236$ | Explica el $23.6\,\%$ de la varianza considerando únicamente la variable más correlacionada. |
| **Regresión múltiple (11 variables)** | **$0.625$** | **$0.403$** | Reduce el error y explica el $40.3\,\%$ de la varianza al incorporar todo el perfil fisicoquímico. |

### 2. Clasificación con Regresión Lineal ($\text{bueno} \ge 7$, umbral $0.5$)

| Enfoque | Porcentaje de Aciertos | Predicciones fuera de $[0, 1]$ |
|---|:---:|:---:|
| **Línea base (predecir siempre «no bueno»)** | $85.31\,\%$ | $0$ ($0.00\,\%$) |
| **Regresión lineal múltiple (corte en $0.5$)** | **$86.25\,\%$** | **$82$ de $320$ ($25.62\,\%$)** |

---

## Conclusiones

1. En regresión se modela una magnitud numérica continua minimizando el error (RMSE). En clasificación el objetivo es construir fronteras de decisión que delimiten la pertenencia a clases.

2. La limitación de la recta como clasificador es que produce valores fuera de rango ($25.62\,\%$ de las muestras de prueba), careciendo de interpretación como probabilidades reales.
3. El efecto del desbalance de clases es que el clasificador lineal aparenta un acierto del $86.25\,\%$, pero este resultado casi no supera a la línea base ($85.31\,\%$).

---

## Requisitos y Entorno de Ejecución

Para reproducir el cuaderno se requiere Python 3.10+ y las bibliotecas especificadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

Para abrir y ejecutar el cuaderno:

```bash
jupyter notebook notebooks/regresion.ipynb
```
