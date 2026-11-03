# Práctica 3: Clasificación de Deserción y Éxito Académico
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción del Proyecto

Este repositorio contiene el desarrollo de la **Práctica 3**, enfocada en construir y comparar distintos clasificadores supervisados para predecir, al cierre del **primer semestre**, la situación final de un estudiante de educación superior (`Dropout`, `Enrolled` o `Graduate`).

- **Conjunto de datos base:** [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success (ID: 697)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), recopilado por el Instituto Politécnico de Portalegre (Portugal).
- **Conjunto de entrenamiento (`desercion-entrenamiento.csv`):** 3,539 estudiantes y 38 columnas (`id`, 36 atributos y `Target`).
- **Conjunto de prueba (`desercion-prueba.csv`):** 885 estudiantes y 37 columnas (`id` y 36 atributos, sin etiqueta `Target`).
- **Restricción temporal (escenario principal):** Exclusión de las 6 variables correspondientes al segundo semestre para predecir el riesgo de abandono al cierre del primer semestre (30 atributos originales + indicadores derivados del primer semestre).
- **Métrica principal de evaluación:** `F1 macro` bajo validación cruzada estratificada de 10 pliegues (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`).

---

## Resumen de Clasificadores Evaluados (Cierre del 1er Semestre)

Se diseñaron flujos de preprocesamiento específicos (`Pipeline` + `ColumnTransformer`) y se evaluaron al menos 50 configuraciones de hiperparámetros por familia mediante `GridSearchCV` sobre las mismas particiones de validación cruzada:

| Posición | Clasificador | Configuraciones evaluadas | F1 macro (media ± DE) | Exactitud | Sensibilidad `Dropout` |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1°** | **Regresión Logística** *(modelo seleccionado)* | 54 | **0.6938 ± 0.0281** | **74.12 %** | 70.27 % |
| **2°** | **Máquinas de Soporte Vectorial (SVM)** | 56 | 0.6900 ± 0.0235 | 75.59 % | **72.30 %** |
| **3°** | **Naive Bayes (`CategoricalNB`)** | 58 | 0.6583 ± 0.0184 | 71.18 % | 70.18 % |
| **4°** | **Árbol de Decisión** | 54 | 0.6550 ± 0.0246 | 72.17 % | 67.37 % |
| **5°** | **k Vecinos más Cercanos (kNN)** | 60 | 0.6514 ± 0.0261 | 73.13 % | 68.51 % |

Adicionalmente, se analizó el impacto de esperar al cierre del segundo semestre (donde el `F1 macro` del mejor clasificador sube a `0.7197 ± 0.0209`) y se generaron las predicciones finales para los 885 estudiantes del conjunto de prueba.

---

## Estructura de la Práctica

```text
practice 3/
├── data/
│   ├── desercion-entrenamiento.csv   # Conjunto de entrenamiento (3,539 estudiantes)
│   ├── desercion-prueba.csv          # Conjunto de prueba sin etiquetas (885 estudiantes)
│   └── predicciones-prueba.csv       # Predicciones generadas para el conjunto de prueba
├── notebooks/
│   └── practice-03.ipynb             # Cuaderno principal
├── predicciones-prueba.csv           # Copia en raíz del archivo de predicciones de entrega
├── requirements.txt                  # Dependencias del entorno
└── README.md                         # Documentación e instrucciones de ejecución
```

---

## Requisitos del Entorno

El proyecto requiere **Python 3.10** o superior (probado en Python 3.13) con las siguientes librerías:

- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.2.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `jupyter >= 1.0.0`
- `ipykernel >= 6.0.0`

---

## Instrucciones de Instalación y Ejecución

El cuaderno utiliza rutas relativas respecto a su ubicación dentro de `notebooks/`, por lo que es reproducible al clonar el repositorio e instalar las dependencias.

### 1. Clonar el repositorio y ubicarse en la carpeta de la práctica
```bash
git clone <URL_DEL_REPOSITORIO>
cd "icd2026/practices/practice 3"
```

### 2. Crear y activar el entorno virtual
En macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Abrir y ejecutar el notebook
```bash
jupyter notebook notebooks/practice-03.ipynb
```
O bien abrir directamente el archivo `notebooks/practice-03.ipynb` en Visual Studio Code o el editor de su preferencia con soporte para Jupyter Notebooks y ejecutar todas las celdas en orden (**Restart & Run All**). Al finalizar la ejecución se actualiza automáticamente el archivo `predicciones-prueba.csv`.

