# Práctica 3: Clasificación de Deserción y Éxito Académico
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción del Proyecto

Este repositorio contiene el desarrollo de la **Práctica 3**, enfocada en construir y comparar distintos clasificadores supervisados para predecir, al cierre del **primer semestre**, la situación final de un estudiante de educación superior (`Dropout`, `Enrolled` o `Graduate`).

- **Métrica principal de evaluación:** `F1 macro` bajo validación cruzada estratificada de 10 pliegues (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`).
- **Restricción temporal (escenario principal):** Exclusión de las 6 variables correspondientes al segundo semestre para predecir el riesgo de abandono al cierre del primer semestre (30 atributos originales + indicadores derivados del primer semestre).

---

## Descripción del Conjunto de Datos

El conjunto de datos proviene del repositorio oficial **UCI Machine Learning Repository**: [*Predict Students' Dropout and Academic Success (ID: 697)*](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), recopilado por el **Instituto Politécnico de Portalegre** (Portugal) a partir de registros administrativos y académicos de estudiantes inscritos en diversas licenciaturas entre los ciclos escolares 2008/2009 y 2018/2019 (Realinho et al., 2022).

- **Total de registros originales:** 4,424 estudiantes, sin valores faltantes (`0` nulos), divididos para esta práctica en:
  - **`desercion-entrenamiento.csv` (80 %):** 3,539 estudiantes y 38 columnas (`id`, 36 atributos predictivos y la variable objetivo `Target`).
  - **`desercion-prueba.csv` (20 %):** 885 estudiantes y 37 columnas (`id` y 36 atributos predictivos, sin la columna `Target`).
- **Variable objetivo (`Target`):** Categórica de 3 clases con desbalance moderado en entrenamiento:
  - `Graduate` (graduado): 1,781 estudiantes (**50.32 %**)
  - `Dropout` (desertor): 1,137 estudiantes (**32.13 %**)
  - `Enrolled` (aún inscrito): 621 estudiantes (**17.55 %**)
- **Atributos predictivos (36 variables en total: 30 disponibles al cierre del 1er semestre + 6 del 2do semestre):**
  - **9 Categóricos nominales** (almacenados como códigos enteros de catálogo): estado civil, modo de solicitud, carrera (`Course`), estudios previos, nacionalidad y códigos de escolaridad y ocupación de madre y padre.
  - **8 Categóricos binarios (`0` / `1`):** turno diurno/vespertino, desplazado, necesidades educativas especiales, deudor (`Debtor`), colegiatura al corriente (`Tuition fees up to date`), género, becario (`Scholarship holder`) y estudiante internacional.
  - **1 Categórico ordinal:** orden de preferencia en la solicitud (`Application order`).
  - **6 Numéricos discretos (al 1er semestre):** edad al inscribirse (`Age at enrollment`) y las 5 variables de conteo de unidades curriculares del 1er semestre (acreditadas, inscritas, evaluadas, aprobadas y sin evaluación).
  - **6 Numéricos continuos (al 1er semestre):** calificación de estudios previos, nota de admisión, promedio ponderado del 1er semestre (`Curricular units 1st sem (grade)`) e indicadores macroeconómicos nacionales al ingreso (`Unemployment rate`, `Inflation rate`, `GDP`).
  - **6 Variables del 2do semestre** *(excluidas del escenario principal y evaluadas en la Sección 7)*: los 5 conteos de unidades curriculares y el promedio ponderado del segundo semestre.

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

