# Introducción a la Ciencia de Datos 2026

**Posgrado en Ciencias de la Computación**  
*Centro de Investigación Científica y de Educación Superior de Ensenada (CICESE)*  

- **Estudiante:** Luis Felipe García Domínguez  
- **Profesor:** Dr. Irvin Hussein López Nava  

---

## Descripción del Repositorio

Este repositorio contiene el conjunto de materiales, prácticas de laboratorio, notas de lectura y el proyecto final desarrollados durante el curso de **Introducción a la Ciencia de Datos (2026)** en el Posgrado en Ciencias de la Computación del CICESE.

---

## Estructura del Repositorio

```text
icd2026/
├── book/                  # Revisión del libro: Python Data Science Handbook (Jake VanderPlas)
│   ├── 00_Preface/        # Prefacio
│   ├── 01_IPython/        # IPython: más allá de Python estándar
│   ├── 02_NumPy/          # Introducción y cálculo con arreglos en NumPy
│   ├── 03_Pandas/         # Manipulación de datos con Pandas
│   ├── 04_Matplotlib/     # Visualización con Matplotlib y Seaborn
│   ├── 05_Machine_Learning/ # Fundamentos de Machine Learning con Scikit-Learn
│   ├── data/              # Conjuntos de datos utilizados en los cuadernos del libro
│   └── figures/           # Figuras y diagramas de apoyo del libro
├── classes/               # Cuadernos y materiales vistos en clase
├── homework/              # Tareas del curso
│   ├── cnn/               # Tarea 5: Comparación de CNN con Global Average Pooling (MNIST)
│   ├── metricas/          # Tarea 4: Métricas de Clasificación y Costo Económico (Bank Marketing)
│   ├── preprocesamiento/  # Tarea 2: Preprocesamiento de Datos (Breast Cancer Wisconsin)
│   └── regresion/         # Tarea 3: Regresión Lineal y Clasificación (Wine Quality)
├── practices/             # Prácticas y análisis de datos
│   ├── practice 1/        # Práctica 1: Análisis Exploratorio de Datos (ATUS 2025 - INEGI)
│   ├── practice 2/        # Práctica 2: Preprocesamiento de Datos (ATUS 2025 - INEGI)
│   ├── practice 3/        # Práctica 3: Clasificación de Deserción y Éxito Académico (UCI 697)
│   └── practice 4/        # Práctica 4: Meta aprendizaje (UCI 697 y Bank Marketing)
├── project/               # Proyecto final de la materia
└── README.md              # Índice general y documentación del repositorio
```

---

## Tareas

- **[Tarea 2: Preprocesamiento de Datos](homework/preprocesamiento/)**
  Auditoría, limpieza, imputación de valores faltantes y reducción de dimensionalidad mediante Análisis de Componentes Principales (PCA) sobre el conjunto de datos de cáncer de mama de Wisconsin (*Breast Cancer Wisconsin*).

- **[Tarea 3: Regresión Lineal y Clasificación](homework/regresion/)**  
  Modelado continuo mediante regresión simple y múltiple sobre calidad de vino tinto (*Wine Quality*), y evaluación de las limitaciones de usar regresión lineal para clasificación binaria con clases desbalanceadas.

- **[Tarea 4: Métricas de Clasificación](homework/metricas/)**  
  Evaluación de modelos de clasificación con clases fuertemente desbalanceadas (*Bank Marketing*), prevención de fuga de información (*data leakage*), análisis comparativo de curvas ROC vs. Precision-Recall y optimización de umbrales bajo una función de costo-beneficio económico.

- **[Tarea 5: CNN compacta con GAP](homework/cnn/)**  
  Diseño y comparación de arquitecturas CNN con Global Average Pooling (GAP) en MNIST para superar 95 % en validación con menos parámetros que Flatten, y evaluación de robustez ante desplazamientos de 0, 2 y 4 píxeles.

---

## Prácticas

- **[Práctica 1: Análisis Exploratorio de Datos (EDA)](practices/practice%201/)**  
  Exploración estadística y visualización de patrones sobre la base de datos de Accidentes de Tránsito Terrestre en Zonas Urbanas y Suburbanas (ATUS 2025) del INEGI.

- **[Práctica 2: Preprocesamiento de Datos](practices/practice%202/)**  
  Implementación de técnicas de limpieza e imputación, aumento de datos mediante SMOTE-NC para clases desbalanceadas y extracción de características temporales y de dominio vial.

- **[Práctica 3: Clasificación de Deserción y Éxito Académico](practices/practice%203/)**  
  Diseño de pipelines de preprocesamiento, ajuste de hiperparámetros con validación cruzada estratificada y comparación de cinco familias de clasificadores (Árbol de Decisión, Naive Bayes, kNN, SVM y Regresión Logística) para predecir la deserción estudiantil al cierre del primer semestre (*Predict Students' Dropout and Academic Success*, UCI 697).

- **[Práctica 4: Meta aprendizaje](practices/practice%204/)**  
  Evaluación de familias de ensamble (Bagging, Boosting y Combinación heterogénea) frente a modelos base en deserción estudiantil (UCI 697), y búsqueda del mejor clasificador para captación bancaria (*Bank Marketing*, 41,188 instancias) con optimización de umbrales de decisión (`TunedThresholdClassifierCV`).

---

## Revisión del Libro

Cuadernos de estudio y ejercicios basados en **[Python Data Science Handbook](book/)** (*Jake VanderPlas*), organizados por capítulo:

- **[00 · Prefacio](book/00_Preface/)**: Introducción y alcance del libro.
- **[01 · IPython](book/01_IPython/)**: Ayuda, atajos, comandos mágicos, integración con shell, depuración y perfilado de tiempo.
- **[02 · NumPy](book/02_NumPy/)**: Tipos de datos, arreglos, funciones universales (*ufuncs*), agregaciones, *broadcasting*, máscaras booleanas, indexación avanzada y ordenamiento.
- **[03 · Pandas](book/03_Pandas/)**: `Series` y `DataFrame`, indexación y selección, valores faltantes, índices jerárquicos, concatenación, uniones (*merge/join*), agrupación, tablas pivote, manejo de cadenas y series de tiempo.
- **[04 · Matplotlib](book/04_Matplotlib/)**: Gráficos de líneas y dispersión, barras de error, contornos, histogramas, personalización de leyendas/ejes/estilos, subgráficas, gráficos 3D, mapas y visualización con Seaborn.
- **[05 · Machine Learning](book/05_Machine_Learning/)**: Introducción a Scikit-Learn, hiperparámetros y validación de modelos, Naive Bayes, Máquinas de Soporte Vectorial (SVM), *K-Means* y Mezclas Gaussianas (GMM).

---

## Autor

- **Luis Felipe García Domínguez**  
- Posgrado en Ciencias de la Computación — CICESE  
