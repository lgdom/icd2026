# Tarea 2: Preprocesamiento de Datos
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción del Proyecto

Este proyecto aborda el flujo de auditoría, análisis exploratorio (EDA) y preprocesamiento de datos sobre el conjunto **Breast Cancer Wisconsin (Original)**.

El objetivo principal es aplicar dos técnicas centrales de preprocesamiento. Se eligió:
1. **Limpieza, imputación y escalado de características.**
2. **Reducción de dimensionalidad mediante Análisis de Componentes Principales (PCA).**

---

## El Conjunto de Datos

- **Fuente:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)
- **Instancias:** 699 observaciones (biopsias con aguja fina).
- **Atributos:** 10 variables numéricas morfológicas (rango entero 1 a 10), 1 identificador de muestra (`id`) y 1 etiqueta de clase (`class`).
- **Valores faltantes:** 16 registros con `'?'` en la columna `bare_nuclei`.

### Diccionario de variables

| Variable | Tipo | Rango | Descripción |
|---|---|:---:|---|
| `id` | Entero | N/A | Código de identificación de la muestra |
| `clump_thickness` | Numérico | 1 – 10 | Grosor del grupo de células |
| `uniformity_cell_size` | Numérico | 1 – 10 | Uniformidad del tamaño celular |
| `uniformity_cell_shape` | Numérico | 1 – 10 | Uniformidad de la forma celular |
| `marginal_adhesion` | Numérico | 1 – 10 | Adhesión marginal entre células vecinas |
| `single_epithelial_size` | Numérico | 1 – 10 | Tamaño de la célula epitelial única |
| `bare_nuclei` | Numérico | 1 – 10 | Núcleos desnudos (contiene 16 valores faltantes) |
| `bland_chromatin` | Numérico | 1 – 10 | Textura de la cromatina nuclear |
| `normal_nucleoli` | Numérico | 1 – 10 | Presencia y tamaño de nucléolos normales |
| `mitoses` | Numérico | 1 – 10 | Ritmo de actividad mitótica |
| `class` | Categórico | 2, 4 | Diagnóstico médico (`2 = Benigno`, `4 = Maligno`) |

---

## Estructura del Directorio

```text
homework/preprocesamiento/
├── data/
│   ├── breast-cancer-wisconsin.data   # Conjunto de datos en texto plano (699 filas)
│   └── breast-cancer-wisconsin.names  # Descripción y metadatos oficiales del repositorio UCI
├── notebooks/
│   └── preprocesamiento.ipynb         # Cuaderno Jupyter con el análisis y transformaciones
└── README.md                          # Documentación del proyecto
```

---

## Hallazgos y Conclusiones

1. **Eficiencia en la imputación:** La imputación con la mediana evitó sacrificar el 2.3% de las observaciones del estudio, manteniendo la tendencia central y la forma de la distribución de `bare_nuclei`.
2. **Redundancia morfológica:** La fuerte correlación observada en el EDA justificó la reducción dimensional.
3. **Separabilidad preservada:** A pesar de comprimir 9 dimensiones a solo 2 componentes principales (PCA), las muestras benignas y malignas se diferencian claramente en el plano cartesiano.

---

## Requisitos y Entorno de Ejecución

Para reproducir el cuaderno se requiere Python 3.10+ y las siguientes bibliotecas:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Para abrir y ejecutar el cuaderno:

```bash
jupyter notebook preprocesamiento.ipynb
```
