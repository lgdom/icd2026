# Tarea 5: CNN compacta con Global Average Pooling (GAP)
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción de la tarea

Esta tarea explora el diseño de una red neuronal convolucional (CNN) compacta con **Global Average Pooling (GAP)** sobre MNIST que supere el **95 % de exactitud en validación** usando **menos de 54 330 parámetros entrenables** (el tamaño de la CNN base con Flatten) y evalúa qué tan bien conserva su desempeño al desplazar los dígitos de posición sin reentrenar. Se abordan los siguientes pasos:

1. **Preparación y exploración inicial:** Reutilización de las particiones estratificadas del notebook de clase (`8 000` entrenamiento, `1 500` validación y `2 000` prueba con `SEMILLA = 42`), adición de un margen negro de 4 píxeles por lado (`36 × 36 × 1`), verificación del balance de clases y comparación visual antes/después del margen.
2. **Modelos de referencia:** Recuperación de `CNN + Flatten base` (`54 330` parámetros, `96.93 %` en validación) y `CNN + GAP base` (`2 650` parámetros, `44.07 %` en validación) bajo un presupuesto fijo de 20 épocas (`LOTE = 128`, `Adam`, `TASA = 0.001`).
3. **Exploración de arquitecturas con GAP:** Entrenamiento de 13 variantes con GAP (`V1`–`V13`), incluyendo comparaciones controladas cambiando una sola decisión (ubicación del pooling, tamaño de kernel, profundidad y número de filtros), más una búsqueda focalizada de variantes compactas de 4 capas y 64 canales finales.
4. **Selección del modelo:** Elección de **V8 · Pirámide 16-32-64** (`23 946` parámetros, `95.40 %` en validación), que ahorra **30 384 parámetros** (**55.92 %**) respecto a `CNN + Flatten base`.
5. **Evaluación ante desplazamiento en prueba:** Comparación de `CNN + Flatten base`, `CNN + GAP base` y `GAP seleccionado (V8)` a `0`, `2` y `4` píxeles (promediando las 4 direcciones para 2 y 4 px) sobre `X_prueba`, sin reentrenar.

---

## Resumen de resultados en prueba (desplazamiento)

| Modelo | Parámetros entrenables | 0 px (%) | 2 px (%) | 4 px (%) | Caída a 4 px (pp) |
|---|---:|---:|---:|---:|---:|
| CNN + Flatten base | 54 330 | 97.10 | 82.79 | 37.30 | 59.80 |
| CNN + GAP base | 2 650 | 46.45 | 46.46 | 46.46 | -0.01 |
| **GAP seleccionado (V8)** | **23 946** | **96.15** | **87.95** | **72.51** | **23.64** |

---

## Estructura del Directorio

```text
homework/cnn/
├── data/
├── notebooks/
│   └── cnn.ipynb          # Cuaderno Jupyter ejecutado con experimentos, gráficas y conclusiones
├── requirements.txt       # Dependencias del cuaderno
└── README.md              # Documentación de la tarea
```

---

## Requisitos y Entorno de Ejecución

```bash
pip install -r requirements.txt
jupyter notebook notebooks/cnn.ipynb
```
