# Práctica 4: Meta aprendizaje
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción del Proyecto

Esta práctica analiza **cuánto ganan los métodos de ensamble sobre clasificadores individuales base**, evaluando tres familias principales:

1. **Bagging (reducción de varianza):** Bagging de árboles sin podar (`warm_start=True`), Random Forest y Extra-Trees optimizados con $\ge 50$ iteraciones de `RandomizedSearchCV`.
2. **Boosting (reducción de sesgo):** AdaBoost con tocones (`staged_predict()`), XGBoost y LightGBM optimizados con $\ge 50$ iteraciones de `RandomizedSearchCV`.
3. **Combinación de modelos heterogéneos:** Votación dura, votación suave (con calibración de probabilidades de SVM vía `CalibratedClassifierCV`) y Stacking con meta-modelo de Regresión Logística.

Se desarrolla en dos fases:

- **Parte 1 (Deserción Estudiantil — UCI 697):** Evaluación multiclase (3 clases: `Dropout`, `Enrolled`, `Graduate`) al cierre del primer semestre, comparando los ensambles contra los cinco clasificadores base optimizados en la Práctica 3 bajo validación cruzada estratificada de 5 pliegues (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

- **Parte 2 (Bank Marketing — Tarea 4):** Búsqueda del mejor modelo para predecir la suscripción a depósitos de una campaña de marketing (`bank-additional-full.csv`), bajo evaluación estricta con `cross_validate(modelo, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=2026), scoring="f1_macro")`. Todo preprocesamiento, escalamiento, codificación y calibración de umbral de decisión reside dentro del objeto `Pipeline` / `TunedThresholdClassifierCV` para prevenir fuga de información (*data leakage*).

El modelo ganador de Bank Marketing se exporta como `luis_garcia-lgbm_umbral.joblib`.

---

## Descripción de los Conjuntos de Datos

### 1. Deserción y Éxito Académico en Educación Superior (UCI 697)
* **Origen:** Repositorio oficial [UCI Machine Learning Repository: Predict Students' Dropout and Academic Success (ID: 697)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), recopilado por el Instituto Politécnico de Portalegre (Portugal; Realinho et al., 2022).
* **Volumen:** 3,539 instancias en el archivo de entrenamiento (`desercion-entrenamiento.csv`) sin valores nulos.
* **Variable objetivo (`Target`):** Categórica multiclase con desbalance moderado:
  - `Graduate` (graduado): 1,767 estudiantes (49.93 %)
  - `Dropout` (desertor): 1,137 estudiantes (32.13 %)
  - `Enrolled` (matriculado / en curso): 635 estudiantes (17.94 %)
* **Atributos predictivos (Escenario al cierre del 1er semestre):** Se utilizan las 30 variables disponibles al ingreso y al término del primer semestre (excluyendo estrictamente las 6 columnas del segundo semestre):
  - *Demográficos y socioeconómicos:* edad al inscribirse, género, estado civil, nacionalidad, becario, deudor, colegiatura al corriente, ocupación y escolaridad de los padres.
  - *Académicos previos y de ingreso:* carrera (`Course`), modo de solicitud, orden de preferencia, calificación de estudios previos y nota de examen de admisión.
  - *Desempeño del 1er semestre:* unidades curriculares acreditadas, inscritas, evaluadas, aprobadas y promedio ponderado de calificaciones.
  - *Contexto macroeconómico nacional:* tasa de desempleo, tasa de inflación y variación del PIB.

### 2. Campañas de Marketing Bancario Directo (Bank Marketing)
* **Origen:** Repositorio oficial [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), recopilado a partir de campañas telefónicas de captación de depósitos a plazo de una entidad bancaria portuguesa entre mayo de 2008 y noviembre de 2010 (Moro et al., 2014).
* **Volumen:** 41,188 instancias y 20 atributos en el archivo `bank-additional-full.csv`.
* **Eliminación de `duration`:** La duración del contacto telefónico en segundos se excluye antes del modelado, ya que se desconoce antes de realizar la llamada.
* **Variable objetivo (`y`):** Binaria con desbalance severo:
  - `no` (rechazó el depósito): 36,548 clientes (**88.73 %**)
  - `yes` (suscribió el depósito): 4,640 clientes (**11.27 %**)
* **Atributos predictivos (19 variables restantes):**
  - *Datos del cliente:* edad (`age`), profesión (`job`), estado civil (`marital`), nivel educativo (`education`), historial crediticio en mora (`default`), crédito hipotecario (`housing`), préstamo personal (`loan`).
  - *Datos de la campaña actual:* canal de contacto (`contact`), mes (`month`), día de la semana (`day_of_week`), número de contactos realizados durante la campaña (`campaign`).
  - *Historial de campañas previas:* días transcurridos desde el último contacto (`pdays`, donde 999 indica no contactado previamente), número de contactos previos (`previous`), resultado de la campaña previa (`poutcome`).
  - *Variables de contexto macroeconómico y social (a nivel mensual/trimestral):* tasa de variación del empleo (`emp.var.rate`), índice de precios al consumidor (`cons.price.idx`), índice de confianza del consumidor (`cons.conf.idx`), tasa Euribor a 3 meses (`euribor3m`), número de empleados en nómina nacional (`nr.employed`).

---

## Resultados Principales

### Parte 1: Deserción Estudiantil (Top 5 de 16 modelos evaluados)

| Posición | Modelo | Familia | F1 macro (media ± std) | Sensibilidad `Enrolled` | Tiempo (s) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1°** | **LightGBM optimizado** *(Mejor modelo Parte 1)* | Boosting | **0.6986 ± 0.0163** | **60.00 %** | 6.4 |
| **2°** | **Random Forest optimizado** | Bagging | 0.6865 ± 0.0100 | 55.12 % | 20.2 |
| **3°** | **Regresión Logística (Práctica 3)** | Lineal | 0.6862 ± 0.0176 | 54.02 % | 0.8 |
| **4°** | **Extra-Trees optimizado** | Bagging | 0.6857 ± 0.0118 | 52.91 % | 16.1 |
| **5°** | **Votación dura (5 modelos base)** | Ensamble | 0.6835 ± 0.0093 | 51.34 % | 0.5 |

### Parte 2: Bank Marketing (Evaluación en 5 pliegues con `SEMILLA_BANK = 2026`)

| Posición | Modelo | Estrategia de desbalance | F1 macro (media ± std) | Sensibilidad `yes` | Tiempo (s) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1°** | **Votación suave (LGBM+XGB+RF) + Umbral** | Umbral interno (`cv=3`) | **0.7172 ± 0.0052** | 55.19 % | 15.64 |
| **2°** | **LightGBM optimizado + Umbral** *(Modelo serializado)* | Umbral interno (`cv=3`) | **0.7168 ± 0.0047** | 54.78 % | 5.49 |
| **3°** | **XGBoost + Umbral** | Umbral interno (`cv=3`) | 0.7110 ± 0.0049 | 50.06 % | 4.71 |
| **4°** | **LightGBM (balanceado)** | Pesos de clase | 0.6922 ± 0.0043 | 62.80 % | 0.80 |
| **5°** | **Random Forest (balanceado)** | Pesos de clase | 0.6895 ± 0.0047 | 46.66 % | 1.68 |
| **6°** | **Random Forest + Umbral** | Umbral interno (`cv=3`) | 0.6893 ± 0.0051 | 47.03 % | 7.27 |
| **7°** | **XGBoost (scale_pos_weight)** | Ponderación minoritaria | 0.6764 ± 0.0034 | 58.32 % | 0.70 |
| **8°** | **Regresión Logística (balanceada)** | Pesos de clase | 0.6739 ± 0.0040 | 62.87 % | 0.44 |
| **9°** | **Extra-Trees (balanceado)** | Pesos de clase | 0.6523 ± 0.0075 | 31.40 % | 2.20 |

---

## Estructura de la Carpeta

```text
practice 4/
├── data/
│   ├── desercion-entrenamiento.csv   # Dataset Parte 1 (3,539 instancias)
│   └── bank-additional-full.csv      # Dataset Parte 2 (41,188 instancias)
├── notebooks/
│   ├── practice-04.ipynb             # Cuaderno principal completamente ejecutado
│   └── luis_garcia-lgbm_umbral.joblib # Modelo final (copia local)
├── luis_garcia-lgbm_umbral.joblib     # Modelo final para entrega
├── requirements.txt                  # Requisitos y dependencias de Python
└── README.md                         # Documentación de la práctica
```

---

## Requisitos del Entorno

El proyecto fue desarrollado y probado en **Python 3.13** (compatible con Python $\ge$ 3.10). Las dependencias principales y sus versiones utilizadas son:

- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.6.0` (requerido para `TunedThresholdClassifierCV`)
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `xgboost >= 2.0.0`
- `lightgbm >= 4.0.0`
- `joblib >= 1.3.0`
- `jupyter >= 1.0.0`
- `ipykernel >= 6.0.0`

---

## Instrucciones de Instalación y Ejecución

### 1. Clonar el repositorio y ubicarse en la carpeta
```bash
git clone <URL_DEL_REPOSITORIO>
cd "icd2026/practices/practice 4"
```

### 2. Crear y activar el entorno virtual
En macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
En Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar las dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar el cuaderno de Jupyter
```bash
jupyter notebook notebooks/practice-04.ipynb
```
O bien:
```bash
jupyter lab notebooks/practice-04.ipynb
```
Seleccionar **Kernel -> Restart Kernel and Run All Cells**. El tiempo de ejecución completo de las 49 celdas es de alrededor de 2 a 3 minutos.

---

## Uso del Modelo (`.joblib`)

Para cargar y utilizar el modelo final entrenado sobre el 100 % de Bank Marketing:

```python
import joblib
import pandas as pd

# Cargar el pipeline completo con umbral calibrado
modelo = joblib.load("luis_garcia-lgbm_umbral.joblib")

# X_nuevos debe tener las 19 columnas de Bank Marketing (sin duration ni y)
# predicciones = modelo.predict(X_nuevos)
# probabilidades = modelo.predict_proba(X_nuevos)[:, 1]
```