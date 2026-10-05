# Práctica 2: Preprocesamiento de Datos
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción del Proyecto

Este repositorio contiene el desarrollo de la **Práctica 2**, enfocada en aplicar técnicas de **preprocesamiento de datos** sobre la base oficial de **Accidentes de Tránsito Terrestre en Zonas Urbanas y Suburbanas (ATUS 2025)** del **Instituto Nacional de Estadística y Geografía (INEGI)**.

A diferencia del análisis exploratorio inicial de la Práctica 1 (donde se recurrió al descarte descriptivo de datos faltantes), en esta práctica se diseñó e implementó un flujo de ingeniería de datos para transformar el conjunto en uno listo para tareas de aprendizaje automático.

- **Fuente oficial:** [INEGI — Accidentes de Tránsito Terrestre](https://www.inegi.org.mx/programas/accidentes/#datos_abiertos)
- **Periodo:** 1 de enero al 31 de diciembre de 2025.
- **Unidad de observación:** Evento individual de accidente de tránsito registrado en México.
- **Dimensiones originales (crudo INEGI):** 380,391 registros y 47 atributos.
- **Subconjunto de trabajo inicial:** 380,391 registros y 39 atributos seleccionados.
- **Conjunto final preprocesado:** 380,391 registros y 53 atributos (+14 variables ingenierizadas).

---

## Resumen de Técnicas de Preprocesamiento Aplicadas

| Técnica | Problema original en ATUS 2025 | Técnica aplicada | Impacto / Beneficio | Variables resultantes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Limpieza de datos** | 88,071 registros (23.15 %) con códigos administrativos `0` (*se fugó*) y `99` (*no especificado*) en la edad del conductor. | Imputación por **mediana** (35 años) con variables indicadoras de contexto. *(Extra: exploración estocástica).* | Se conservó el 100 % de las observaciones ($N = 380,391$) sin sesgo de selección por descarte. | `SE_FUGO`<br>`EDAD_ERA_FALTANTE`<br>`EDAD_IMPUTADA`<br>*(Extra: `EDAD_ESTOCASTICA`)* |
| **2. Aumento de datos** | Desbalance de 14 a 1 en pruebas de alcoholemia confirmadas (`No`: 93.4 %, `Sí`: 6.6 %). | Muestras sintéticas con **SMOTE-NC** ($k=5$). | Igualación al 50 % - 50 % ($229,476$ casos por clase). Se previno el sesgo hacia la clase mayoritaria. | Conjunto balanceado:<br>`df_aliento_balanceado`<br>($458,952$ registros $\times$ 7 cols) |
| **3. Extracción de características** | Discontinuidad en la escala horaria y valores vehiculares y de víctimas dispersos. | Representación trigonométrica continua $(\sin, \cos)$ y agregación de variables. | Continuidad entre noche/madrugada y diciembre/enero. Se redujeron 25 columnas en 4 descriptores con mayor información. | `HORA_SIN`<br>`HORA_COS`<br>`MES_SIN`<br>`MES_COS`<br>`TOTAL_VEHICULOS`<br>`TOTAL_VICTIMAS`<br>`INDICE_LETALIDAD` |

---

## Nota sobre los Datos (Single-Source)

Para evitar duplicar innecesariamente **95 MB** en el repositorio de GitHub, esta práctica reutiliza directamente los archivos oficiales alojados en `practices/practice 1/data/`:
* `atus_anual_2025.csv`: Base de datos de accidentes de tránsito 2025 (INEGI).
* `tc_entidad.csv`: Catálogo de entidades federativas utilizado para el mapeo institucional de la variable `ENTIDAD`.

---

## Estructura de la Práctica

```text
practice 2/
├── data/
│   └── README.md          # Nota de referencia a los datos de practice 1
├── notebooks/
│   └── practice-02.ipynb  # Cuaderno principal con el preprocesamiento completo
├── requirements.txt       # Dependencias requeridas del entorno
└── README.md              # Documentación de la práctica
```

---

## Requisitos del Entorno

El proyecto requiere **Python 3.10** o superior (probado en Python 3.13) con las siguientes librerías:

- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.2.0`
- `imbalanced-learn >= 0.12.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `jupyter >= 1.0.0` / `ipykernel`

---

## Instrucciones de Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd "icd2026/practices/practice 2"
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
jupyter notebook notebooks/practice-02.ipynb
```
O bien abrir directamente el archivo `notebooks/practice-02.ipynb` en Visual Studio Code o el editor de su preferencia con soporte para Jupyter Notebooks.
