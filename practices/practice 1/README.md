# Práctica 1: Análisis Exploratorio de Datos (EDA)
**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  

---

## Descripción del Proyecto

Este repositorio contiene el desarrollo de la **Práctica 1**, enfocada en realizar un análisis exploratorio de datos (EDA). Se ha seleccionado la base de datos oficial de **Accidentes de Tránsito Terrestre en Zonas Urbanas y Suburbanas (ATUS 2025)**, recopilada y publicada por el **Instituto Nacional de Estadística y Geografía (INEGI)**.

- **Fuente oficial:** [INEGI — Accidentes de Tránsito Terrestre](https://www.inegi.org.mx/programas/accidentes/#datos_abiertos)
- **Metadatos y Diccionario:** [Catálogo Nacional de Metadatos del INEGI](https://www.inegi.org.mx/rnm/index.php/catalog/1115/data-dictionary)
- **Periodo:** 1 de enero al 31 de diciembre de 2025.
- **Unidad de observación:** Cada evento de accidente de tránsito terrestre registrado en zonas urbanas y suburbanas del territorio nacional.
- **Dimensiones originales:** 380,391 registros y 47 atributos.
- **Subconjunto de trabajo:** 26 atributos seleccionados

## Nota sobre el dataset

Cuando se accede al repositorio desde la fuente oficial, la descarga del archivo `conjunto_de_datos_atus_anual_csv.zip` contiene los datos del año 1997 al 2025. Sin embargo, para este proyecto solo se utiliza el año 2025.
Así mismo, contiene los metadatos, el diccionario de datos y un conjunto de catálogos, entre ellos el catálogo de entidades federativas `tc_entidad.csv`, el cual se utiliza en la presente práctica para facilitar la interpretación de los datos de la variable `ENTIDAD`.

---

## Estructura de la práctica

```text
practice 1/
├── data/
│   ├── atus_anual_2025.csv                          # Base principal ATUS 2025 (INEGI)
│   ├── diccionario_de_datos_atus_anual_1997_2025.csv # Catálogo de variables y descripciones
│   ├── metadatos_atus_anual_1997_2025.txt           # Ficha metodológica oficial
│   └── tc_entidad.csv                               # Catálogo de entidades federativas
├── notebooks/
│   └── practice-01.ipynb                            # Cuaderno principal con el EDA completo
├── requirements.txt                                 # Dependencias de Python requeridas
├── .gitignore                                       # Archivos excluidos del control de versiones
└── README.md                                        # Documentación del proyecto
```

---

## Requisitos del Entorno

El proyecto requiere **Python 3.10** o superior (probado en Python 3.13) con las siguientes librerías:

- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `jupyter >= 1.0.0` / `ipykernel`

---

## Instrucciones de Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd "practice 1"
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
jupyter notebook notebooks/practice-01.ipynb
```
O bien abrir el archivo `notebooks/practice-01.ipynb` directamente en Visual Studio Code o el editor de su preferencia con soporte para Jupyter Notebooks.
