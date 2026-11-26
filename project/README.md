# Proyecto Final: Identificación de Especies de Aves por Audio (BirdCLEF 2026)

**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — CICESE  
**Profesor:** Dr. Irvin Hussein López Nava  
**Estudiante:** Luis Felipe García Domínguez  

---

## Descripción del Proyecto

Este proyecto evalúa la clasificación de **30 especies de aves** (15 de vocalizaciones graves y 15 de vocalizaciones agudas) a partir de grabaciones de audio del conjunto de datos **BirdCLEF 2026** (Kaggle), comparando descriptores acústicos clásicos (**80 atributos MFCC + deltas** con `SVM` y `Random Forest`) frente a representaciones profundas preentrenadas en bioacústica (**Perch**, *Google Bird Vocalization Classifier*, 1,280 dimensiones).

Todo el modelado emplea validación cruzada agrupada por archivo físico (`StratifiedGroupKFold` de 5 pliegues sobre `filename`) con `StandardScaler` encapsulado dentro de `Pipeline` y búsqueda de hiperparámetros (`GridSearchCV` maximizando Macro-F1) para evitar fuga de información (*data leakage*).

### Preguntas de Investigación

El estudio se estructura alrededor de tres preguntas cuantitativas con criterios explícitos de evaluación:

1. **P1:** ¿Resultan más fáciles de identificar las aves con cantos graves que las de cantos agudos usando métodos clásicos? *Criterio:* diferencia significativa en F1 por especie (prueba U de Mann-Whitney, $\alpha = 0.05$).
2. **P2:** ¿Permiten las representaciones profundas preentrenadas distinguir mejor las especies que suelen confundirse entre sí? *Criterio:* reducción de al menos 15 % en la tasa de error ($1 - \text{F1}$) en las tres especies más difíciles del mejor modelo clásico.
3. **P3:** ¿Qué enfoque resiste mejor el deterioro del audio cuando aumenta el ruido ambiental y cantan varias especies al mismo tiempo? *Criterio:* menor caída relativa de Macro-F1 al bajar la calidad del audio.

### Resultados Principales

| Modelo | Mejores Hiperparámetros | Macro-F1 Ventana (OOF) | Macro-F1 Grabación (OOF) | Estrés Ventana | Estrés Grabación | Caída Rel. (Ventana) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM (MFCC)** | `C=1.0, gamma=0.01` | `0.6390` (`±0.0245`) | `0.6590` (`±0.0298`) | `0.4389` `[0.421, 0.456]` | `0.4623` `[0.442, 0.481]` | `-34.2%` |
| **Random Forest (MFCC)** | `n=300, depth=30, feat='sqrt'` | `0.5708` (`±0.0220`) | `0.5878` (`±0.0255`) | `0.3759` `[0.358, 0.392]` | `0.3951` `[0.374, 0.413]` | `-36.8%` |
| **Perch (Embeddings)** | `C=0.01` | **`0.9468`** (`±0.0039`) | **`0.9564`** (`±0.0070`) | **`0.8465`** `[0.833, 0.859]` | **`0.8765`** `[0.863, 0.890]` | **`-11.4%`** |

*(DE entre los 5 pliegues en OOF e intervalos de confianza al 95 % por bootstrap estratificado remuestreando grabaciones `filename`, $B = 1,000$, calculados directamente en la Sección 23.1 del cuaderno).*

- **Respuesta a P1 ($N=15$ por grupo):** Aunque los modelos sobre MFCCs promedian ligeramente más alto en especies graves que en agudas (`0.6645` vs. `0.6134` en SVM), la prueba U de Mann-Whitney no es estadísticamente significativa ($p = 0.3615$ en SVM; $p = 0.6187$ en Random Forest). La frecuencia dominante por sí sola no explica la dificultad de clasificación.
- **Respuesta a P2:** En las 3 especies con mayor error en `SVM (MFCC)` (`swthum1`, `saytan1`, `camfli1`), **Perch** reduce la tasa de error entre **79.7 % y 87.7 %**, superando ampliamente el criterio del 15 %.
- **Respuesta a P3:** Al pasar de `Alta Calidad` a `Estrés / Campo Real`, **Perch** presenta una caída relativa tres veces menor (`-11.4 %`) que los modelos clásicos (`-34.2 %` en SVM y `-36.8 %` en Random Forest) y alcanza **0.8765** de Macro-F1 a nivel grabación completa (`filename`).

---

## Origen de los Datos y Reproducibilidad

Los datos provienen de la competencia **[BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026)** en Kaggle (grabaciones de Xeno-canto e iNaturalist en formato `.ogg` a 32 kHz mono, junto con `train.csv` y `taxonomy.csv`).

- Por su tamaño (> 50 MB), las grabaciones crudas completas (`project/data/train_audio/`) no se incluyen en el repositorio.
- Para que el cuaderno pueda ejecutarse de principio a fin (`Restart & Run All`) en cualquier equipo sin descargar todo `train_audio/`, se incluyen en `project/data/ejemplos_audio/` únicamente los **6 audios `.ogg` de ejemplo** (1.34 MB en total, bajo licencias Creative Commons no comerciales con atribución) usados para graficar los espectrogramas de las secciones 14 y 24:
  - `picpig2/XC992009.ogg` — Bernabe Lopez-Lanus (CC BY-NC-SA, Xeno-canto)
  - `whbwar2/XC365331.ogg` — Lance A. M. Benner (CC BY-NC-SA, Xeno-canto)
  - `saytan1/iNat1659978.ogg` — Geografiaargentina (CC BY-NC, iNaturalist)
  - `swthum1/XC556107.ogg` — Jayrson Araujo de Oliveira (CC BY-NC-SA, Xeno-canto)
  - `saytan1/XC145695.ogg` — Antonio Silveira (CC BY-NC-SA, Xeno-canto)
  - `camfli1/XC620980.ogg` — Meena Haribal (CC BY-NC-SA, Xeno-canto)
- Para garantizar reproducibilidad directa sin exceder el límite de 50 MB por archivo, se incluyen las matrices precalculadas de las 30 especies (`features_mfcc_30_especies.csv`, 25.9 MB, y `embeddings_perch_30_especies.npy`, 40.5 MB en `float16`, reconvertidos a `float32` en memoria al cargar con impacto nulo en Macro-F1).
- El script [project/scripts/extract_features.py](scripts/extract_features.py) permite regenerar ambos archivos de forma idéntica tanto en local como directamente en un Kaggle Notebook con el dataset `birdclef-2026` adjunto.

---

## Estructura de la Carpeta

```text
project/
├── data/
│   ├── train.csv                                # Metadatos originales de BirdCLEF 2026
│   ├── taxonomy.csv                             # Taxonomía oficial de especies
│   ├── candidatas_56_especies_completo.csv      # 56 especies candidatas (Parte 1)
│   ├── muestra_descarga_56_especies.csv         # Muestra de 1,120 audios (Parte 2)
│   ├── metricas_muestra_56_especies.csv         # Métricas empíricas de duración y frecuencia
│   ├── train_subset_30_especies_limpio.csv      # 2,235 audios limpios (30 especies)
│   ├── train_subset_30_especies_completo.csv    # 4,793 audios totales (limpios + estrés)
│   ├── features_mfcc_30_especies.csv            # 15,834 ventanas × 80 atributos MFCC + deltas
│   ├── embeddings_perch_30_especies.npy         # 15,834 × 1,280 embeddings Perch (float16)
│   ├── ejemplos_audio/                          # 6 audios .ogg de ejemplo (1.40 MB) para espectrogramas
│   └── train_audio/                             # Audios .ogg completos locales (ignorado en Git)
├── notebooks/
│   └── project.ipynb                            # Cuaderno principal completamente ejecutado ([1..39])
├── scripts/
│   └── extract_features.py                      # Script reproducible de extracción (local / Kaggle)
├── requirements.txt                             # Dependencias de Python
└── README.md                                    # Documentación del proyecto
```

---

## Requisitos del Entorno

El proyecto fue desarrollado y probado en **Python 3.13.12** (compatible con Python $\ge$ 3.10). Las dependencias principales se encuentran en `project/requirements.txt`:

- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `scikit-learn >= 1.6.0`
- `librosa`, `soundfile`, `umap-learn`
- `tensorflow`, `kagglehub` (solo requeridos si se vuelve a ejecutar la extracción de Perch)
- `jupyter`, `ipykernel`

---

## Instrucciones de Instalación y Ejecución

### 1. Crear y activar el entorno virtual
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

### 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r project/requirements.txt
```

### 3. (Opcional) Regenerar características desde audios `.ogg`

#### Opción A: Reproducción en Kaggle Notebook (recomendada)
Permite leer los `.ogg` directamente desde el dataset montado en Kaggle (sin límites de tasa de la API) y acelerar la inferencia de Perch con GPU T4 (~3–5 minutos):

1. **Crear cuenta en Kaggle:** Registrarse o iniciar sesión en [kaggle.com](https://www.kaggle.com/) (es necesario verificar un número telefónico en la cuenta para habilitar aceleradores GPU y acceso a Internet en notebooks).
2. **Aceptar las reglas de la competencia:** Ingresar a **[BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026)**, hacer clic en *Join Competition* y aceptar los términos y condiciones (*I Understand and Accept*). Sin este paso Kaggle no permite montar los audios del reto.
3. **Subir los insumos del repositorio como Dataset privado:** En la sección *Datasets → New Dataset* (por ejemplo, nombrándolo `icd2026`, que quedará montado en `/kaggle/input/datasets/<usuario>/<dataset>/`), subir estos dos archivos:
   - `project/scripts/extract_features.py`
   - `project/data/train_subset_30_especies_completo.csv`
4. **Crear y configurar el Kaggle Notebook:** Ir a *Code → New Notebook* y, en el panel derecho (*Session options*), seleccionar el acelerador **GPU T4** y activar **Internet** (necesario para que `kagglehub` descargue el modelo Perch).
5. **Adjuntar las entradas (*Add Input*):** Desde el panel derecho (*Input → Add Input*), añadir tanto la competencia **`birdclef-2026`** como el Dataset privado creado en el paso 3.
6. **Ejecutar la extracción en una celda del notebook** (rutas verificadas en Kaggle):
   ```bash
   !python /kaggle/input/datasets/<usuario>/<dataset>/extract_features.py \
       --csv /kaggle/input/datasets/<usuario>/<dataset>/train_subset_30_especies_completo.csv \
       --audio-dir /kaggle/input/competitions/birdclef-2026/train_audio \
       --out-dir /kaggle/working
   ```
   *(Por ejemplo, en la corrida de referencia: `/kaggle/input/datasets/lgdom01/icd2026/`).*
7. **Descargar las matrices generadas:** Desde `/kaggle/working/` (pestaña *Output*), descargar los dos archivos resultantes y colocarlos en `project/data/`:
   - `features_mfcc_30_especies.csv`
   - `embeddings_perch_30_especies.npy`
8. *(Opcional — solo si se desean todos los `.ogg` en local)* Empaquetar las grabaciones de las 30 especies en un solo `.zip` desde el mismo Kaggle Notebook y descomprimirlo en `project/data/train_audio/`:
   ```python
   import os, shutil, pandas as pd

   df = pd.read_csv('/kaggle/input/datasets/<usuario>/<dataset>/train_subset_30_especies_completo.csv')
   especies = sorted(df['primary_label'].unique())

   os.makedirs('/kaggle/working/tmp_audio', exist_ok=True)
   for sp in especies:
       shutil.copytree(
           f'/kaggle/input/competitions/birdclef-2026/train_audio/{sp}',
           f'/kaggle/working/tmp_audio/{sp}',
           dirs_exist_ok=True
       )
   shutil.make_archive('/kaggle/working/train_audio_30_especies', 'zip', '/kaggle/working/tmp_audio')
   ```

#### Opción B: Reproducción local
Con los audios `.ogg` de las 30 especies descargados dentro de `project/data/train_audio/`, ejecutar desde la raíz del repositorio:
```bash
python project/scripts/extract_features.py
```

### 4. Ejecutar el cuaderno principal
```bash
jupyter notebook project/notebooks/project.ipynb
```
Seleccionar **Kernel -> Restart Kernel and Run All Cells**. La ejecución completa de principio a fin toma alrededor de 15 minutos.
