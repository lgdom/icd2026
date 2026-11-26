"""
Extracción reproducible de características acústicas (MFCCs + Perch) para BirdCLEF 2026.

Este script genera:
1. `features_mfcc_30_especies.csv`: 80 atributos (20 MFCCs + 20 deltas, media y DE) por ventana
   activa de 5 s seleccionada mediante VAD (350–8,500 Hz, planicidad espectral, Top-k con k=4).
2. `embeddings_perch_30_especies.npy`: embeddings de 1,280 dimensiones extraídos con Perch
   (Google Bird Vocalization Classifier) almacenados en `float16` para mantenerse por debajo
   de 50 MB sin pérdida práctica de precisión al reconvertirse a `float32` en memoria.

Uso en local (desde la raíz del repositorio):
    .venv/bin/python project/scripts/extract_features.py

Uso en Kaggle Notebook (con dataset birdclef-2026 adjunto y GPU T4 recomendada):
    python extract_features.py \
        --csv /kaggle/input/<tu-dataset>/train_subset_30_especies_completo.csv \
        --audio-dir /kaggle/input/birdclef-2026/train_audio \
        --out-dir /kaggle/working
"""

import argparse
import os
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# Parámetros globales de preprocesamiento (idénticos a project.ipynb)
SR_OBJETIVO = 22050
SR_PERCH = 32000
DURACION_MAX_SEG = 120.0
VENTANA_SEG = 5.0
SALTO_SEG = 2.5
F_MIN_HZ = 350
F_MAX_HZ = 8500
K_MAX_VENTANAS = 4
N_MFCC = 20

N_MUESTRAS_WIN = int(VENTANA_SEG * SR_OBJETIVO)
N_MUESTRAS_SALTO = int(SALTO_SEG * SR_OBJETIVO)
N_WIN_PERCH = int(VENTANA_SEG * SR_PERCH)


def segmentar_audio(y, n_win=N_MUESTRAS_WIN, n_salto=N_MUESTRAS_SALTO):
    """Divide una señal en ventanas de 5 s con 50% de traslape; aplica padding si dura < 5 s."""
    if len(y) < n_win:
        y_pad = np.pad(y, (0, n_win - len(y)), mode='constant')
        return [(0.0, y_pad)]

    ventanas = []
    for inicio in range(0, len(y) - n_win + 1, n_salto):
        fin = inicio + n_win
        t_inicio = inicio / SR_OBJETIVO
        ventanas.append((t_inicio, y[inicio:fin]))
    return ventanas


def evaluar_ventana_vad(y_win, sr=SR_OBJETIVO, f_min=F_MIN_HZ, f_max=F_MAX_HZ):
    """Calcula energía en 350–8,500 Hz, planicidad espectral (p25) y Score compuesto."""
    S = np.abs(librosa.stft(y_win, n_fft=1024, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    mask = (freqs >= f_min) & (freqs <= f_max)
    S_banda = S[mask, :]

    energia_banda = float(np.mean(S_banda ** 2))

    # Planicidad espectral dentro de la banda (0 = tonal, 1 = ruido plano)
    S_pow = (S_banda ** 2) + 1e-10
    geo_mean = np.exp(np.mean(np.log(S_pow), axis=0))
    arith_mean = np.mean(S_pow, axis=0)
    flatness_frames = geo_mean / arith_mean
    flatness = float(np.percentile(flatness_frames, 25))

    score = energia_banda * (1.0 - flatness)
    return energia_banda, flatness, score


def seleccionar_ventanas_top_k(ventanas, k=K_MAX_VENTANAS):
    """Aplica piso de ruido adaptativo (p15) en audios largos y selecciona Top-k por Score."""
    evaluadas = []
    for t_ini, y_win in ventanas:
        energia, flat, score = evaluar_ventana_vad(y_win)
        evaluadas.append({
            't_inicio': t_ini,
            'y_win': y_win,
            'energia': energia,
            'flatness': flat,
            'score': score,
        })

    if len(evaluadas) > k:
        energias = [v['energia'] for v in evaluadas]
        piso_ruido = np.percentile(energias, 15)
        filtradas = [v for v in evaluadas if v['energia'] > piso_ruido * 1.1]
        if len(filtradas) >= k:
            evaluadas = filtradas

    evaluadas.sort(key=lambda x: x['score'], reverse=True)
    return evaluadas[:k]


def extraer_mfcc_ventana(y_win, sr=SR_OBJETIVO, n_mfcc=N_MFCC):
    """Extrae media y desviación estándar de 20 MFCCs y sus 20 deltas (80 atributos)."""
    mfcc = librosa.feature.mfcc(y=y_win, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)

    feats = {}
    for i in range(n_mfcc):
        feats[f'mfcc_{i + 1}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc_{i + 1}_std'] = float(np.std(mfcc[i]))
        feats[f'delta_{i + 1}_mean'] = float(np.mean(delta[i]))
        feats[f'delta_{i + 1}_std'] = float(np.std(delta[i]))
    return feats


def extraer_mfccs(df_meta, audio_dir, ruta_salida, forzar=False):
    """Ejecuta VAD y extracción de 80 MFCCs+deltas sobre todos los audios del subconjunto."""
    if os.path.exists(ruta_salida) and not forzar:
        df_features = pd.read_csv(ruta_salida)
        print(f"[MFCC] Caché encontrado: {ruta_salida} ({df_features.shape[0]:,} ventanas).")
        return df_features

    # Verificar que los archivos existan antes de procesar
    faltantes = [f for f in df_meta['filename'] if not os.path.exists(os.path.join(audio_dir, f))]
    if faltantes:
        especies_falt = sorted({f.split('/')[0] for f in faltantes})
        raise FileNotFoundError(
            f"Faltan {len(faltantes)} archivos .ogg en '{audio_dir}'. "
            f"Especies incompletas ({len(especies_falt)}): {especies_falt}"
        )

    t0 = time.time()
    filas = []
    total = len(df_meta)
    for i, (_, row) in enumerate(df_meta.iterrows(), start=1):
        ruta_ogg = os.path.join(audio_dir, row['filename'])
        y, _ = librosa.load(ruta_ogg, sr=SR_OBJETIVO, mono=True, duration=DURACION_MAX_SEG)

        ventanas = segmentar_audio(y)
        elegidas = seleccionar_ventanas_top_k(ventanas, k=K_MAX_VENTANAS)

        for rank, v in enumerate(elegidas, start=1):
            feats = extraer_mfcc_ventana(v['y_win'], sr=SR_OBJETIVO, n_mfcc=N_MFCC)
            feats['filename'] = row['filename']
            feats['primary_label'] = row['primary_label']
            feats['common_name'] = row['common_name']
            feats['grupo_acustico'] = row['grupo_acustico']
            feats['estrato_calidad'] = row['estrato_calidad']
            feats['ventana_rank'] = rank
            feats['t_inicio_s'] = round(v['t_inicio'], 2)
            feats['vad_score'] = v['score']
            filas.append(feats)

        if i % 500 == 0 or i == total:
            print(f"[MFCC] Procesados {i:,}/{total:,} audios ({time.time() - t0:.1f} s)...")

    df_features = pd.DataFrame(filas)
    df_features.to_csv(ruta_salida, index=False)
    tam_mb = os.path.getsize(ruta_salida) / 1e6
    print(f"[MFCC] Guardado: {ruta_salida} ({df_features.shape}, {tam_mb:.2f} MB).")
    return df_features


def extraer_perch(df_features, audio_dir, ruta_salida, batch_size=32, forzar=False):
    """Extrae embeddings de Perch (1,280 dims a 32 kHz) y los guarda en float16 (< 50 MB)."""
    if os.path.exists(ruta_salida) and not forzar:
        emb = np.load(ruta_salida)
        print(f"[Perch] Caché encontrado: {ruta_salida} (shape={emb.shape}, dtype={emb.dtype}).")
        return emb

    import kagglehub
    import tensorflow as tf

    print("[Perch] Cargando modelo google/bird-vocalization-classifier...")
    model_path = kagglehub.model_download(
        'google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier'
    )
    infer = tf.saved_model.load(model_path).signatures['serving_default']

    t0 = time.time()
    n_total = len(df_features)
    embeddings = np.zeros((n_total, 1280), dtype=np.float32)

    batch_x, batch_idx = [], []
    grupos = list(df_features.groupby('filename', sort=False))
    total_files = len(grupos)

    for f_i, (fname, grp) in enumerate(grupos, start=1):
        y, sr = sf.read(os.path.join(audio_dir, fname), dtype='float32')
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SR_PERCH:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR_PERCH)

        for idx, row in grp.iterrows():
            start = int(round(row['t_inicio_s'] * SR_PERCH))
            win = y[start : start + N_WIN_PERCH]
            if len(win) < N_WIN_PERCH:
                win = np.pad(win, (0, N_WIN_PERCH - len(win)), mode='constant')
            batch_x.append(win)
            batch_idx.append(idx)

            if len(batch_x) == batch_size:
                out = infer(inputs=tf.convert_to_tensor(np.stack(batch_x), dtype=tf.float32))
                embeddings[batch_idx] = out['embedding'].numpy()
                batch_x.clear()
                batch_idx.clear()

        if f_i % 500 == 0 or f_i == total_files:
            print(f"[Perch] Procesados {f_i:,}/{total_files:,} archivos ({time.time() - t0:.1f} s)...")

    if batch_x:
        out = infer(inputs=tf.convert_to_tensor(np.stack(batch_x), dtype=tf.float32))
        embeddings[batch_idx] = out['embedding'].numpy()

    # Guardar en float16 para mantener el archivo por debajo de 50 MB sin pérdida práctica
    embeddings_f16 = embeddings.astype(np.float16)
    np.save(ruta_salida, embeddings_f16)
    tam_mb = os.path.getsize(ruta_salida) / 1e6
    print(
        f"[Perch] Guardado: {ruta_salida} "
        f"(shape={embeddings_f16.shape}, dtype={embeddings_f16.dtype}, {tam_mb:.2f} MB)."
    )
    return embeddings_f16


def main():
    parser = argparse.ArgumentParser(
        description="Extracción reproducible de MFCCs y embeddings de Perch (30 especies)."
    )
    parser.add_argument(
        '--csv',
        default='project/data/train_subset_30_especies_completo.csv',
        help='Ruta al CSV con los metadatos del subconjunto de 30 especies.'
    )
    parser.add_argument(
        '--audio-dir',
        default='project/data/train_audio',
        help='Carpeta raíz con las subcarpetas de audio .ogg por especie.'
    )
    parser.add_argument(
        '--out-dir',
        default='project/data',
        help='Carpeta de salida para features_mfcc_30_especies.csv y embeddings_perch_30_especies.npy.'
    )
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño de lote para Perch.')
    parser.add_argument('--solo-mfcc', action='store_true', help='Ejecutar únicamente la extracción de MFCCs.')
    parser.add_argument('--solo-perch', action='store_true', help='Ejecutar únicamente la extracción de Perch.')
    parser.add_argument('--forzar', action='store_true', help='Sobrescribir archivos de salida existentes.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ruta_mfcc = os.path.join(args.out_dir, 'features_mfcc_30_especies.csv')
    ruta_perch = os.path.join(args.out_dir, 'embeddings_perch_30_especies.npy')

    df_meta = pd.read_csv(args.csv)
    print(f"Metadatos cargados: {len(df_meta):,} audios ({df_meta['primary_label'].nunique()} especies).")

    if not args.solo_perch:
        df_features = extraer_mfccs(df_meta, args.audio_dir, ruta_mfcc, forzar=args.forzar)
    else:
        df_features = pd.read_csv(ruta_mfcc)

    if not args.solo_mfcc:
        extraer_perch(
            df_features, args.audio_dir, ruta_perch,
            batch_size=args.batch_size, forzar=args.forzar
        )


if __name__ == '__main__':
    main()
