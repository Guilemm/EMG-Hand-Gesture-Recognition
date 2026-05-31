import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import butter, filtfilt, iirnotch, welch

def moving_average(signal, N):
    return np.convolve(signal, np.ones(N)/N, mode='same')

#Estos filtros estan a la espera de ser validados por el teacher

def bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, f0=50, Q=30):#Q=10 es considerablemente agesrivo
    nyq = fs / 2
    b, a = iirnotch(f0 / nyq, Q)
    return filtfilt(b, a, signal)

FILE = "palma9.46db.csv" #La del pulagr usada fue pulgar9.78db y la del puño fue puno10.32db y palma la de palma9.46db
FS = 1000  # frecuencia de muestreo fija del Arduino

df = pd.read_csv(FILE, header=None)

time = df.iloc[:, 0].astype(float).values
raw  = df.iloc[:, 1].astype(float).values

# ============================================
# PSD SEÑAL RAW (ANTES DEL FILTRADO)
# ============================================

raw_centered = raw - np.mean(raw)

freqs_raw, psd_raw = welch(
    raw_centered,
    fs=FS,
    window='hann',
    nperseg=min(1024, len(raw_centered)),
    noverlap=min(512, len(raw_centered) // 2),
    detrend='constant'
)

plt.figure(figsize=(10,5))
plt.semilogy(freqs_raw, psd_raw)
plt.axvline(50, linestyle="--", label="50 Hz")
plt.title("PSD señal RAW (antes del filtrado)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 500)
plt.grid()
plt.legend()
plt.show()




raw_notched = notch_filter(raw, FS, f0=50, Q=10)
raw_notched_bandpassed = bandpass_filter(raw_notched, FS, lowcut=20, highcut=450, order=4)

print("Número de muestras:", len(raw))
print("FS:", FS)
print("Primeros valores raw:", raw[:10])


# ============================================
# PSD SEÑAL TRAS NOTCH
# ============================================

notched_centered = raw_notched - np.mean(raw_notched)

freqs_notch, psd_notch = welch(
    notched_centered,
    fs=FS,
    window='hann',
    nperseg=min(1024, len(notched_centered)),
    noverlap=min(512, len(notched_centered) // 2),
    detrend='constant'
)

plt.figure(figsize=(10,5))
plt.semilogy(freqs_notch, psd_notch)
plt.axvline(50, linestyle="--", label="50 Hz")
plt.title("PSD señal tras filtro Notch")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 500)
plt.grid()
plt.legend()
plt.show()

# ============================================
# PSD SEÑAL FILTRADA (PARA VALIDAR FILTRO) TANTO NOTCH COMO BANDPASS
# ============================================

filtered_centered = raw_notched_bandpassed - np.mean(raw_notched_bandpassed)

freqs_filt, psd_filt = welch(
    filtered_centered,
    fs=FS,
    window='hann',
    nperseg=min(1024, len(filtered_centered)),
    noverlap=min(512, len(filtered_centered)//2),
    detrend='constant'
)

plt.figure(figsize=(10,5))
plt.semilogy(freqs_filt, psd_filt)
plt.axvline(50, linestyle="--", label="50 Hz")
plt.title("PSD señal filtrada (Notch + Bandpass)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 500)
plt.grid()
plt.legend()
plt.show()

# ============================================
# SEGMENTACIÓN ROBUSTA (SOBRE RAW)
# ============================================

# Señal RAW (usa esta directamente)
signal_for_snr = raw

# Rectificación simple (sin DSP fuerte)
baseline = np.median(raw)
rectified = np.abs(raw - baseline)

# Suavizado ligero para detección
smooth = moving_average(rectified, N=int(0.05 * FS))  # 50 ms

# Umbral robusto
threshold = np.mean(smooth) + 0.3 * np.std(smooth) #Aqui con el 0.3 se ajusta cuánto por encima del ruido necesitas estar

# Máscara de actividad
active_mask = smooth > threshold #Esto hace que la mascara bianria tenga 0 y uno segun activacion o reposo en terminos del umbral, pero de esta manera hay ceros y unos mexclados, no hay consistencia temporal, por eso luego se aplica media movil de 200 ms sobre esa ventana binaria

# Limpiar máscara (eliminar ruido corto)
mask_smooth = moving_average(active_mask.astype(float), N=int(0.2 * FS)) #Aqui aplico el suavizado con 200 ms, DISTINTO DEL DE 50 MS, la ventana de 200 ms hace: solo considero activación si hay suficiente actividad sostenida en 200 ms (o en los ultimos al estqar centrada)
mask_smooth = mask_smooth > 0.3 #CON EL ASTYPE(FLOAT) CONVIERTO LA MASCARA BINARIA 0 Y UNOS EN 0.0, 0.0, 1.0..., Y TRAS LA MEDIA MOVIL OBTENGO [0.1, 0.2, 0.6, 0.8, 0.7, 0.5, 0.4, 0.2, 0.1, ...]
#Lo de mask_smooth es para reconstruir la mascara bianria, pero sin fragmentacion y eliminando detecciones espurias
#Usar 0.3 en  mask_smooth = mask_smooth > 0.3, hace que lo que en la nueva señal con valores flotantes por encima de 0.3, sea considerado un uno en la mascara binaria reconstruida
#binaria -> continua -> binaria regularizada

# Detectar segmentos
diff_mask = np.diff(mask_smooth.astype(int))
starts = np.where(diff_mask == 1)[0] + 1
ends = np.where(diff_mask == -1)[0] + 1

# Corregir bordes
if mask_smooth[0]:
    starts = np.insert(starts, 0, 0)
if mask_smooth[-1]:
    ends = np.append(ends, len(mask_smooth))

# Filtrar por duración mínima
min_len = int(0.5 * FS)

segments = []
for s, e in zip(starts, ends):
    if e - s >= min_len:
        segments.append((s, e))

print("Segmentos detectados:", len(segments))



# ============================================
# FUSIONAR SEGMENTOS CERCANOS
# ============================================

merge_gap = int(0.8 * FS)  # fusionar si el hueco es menor de 800 muestras (al estar con una frecuencia de muiesyreo de 1000 Hz)

merged_segments = []
if len(segments) > 0:
    current_start, current_end = segments[0]

    for s, e in segments[1:]:
        if s - current_end <= merge_gap:
            # fusionar
            current_end = e
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = s, e

    merged_segments.append((current_start, current_end))

segments = merged_segments

print("\n--- Segmentos tras fusionar cercanos ---")
for i, (s, e) in enumerate(segments, 1):
    print(f"{i}: inicio={s}, fin={e}, longitud={e-s}")


# ============================================
# SELECCIONAR LAS 5 REPETICIONES REALES
# ============================================

segment_lengths = []
for s, e in segments:
    segment_lengths.append((e - s, s, e))

segment_lengths.sort(reverse=True, key=lambda x: x[0])

top5 = segment_lengths[:5]
top5 = sorted(top5, key=lambda x: x[1])

active_segments_idx = [(s, e) for _, s, e in top5]

print("\n--- Segmentos finales elegidos ---")
for i, (s, e) in enumerate(active_segments_idx, 1):
    print(f"Seg {i}: inicio={s}, fin={e}, longitud={e-s}")


trim = int(0.2 * FS)

active_segments_trimmed_idx = []
for s, e in active_segments_idx:
    if e - s > 2 * trim:
        active_segments_trimmed_idx.append((s + trim, e - trim))


# ============================================
# EXTRACCIÓN DE FEATURES
# ============================================

def rms(x):
    return np.sqrt(np.mean(x**2))

def mav(x):
    return np.mean(np.abs(x))

def zero_crossings(x):
    return np.sum(np.diff(np.sign(x)) != 0)

def variance(x):
    return np.var(x)

def waveform_length(x):
    return np.sum(np.abs(np.diff(x)))

def extract_features(segment):
    x = np.asarray(segment)
    return {
        "RMS": rms(x),
        "MAV": mav(x),
        "ZC": zero_crossings(x),
        "VAR": variance(x),
        "WL": waveform_length(x),
    }

features_list = []

window_size = int(0.2 * FS)   # 200 ms
step = int(0.1 * FS)          # 50% overlap (son 100 muestras con frecuencia de muestreo de 1000 Hz)

# -------------------------
# FEATURES DE ACTIVACIÓN
# -------------------------
for rep_id, (s, e) in enumerate(active_segments_trimmed_idx, 1):
    seg = raw_notched_bandpassed[s:e]
    seg = seg - np.mean(seg)

    for start in range(0, len(seg) - window_size + 1, step):
        window = seg[start:start + window_size]
        window = window - np.mean(window)
        feat = extract_features(window)
        feat["label"] = "palma"   # CAMBIA esto según el archivo: palma / pulgar / puno
        feat["group"] = f"{FILE}_act_{rep_id}"
        features_list.append(feat)

# -------------------------
# ÍNDICES DE REPOSO
# -------------------------
rest_segments_idx = []
for i in range(len(active_segments_trimmed_idx) - 1):
    end_act = active_segments_trimmed_idx[i][1]
    start_next = active_segments_trimmed_idx[i + 1][0]

    if start_next > end_act:
        rest_segments_idx.append((end_act, start_next))

# -------------------------
# FEATURES DE REPOSO
# -------------------------
for rest_id, (s, e) in enumerate(rest_segments_idx, 1):
    seg = raw_notched_bandpassed[s:e]
    seg = seg - np.mean(seg)

    for start in range(0, len(seg) - window_size + 1, step):
        window = seg[start:start + window_size]
        window = window - np.mean(window)
        feat = extract_features(window)
        feat["label"] = "neutral"
        feat["group"] = f"{FILE}_rest_{rest_id}"
        features_list.append(feat)

df_features = pd.DataFrame(features_list)

# ---------------------------------
# FILTRADO DE OUTLIERS EN NEUTRAL
# ---------------------------------
df_features = df_features[
    ~((df_features["label"] == "neutral") & (df_features["RMS"] > 5))
]

# -------------------------
# GUARDAR EN CSV / EXCEL
# -------------------------
df_features.to_csv("palmafeatures.csv", index=False) #CAMBIAR ESTO SEGUN EL QUE GUARDE

print("\n--- FEATURES EXTRAÍDAS ---")
print(df_features)
print("\nConteo por clase:")
print(df_features["label"].value_counts())
print("\nMedia de features por clase:")
print(df_features.groupby("label")[["RMS", "MAV", "ZC", "VAR", "WL"]].mean())

#PARA CALCULO DE SNR

# ============================================
# SEGMENTOS ACTIVOS Y DE REPOSO (PARA SNR)
# ============================================

# Segmentos activos sobre raw
act_segments = [raw[s:e] for s, e in active_segments_trimmed_idx]

# Segmentos de reposo entre activaciones
rest_segments = []
for i in range(len(active_segments_trimmed_idx) - 1):
    end_act = active_segments_trimmed_idx[i][1]
    start_next = active_segments_trimmed_idx[i + 1][0]

    if start_next > end_act:
        rest_segments.append(raw[end_act:start_next])


def rms(x):
    return np.sqrt(np.mean(np.square(x)))

epsilon = 1e-12

rms_act_list = [rms(seg - np.mean(seg)) for seg in act_segments if len(seg) > 0]
rms_rest_list = [rms(seg - np.mean(seg)) for seg in rest_segments if len(seg) > 0]

rms_act = np.mean(rms_act_list) if len(rms_act_list) > 0 else 0
rms_rest = np.mean(rms_rest_list) if len(rms_rest_list) > 0 else 0

snr_rms_db = 20 * np.log10((rms_act + epsilon) / (rms_rest + epsilon))

print("---- SNR RMS global ----")
print(f"RMS activación media: {rms_act:.6f}")
print(f"RMS reposo media:     {rms_rest:.6f}")
print(f"SNR RMS en dB:        {snr_rms_db:.2f} dB")

# -------------------------
# SNR RAW + NOTCH + BANDPASS
# -------------------------
act_segments_notched = [raw_notched_bandpassed[s:e] for s, e in active_segments_trimmed_idx]

rest_segments_notched = []
for i in range(len(active_segments_trimmed_idx) - 1):
    end_act = active_segments_trimmed_idx[i][1]
    start_next = active_segments_trimmed_idx[i + 1][0]
    if start_next > end_act:
        rest_segments_notched.append(raw_notched_bandpassed[end_act:start_next])

rms_act_notched_list = [rms(seg - np.mean(seg)) for seg in act_segments_notched if len(seg) > 0]
rms_rest_notched_list = [rms(seg - np.mean(seg)) for seg in rest_segments_notched if len(seg) > 0]

rms_act_notched = np.mean(rms_act_notched_list) if len(rms_act_notched_list) > 0 else 0
rms_rest_notched = np.mean(rms_rest_notched_list) if len(rms_rest_notched_list) > 0 else 0

snr_notched_db = 20 * np.log10((rms_act_notched + epsilon) / (rms_rest_notched + epsilon))

print("\n---- SNR RMS RAW + NOTCH + BANDPASS ----")
print(f"RMS activación notched: {rms_act_notched:.6f}")
print(f"RMS reposo notched:     {rms_rest_notched:.6f}")
print(f"SNR raw+notch+bandpass en dB:    {snr_notched_db:.2f} dB")


# ============================================
# FOLDING (PROMEDIO DE REPETICIONES) CON MEJOR ALINEADO, AQUI ALINEO REPETICIONES POR EL PICO DE ACTIVACION
# ============================================

# Usamos smooth para encontrar el pico de activación dentro de cada segmento,
# pero extraemos la ventana de la señal raw, para que el folding siga siendo sobre raw.

window_before = int(3.5 * FS)   # los ms antes del pico CUANDO GRABE UNA BIEN CON ANCHURA AL FINAL, AUMENTAR EL 2.75 A 3.5 !!!!
window_after  = int(3.5 * FS)   # los ms después del pico esto es ajustable para ir cogiendo lo de antes, que quede como bonito (Con las zonas previas y posteriores de relajacion)
window_len = window_before + window_after

segments_equal = []

for i, (s, e) in enumerate(active_segments_trimmed_idx, 1):
    # Segmento suavizado para localizar el pico
    smooth_seg = smooth[s:e]

    # Índice local del máximo dentro del segmento
    peak_local = np.argmax(smooth_seg)

    # Índice global del pico
    peak_global = s + peak_local

    # Ventana centrada en el pico, pero tomada de la raw
    start_win = peak_global - window_before
    end_win   = peak_global + window_after

    # Comprobar que la ventana cabe dentro de la señal
    if start_win >= 0 and end_win <= len(raw):
        seg = raw[start_win:end_win]
        if len(seg) == window_len:
            segments_equal.append(seg)
            print(f"Rep {i}: pico en {peak_global}, ventana [{start_win}:{end_win}]")
        else:
            print(f"Rep {i}: descartada por longitud inesperada")
    else:
        print(f"Rep {i}: descartada por estar demasiado cerca del borde")

# Convertir a matriz
segments_matrix = np.array(segments_equal)


# Folding
folded_signal = np.mean(segments_matrix, axis=0)
folded_signal_mV = folded_signal * 5000 / 1023

# Quitar componente DC para que el espectro sea más interpretable. ESTO ES LA PARTE PARA CALCULAR EL PSD (Power density spectrum)
folded_signal_mV_centered = folded_signal_mV - np.mean(folded_signal_mV)

freqs, psd = welch(
    folded_signal_mV_centered,
    fs=FS,
    window='hann',
    nperseg=min(1024, len(folded_signal_mV_centered)),
    noverlap=min(512, len(folded_signal_mV_centered) // 2),
    detrend='constant'
)

plt.figure(figsize=(10,5))
plt.semilogy(freqs, psd)
plt.title("PSD de la señal folded raw en mV")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (mV²/Hz)")
plt.xlim(0, 500)
plt.grid()
plt.show()


print("Folding completado")
print("Número de repeticiones usadas:", len(segments_equal))

# Graficar repeticiones alineadas y promedio
plt.figure(figsize=(12,6))
for i in range(segments_matrix.shape[0]):
    plt.plot(segments_matrix[i], alpha=0.5, label=f"Rep {i+1}")
plt.plot(folded_signal, linewidth=3, label="Promedio")
plt.title("Repeticiones raw alineadas por pico de activación")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()
plt.legend()
plt.show()

# ============================================
# GUARDAR LAS 5 REPETICIONES ALINEADAS
# ============================================
reps_df = pd.DataFrame(
    segments_matrix.T,
    columns=[f"rep_{i+1}" for i in range(segments_matrix.shape[0])]
)
reps_df.to_csv("aligned_reps_raw.csv", index=False)

print("Repeticiones alineadas guardadas en aligned_reps_raw.csv")

#Esto es solom para visualizarla en mV, pero no guardarla, la que guardo no esta en mV

plt.figure(figsize=(12,4))
plt.plot(folded_signal_mV)
plt.title("Señal promediada (folding) en mV")
plt.xlabel("Muestras")
plt.ylabel("Amplitud (mV)")
plt.grid()
plt.show()

folded_df = pd.DataFrame({"folded_raw": folded_signal})
folded_df.to_csv("folded_signal.csv", index=False)

print("Folding guardado en folded_signal.csv")



#Para ver la señal?
plt.figure(figsize=(14,5))
plt.plot(rectified, label="rectified", alpha=0.6)
plt.plot(smooth, label="smooth", linewidth=2)
plt.axhline(threshold, linestyle="--", label="threshold")

for s, e in segments:
    plt.axvspan(s, e, alpha=0.25)

plt.title("Detección de activaciones")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()
plt.legend()
plt.show()



#Para ver como ha seleccionado los segmentos
plt.figure(figsize=(14,5))
plt.plot(raw, label="raw")

for i, (s, e) in enumerate(active_segments_idx):
    plt.axvspan(s, e, alpha=0.3, label=f"seg {i+1}" if i == 0 else None)

plt.title("Top 5 segmentos seleccionados")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()
plt.legend()
plt.show()

print("\n--- Segmentos finales elegidos ---")
for i, (s, e) in enumerate(active_segments_idx, 1):
    print(f"Seg {i}: inicio={s}, fin={e}, longitud={e-s}")


# ============================================
# SEÑAL IDEAL PARA BENCHMARKING CON NINAPRO
# ============================================

# 1. Crear señal filtrada + rectificada + suavizada
filtered_rectified = np.abs(raw_notched_bandpassed)
filtered_envelope = moving_average(filtered_rectified, N=int(0.05 * FS))

# 2. Reutilizar los mismos parámetros del folding que ya usabas
window_before_bench = int(2.5 * FS) #LO PONGO DISTINTO A COMO YO LO TENIA, SOLAMENTE PARA LA COMPARACION CON NINAPRO, NO PARA REALIZAR YO LA SEGMENTACION, DECIR QUE ESPERIMENTALMENTE YO HICE TRES SEGUNDOS DE DESCANSO, PERO NINAPRO HACE CINCO
window_after_bench  = int(2.5 * FS) #Originalmente lo tenia a 3.5, el pearson ha bajado al recortar informacion de la señal, al disminuir el tamaño de la ventana, sin embargo ahora sí es comparable con Ninapro, que tambien usa ventanas de 5 segundos para sus señales
window_len_bench = window_before_bench + window_after_bench

segments_equal_bench = []

for i, (s, e) in enumerate(active_segments_trimmed_idx, 1):
    # localizar el pico usando la envolvente filtrada, ASI LO HAGO TAMBIEN EN EL DE NINAPRO
    smooth_seg = filtered_envelope[s:e]
    peak_local = np.argmax(smooth_seg)
    peak_global = s + peak_local

    # ventana centrada en el pico, pero ahora tomada de la envolvente filtrada
    start_win = peak_global - window_before_bench
    end_win   = peak_global + window_after_bench

    if start_win >= 0 and end_win <= len(filtered_envelope):
        seg = filtered_envelope[start_win:end_win]
        if len(seg) == window_len_bench:
            segments_equal_bench.append(seg)
            print(f"[BENCH] Rep {i}: pico en {peak_global}, ventana [{start_win}:{end_win}]")
        else:
            print(f"[BENCH] Rep {i}: descartada por longitud inesperada")
    else:
        print(f"[BENCH] Rep {i}: descartada por estar demasiado cerca del borde")

# 3. Convertir a matriz
segments_matrix_bench = np.array(segments_equal_bench)

# 4. Promedio de repeticiones
if len(segments_matrix_bench) > 0:
    folded_benchmark_signal = np.mean(segments_matrix_bench, axis=0)

    # 5. Guardar CSV final para NinaPro
    benchmark_df = pd.DataFrame({"emg_filtrado": folded_benchmark_signal})
    benchmark_df.to_csv("palma_procesado.csv", index=False) #CAMBIAR ESTO SEGUN LO QUE QUIERA GUARDAR

    print("Señal ideal para NinaPro guardada en pulgar_procesado.csv")

    # 6. Guardar también todas las repeticiones alineadas, por si quieres inspeccionarlas
    reps_bench_df = pd.DataFrame(
        segments_matrix_bench.T,
        columns=[f"rep_{i+1}" for i in range(segments_matrix_bench.shape[0])]
    )
    reps_bench_df.to_csv("pulgar_reps_benchmark.csv", index=False)

    print("Repeticiones benchmark guardadas en puno_reps_benchmark.csv") #esto pone puno, pero sera lo que tenga que ser en cada momento

    # 7. Visualización opcional
    plt.figure(figsize=(12, 5))
    for i in range(segments_matrix_bench.shape[0]):
        plt.plot(segments_matrix_bench[i], alpha=0.4, label=f"Rep {i+1}" if i == 0 else None)
    plt.plot(folded_benchmark_signal, linewidth=3, label="Promedio benchmark")
    plt.title("Señal benchmark para NinaPro")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.legend()
    plt.show()

else:
    print("No se pudo construir la señal benchmark porque no hubo repeticiones válidas.")