import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import resample
from scipy.stats import pearsonr
from scipy.signal import resample, butter, filtfilt, iirnotch
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# ==============================
# FUNCIONES DE PREPROCESADO
# ==============================

def normalize_01(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)

def moving_average(signal, N):
    return np.convolve(signal, np.ones(N)/N, mode='same')

def bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, f0=50, Q=10):
    nyq = fs / 2
    b, a = iirnotch(f0 / nyq, Q)
    return filtfilt(b, a, signal)

def compute_rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def benchmark_gesto(nombre_gesto, gesture_id, mi_signal, emg, stimulus, restimulus, repetition, FS, idx_plot):
    # =========================
    # Buscar mejor canal sin folding
    # =========================
    signal_gesto = emg[stimulus == gesture_id]

    best_r = -1
    best_channel = None
    best_ninapro_norm = None

    mi_norm = normalize_01(mi_signal)

    for ch in range(signal_gesto.shape[1]):
        ninapro_raw = signal_gesto[:, ch]

        ninapro_notched = notch_filter(ninapro_raw, FS, f0=50, Q=10)
        ninapro_bandpassed = bandpass_filter(ninapro_notched, FS, lowcut=20, highcut=450, order=4)
        ninapro_rectified = np.abs(ninapro_bandpassed)
        ninapro_envelope = moving_average(ninapro_rectified, N=int(0.05 * FS))

        ninapro_resampled = resample(ninapro_envelope, len(mi_signal))
        ninapro_norm = normalize_01(ninapro_resampled)

        r_temp, _ = pearsonr(mi_norm, ninapro_norm)

        if r_temp > best_r:
            best_r = r_temp
            best_channel = ch
            best_ninapro_norm = ninapro_norm

    r = best_r
    ninapro_norm = best_ninapro_norm
    p_value = pearsonr(mi_norm, ninapro_norm)[1]
    rmse = compute_rmse(mi_norm, ninapro_norm)

    print(f"\n===== BENCHMARKING {nombre_gesto.upper()} =====")
    print("Mejor canal NinaPro:", best_channel)
    print("Mejor Pearson encontrado:", best_r)

    # =========================
    # Folding NinaPro
    # =========================
    rep_ids = np.unique(repetition[(restimulus == gesture_id) & (repetition > 0)])
    print("Repeticiones encontradas para el gesto:", rep_ids)

    segments_equal_nina = []

    for rep_id in rep_ids:
        mask_rep = (restimulus == gesture_id) & (repetition == rep_id)
        seg_raw = emg[mask_rep, best_channel]

        print(f"Rep {rep_id}: longitud original = {len(seg_raw)}")

        if len(seg_raw) < 50:
            print(f"Rep {rep_id}: descartada por ser demasiado corta")
            continue

        seg_notched = notch_filter(seg_raw, FS, f0=50, Q=10)
        seg_bandpassed = bandpass_filter(seg_notched, FS, lowcut=20, highcut=450, order=4)
        seg_rectified = np.abs(seg_bandpassed)
        seg_envelope = moving_average(seg_rectified, N=int(0.05 * FS))

        peak_idx = np.argmax(seg_envelope)

        left_len = peak_idx
        right_len = len(seg_envelope) - peak_idx - 1
        half_window = min(left_len, right_len)

        if half_window < 30:
            print(f"Rep {rep_id}: descartada por ventana útil demasiado pequeña")
            continue

        start_win = peak_idx - half_window
        end_win = peak_idx + half_window + 1
        window_seg = seg_envelope[start_win:end_win]

        print(f"Rep {rep_id}: peak={peak_idx}, start={start_win}, end={end_win}, len_window={len(window_seg)}")
        segments_equal_nina.append(window_seg)

    if len(segments_equal_nina) > 0:
        target_len = max(len(seg) for seg in segments_equal_nina)

        segments_resampled_nina = []
        for seg in segments_equal_nina:
            seg_rs = resample(seg, target_len)
            segments_resampled_nina.append(seg_rs)

        segments_matrix_nina = np.array(segments_resampled_nina)

        folded_ninapro = np.mean(segments_matrix_nina, axis=0)
        folded_ninapro_resampled = resample(folded_ninapro, len(mi_signal))

        ninapro_folded_norm = normalize_01(folded_ninapro_resampled)

        r_folded, p_folded = pearsonr(mi_norm, ninapro_folded_norm)
        rmse_folded = compute_rmse(mi_norm, ninapro_folded_norm)

        print(f"\n--- PEARSON CON NINAPRO FOLDED ({nombre_gesto.upper()}) ---")
        print("Pearson r =", r_folded)
        print("p-value =", p_folded)
        print("RMSE =", rmse_folded)

        # gráfica folded
        plt.figure(figsize=(12, 5))
        #plt.subplot(3,1,idx_plot)
        plt.plot(mi_norm, label=f"Mi señal de {nombre_gesto}")
        plt.plot(ninapro_folded_norm, label=f"NinaPro {nombre_gesto} folded")

        plt.ylim(0,1)
        plt.title(f"Comparación morfológica: Mi {nombre_gesto} vs NinaPro")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud normalizada")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("No se pudieron extraer repeticiones válidas en NinaPro.")

    print(f"\n--- CORRELACIÓN PEARSON CON NINAPRO ({nombre_gesto.upper()}) SIN FOLDING ---")
    print("Pearson r =", r)
    print("p-value =", p_value)
    print("RMSE =", rmse)

    # gráfica sin folding
    plt.figure(figsize=(12, 5))
    plt.plot(mi_norm, label=f"Mi señal de {nombre_gesto}")
    plt.plot(ninapro_norm, label=f"NinaPro {nombre_gesto} sin folding")
    plt.title(f"Comparación morfológica sin folding: Mi {nombre_gesto} vs NinaPro")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud normalizada")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================
# 1. Cargar los CSV de features
# ==============================

df_palma = pd.read_csv("palmafeatures.csv") #O el nombre que tengan
df_puno = pd.read_csv("punofeatures.csv")
df_pulgar = pd.read_csv("pulgarfeatures.csv")

# =========================================================
# 2. CARGAR TU SEÑAL TEMPORAL DE PUÑO PARA COMPARAR CON NINAPRO, A PARTIR DE AQUI PARA COMPARAR CON NINAPRO
# =========================================================


mi_pulgar_df = pd.read_csv("pulgar_procesado.csv")

mi_pulgar = mi_pulgar_df["emg_filtrado"].values

mi_puno_df = pd.read_csv("puno_procesado.csv")
mi_puno = mi_puno_df["emg_filtrado"].values

mi_palma_df = pd.read_csv("palma_procesado.csv")
mi_palma = mi_palma_df["emg_filtrado"].values

FS = 1000

# =========================================================
# 3. CARGAR ARCHIVO NINAPRO
# =========================================================


data = loadmat("S10_A1_E1.mat") #Dentro del sujeto 10 que cargo aqui, vienen tanto puño, como pulgar como palma

# Variables típicas de NinaPro
emg = data["emg"]
stimulus = data["stimulus"].flatten()

restimulus = data["restimulus"].flatten()
repetition = data["repetition"].flatten()

benchmark_gesto("pulgar", 10, mi_pulgar, emg, stimulus, restimulus, repetition, FS, 1)
benchmark_gesto("puño", 1, mi_puno, emg, stimulus, restimulus, repetition, FS, 2)
benchmark_gesto("palma", 2, mi_palma, emg, stimulus, restimulus, repetition, FS, 3)



# ==============================
# 2. Unir todos los dataframes
# ==============================

df = pd.concat([df_palma, df_puno, df_pulgar], ignore_index=True)

# ==============================
# 3. Separar variables predictoras (X) y etiquetas (y)
# ==============================

X = df[["RMS", "MAV", "ZC", "VAR", "WL"]]
y = df["label"]
groups = df["group"]

print("\n--- DISTRIBUCIÓN DE CLASES ---")
print(y.value_counts())

# ==============================
# Mostrar información
# ==============================

print("--- DATASET UNIFICADO ---")
print(df.head())

print("\nNúmero total de filas:", len(df))
print("Dimensión:", df.shape)

# ==============================
# Guardar archivo unificado
# ==============================

df.to_csv("datasetunificado.csv", index=False) #Para comprobar que se guarde

# ==============================
# 5. Crear modelo
# ==============================

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==============================
# 6. Configurar cross-validation
# ==============================

cv = GroupKFold(n_splits=5)

# ==============================
# 7. Predicción cruzada
# ==============================

y_pred = cross_val_predict(
    modelo,
    X,
    y,
    cv=cv,
    groups=groups
)

# ==============================
# 8. Accuracy global
# ==============================

acc = accuracy_score(y, y_pred)

print("\n--- ACCURACY GLOBAL ---")
print(acc)

# ==============================
# 9. Matriz de confusión
# ==============================

cm = confusion_matrix(y, y_pred)

print("\n--- MATRIZ DE CONFUSIÓN ---")
print(cm)

# ==============================
# 10. Informe clasificación
# ==============================

print("\n--- INFORME DE CLASIFICACIÓN ---")
print(classification_report(y, y_pred))

# ==============================
# MÉTRICAS POR CLASE Y MACRO-PRECISION
# ==============================


labels = sorted(y.unique())

precision_por_clase = precision_score(
    y,
    y_pred,
    labels=labels,
    average=None,
    zero_division=0
)

recall_por_clase = recall_score(
    y,
    y_pred,
    labels=labels,
    average=None,
    zero_division=0
)

f1_por_clase = f1_score(
    y,
    y_pred,
    labels=labels,
    average=None,
    zero_division=0
)

print("\n--- MÉTRICAS POR CLASE ---")
for label, p, r, f1 in zip(labels, precision_por_clase, recall_por_clase, f1_por_clase):
    print(f"{label}: precision={p:.4f}, recall={r:.4f}, f1-score={f1:.4f}")

macro_precision = precision_score(
    y,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n--- PROMEDIOS MACRO ---")
print(f"Macro-precision: {macro_precision:.4f}")
print(f"Macro-recall:    {macro_recall:.4f}")
print(f"Macro-F1:        {macro_f1:.4f}")

# ==============================
# IMPORTANCIA DE FEATURES
# ==============================

modelo.fit(X, y)

importancias = modelo.feature_importances_

print("\n--- IMPORTANCIA DE FEATURES ---")

for nombre, valor in zip(X.columns, importancias):
    print(f"{nombre}: {valor:.4f}")

# ==============================
# 11. Dibujar matriz de confusión
# ==============================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=modelo.classes_ if hasattr(modelo, "classes_") else sorted(y.unique())
)

disp.plot()
plt.title("Matriz de Confusión - EMG Gesture Classification")
plt.show()