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
# PREPROCESSING FUNCTIONS
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
    # Find best channel without folding
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
    print("Best NinaPro channel:", best_channel)
    print("Best Pearson found:", best_r)

    # =========================
    # NinaPro folding
    # =========================
    rep_ids = np.unique(repetition[(restimulus == gesture_id) & (repetition > 0)])
    print("Repetitions found for the gesture:", rep_ids)

    segments_equal_nina = []

    for rep_id in rep_ids:
        mask_rep = (restimulus == gesture_id) & (repetition == rep_id)
        seg_raw = emg[mask_rep, best_channel]

        print(f"Rep {rep_id}: original length = {len(seg_raw)}")

        if len(seg_raw) < 50:
            print(f"Rep {rep_id}: discarded for being too short")
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
            print(f"Rep {rep_id}: discarded for having too small a usable window")
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

        print(f"\n--- PEARSON WITH NINAPRO FOLDED ({nombre_gesto.upper()}) ---")
        print("Pearson r =", r_folded)
        print("p-value =", p_folded)
        print("RMSE =", rmse_folded)

        # folded plot
        plt.figure(figsize=(12, 5))
        #plt.subplot(3,1,idx_plot)
        plt.plot(mi_norm, label=f"My {nombre_gesto} signal")
        plt.plot(ninapro_folded_norm, label=f"NinaPro {nombre_gesto} folded")

        plt.ylim(0,1)
        plt.title(f"Morphological comparison: My {nombre_gesto} vs NinaPro")
        plt.xlabel("Samples")
        plt.ylabel("Normalized amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("Could not extract valid repetitions from NinaPro.")

    print(f"\n--- PEARSON CORRELATION WITH NINAPRO ({nombre_gesto.upper()}) WITHOUT FOLDING ---")
    print("Pearson r =", r)
    print("p-value =", p_value)
    print("RMSE =", rmse)

    # plot without folding
    plt.figure(figsize=(12, 5))
    plt.plot(mi_norm, label=f"My {nombre_gesto} signal")
    plt.plot(ninapro_norm, label=f"NinaPro {nombre_gesto} without folding")
    plt.title(f"Morphological comparison without folding: My {nombre_gesto} vs NinaPro")
    plt.xlabel("Samples")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================
# 1. Load feature CSVs
# ==============================

df_palma = pd.read_csv("palmafeatures.csv")  # or whatever name they have
df_puno = pd.read_csv("punofeatures.csv")
df_pulgar = pd.read_csv("pulgarfeatures.csv")

# =========================================================
# 2. LOAD YOUR TEMPORAL FIST SIGNAL TO COMPARE WITH NINAPRO, FROM HERE ON IT'S FOR COMPARING WITH NINAPRO
# =========================================================


mi_pulgar_df = pd.read_csv("pulgar_procesado.csv")

mi_pulgar = mi_pulgar_df["emg_filtrado"].values

mi_puno_df = pd.read_csv("puno_procesado.csv")
mi_puno = mi_puno_df["emg_filtrado"].values

mi_palma_df = pd.read_csv("palma_procesado.csv")
mi_palma = mi_palma_df["emg_filtrado"].values

FS = 1000

# =========================================================
# 3. LOAD NINAPRO FILE
# =========================================================


data = loadmat("S10_A1_E1.mat")  # subject 10 loaded here includes fist, thumb and palm gestures

# Typical NinaPro variables
emg = data["emg"]
stimulus = data["stimulus"].flatten()

restimulus = data["restimulus"].flatten()
repetition = data["repetition"].flatten()

benchmark_gesto("thumb", 10, mi_pulgar, emg, stimulus, restimulus, repetition, FS, 1)
benchmark_gesto("fist", 1, mi_puno, emg, stimulus, restimulus, repetition, FS, 2)
benchmark_gesto("palm", 2, mi_palma, emg, stimulus, restimulus, repetition, FS, 3)



# ==============================
# 2. Merge all dataframes
# ==============================

df = pd.concat([df_palma, df_puno, df_pulgar], ignore_index=True)

# ==============================
# 3. Split predictor variables (X) and labels (y)
# ==============================

X = df[["RMS", "MAV", "ZC", "VAR", "WL"]]
y = df["label"]
groups = df["group"]

print("\n--- CLASS DISTRIBUTION ---")
print(y.value_counts())

# ==============================
# Show information
# ==============================

print("--- UNIFIED DATASET ---")
print(df.head())

print("\nTotal number of rows:", len(df))
print("Shape:", df.shape)

# ==============================
# Save unified file
# ==============================

df.to_csv("datasetunificado.csv", index=False)  # to check that it saved correctly

# ==============================
# 5. Create model
# ==============================

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==============================
# 6. Configure cross-validation
# ==============================

cv = GroupKFold(n_splits=5)

# ==============================
# 7. Cross-validated prediction
# ==============================

y_pred = cross_val_predict(
    modelo,
    X,
    y,
    cv=cv,
    groups=groups
)

# ==============================
# 8. Overall accuracy
# ==============================

acc = accuracy_score(y, y_pred)

print("\n--- OVERALL ACCURACY ---")
print(acc)

# ==============================
# 9. Confusion matrix
# ==============================

cm = confusion_matrix(y, y_pred)

print("\n--- CONFUSION MATRIX ---")
print(cm)

# ==============================
# 10. Classification report
# ==============================

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y, y_pred))

# ==============================
# PER-CLASS METRICS AND MACRO-PRECISION
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

print("\n--- PER-CLASS METRICS ---")
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

print("\n--- MACRO AVERAGES ---")
print(f"Macro-precision: {macro_precision:.4f}")
print(f"Macro-recall:    {macro_recall:.4f}")
print(f"Macro-F1:        {macro_f1:.4f}")

# ==============================
# FEATURE IMPORTANCE
# ==============================

modelo.fit(X, y)

importancias = modelo.feature_importances_

print("\n--- FEATURE IMPORTANCE ---")

for nombre, valor in zip(X.columns, importancias):
    print(f"{nombre}: {valor:.4f}")

# ==============================
# 11. Plot confusion matrix
# ==============================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=modelo.classes_ if hasattr(modelo, "classes_") else sorted(y.unique())
)

disp.plot()
plt.title("Confusion Matrix - EMG Gesture Classification")
plt.show()