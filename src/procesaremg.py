import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

FILE = "folded_signal.csv"
FS = 1000  # frecuencia de muestreo aproximada

def bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, f0=50, Q=30):
    nyq = fs / 2
    b, a = iirnotch(f0 / nyq, Q)
    return filtfilt(b, a, signal)

def moving_average(signal, N=20):
    kernel = np.ones(N) / N
    return np.convolve(signal, kernel, mode='same')

def rms(x):
    return np.sqrt(np.mean(x**2))


# Leer CSV

df = pd.read_csv(FILE)

folded_raw = pd.to_numeric(df["folded_raw"], errors="coerce").dropna().values

# ============================================
# CONVERSIÓN A mV (MUY IMPORTANTE)
# ============================================
folded_raw_mV = folded_raw * 5000 / 1023


# Baseline
baseline = np.mean(folded_raw_mV)
centered = folded_raw_mV - baseline

# Filtrado
filtered_bp = bandpass_filter(centered, FS, lowcut=20, highcut=450, order=4)
filtered = notch_filter(filtered_bp, FS, f0=50, Q=30)

# Rectificación
rectified = np.abs(filtered)

# Envolvente
envelope = moving_average(rectified, N=150)


# =========================
# SUAVIZADO + NORMALIZACIÓN + REESCALADO, PARA LA VISUALIZACION PRINCIPALMENTE
# =========================

# 1. Quitar baseline (reposo → 0)
envelope_norm = envelope - np.min(envelope)

# Normalizar
envelope_norm = envelope_norm / np.max(envelope_norm)

# Reescalar a 500 puntos (para visualización más limpia)
n_points = 500
x_old = np.linspace(0, 1, len(envelope_norm))
x_new = np.linspace(0, 1, n_points)

envelope_resampled = np.interp(x_new, x_old, envelope_norm)

# =========================
# GRÁFICA FINAL BONITA, ESTA ES LA NORMALIZADA
# =========================

plt.figure(figsize=(8,4))
plt.plot(envelope_resampled)
plt.title("Envolvente suavizada y reescalada")
plt.xlabel("Muestras reescaladas")
plt.ylabel("Amplitud normalizada")
plt.ylim(0, 1.05)
plt.grid()
plt.show()

# =========================
# GRÁFICA FINAL BONITA, ESTA ES LA QUE LO REPRESENTA EN mV
# =========================


n_points = 500
x_old = np.linspace(0, 1, len(envelope))
x_new = np.linspace(0, 1, n_points)

envelope_resampled_mV = np.interp(x_new, x_old, envelope)

plt.figure(figsize=(8,4))
plt.plot(envelope_resampled_mV)
plt.title("Envolvente suavizada y reescalada en mV")
plt.xlabel("Muestras reescaladas")
plt.ylabel("Amplitud (mV)")
plt.grid()
plt.show()

# =========================
# GUARDAR ENVELOPE resampled EN mV
# =========================

duration = len(envelope) / FS
time_resampled = np.linspace(0, duration, len(envelope_resampled_mV))

df_out = pd.DataFrame({
    "time_s": time_resampled,
    "envelope_resampled_mV": envelope_resampled_mV
})

df_out.to_csv("envelope_resampled_mV.csv", index=False)

print("Envelope reescalado guardado en envelope_resampled_mV.csv")



# ============================================
# COMPARATIVA RAW vs DSP (MISMA ESCALA TEMPORAL)
# ============================================

# Eje temporal
time = np.arange(len(folded_raw_mV)) / FS

plt.figure(figsize=(12,5))

plt.plot(time, centered, label="Raw centered (mV)", alpha=0.6)
plt.plot(time, filtered, label="Filtrada (mV)", alpha=0.7)
plt.plot(time, envelope, label="Envolvente (mV)", linewidth=2.5) #Todas estan ya en mV de antes

plt.title("Comparativa Raw vs DSP")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mV)")
plt.grid()
plt.legend()
plt.show()


# Gráficas
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.plot(folded_raw_mV)
plt.title("Folded Raw (mV)")
plt.ylabel("mV")

plt.subplot(5, 1, 2)
plt.plot(centered)
plt.title("Centered")

plt.subplot(5, 1, 3)
plt.plot(filtered)
plt.title("Filtered")

plt.subplot(5, 1, 4)
plt.plot(rectified)
plt.title("Rectified")

plt.subplot(5, 1, 5)
plt.plot(envelope)
plt.title("Envelope (mV)")
plt.ylabel("mV")

plt.tight_layout()
plt.show()

