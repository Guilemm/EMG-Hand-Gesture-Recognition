import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

FILE = "folded_signal.csv"
FS = 1000  # approximate sampling frequency

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


# Read CSV

df = pd.read_csv(FILE)

folded_raw = pd.to_numeric(df["folded_raw"], errors="coerce").dropna().values

# ============================================
# CONVERT TO mV (IMPORTANT)
# ============================================
folded_raw_mV = folded_raw * 5000 / 1023


# Baseline
baseline = np.mean(folded_raw_mV)
centered = folded_raw_mV - baseline

# Filtering
filtered_bp = bandpass_filter(centered, FS, lowcut=20, highcut=450, order=4)
filtered = notch_filter(filtered_bp, FS, f0=50, Q=30)

# Rectification
rectified = np.abs(filtered)

# Envelope
envelope = moving_average(rectified, N=150)


# =========================
# SMOOTHING + NORMALIZATION + RESCALING, MAINLY FOR VISUALIZATION
# =========================

# 1. Remove baseline (rest -> 0)
envelope_norm = envelope - np.min(envelope)

# Normalize
envelope_norm = envelope_norm / np.max(envelope_norm)

# Rescale to 500 points (for cleaner visualization)
n_points = 500
x_old = np.linspace(0, 1, len(envelope_norm))
x_new = np.linspace(0, 1, n_points)

envelope_resampled = np.interp(x_new, x_old, envelope_norm)

# =========================
# FINAL PLOT, NORMALIZED VERSION
# =========================

plt.figure(figsize=(8,4))
plt.plot(envelope_resampled)
plt.title("Smoothed and rescaled envelope")
plt.xlabel("Rescaled samples")
plt.ylabel("Normalized amplitude")
plt.ylim(0, 1.05)
plt.grid()
plt.show()

# =========================
# FINAL PLOT, mV VERSION
# =========================


n_points = 500
x_old = np.linspace(0, 1, len(envelope))
x_new = np.linspace(0, 1, n_points)

envelope_resampled_mV = np.interp(x_new, x_old, envelope)

plt.figure(figsize=(8,4))
plt.plot(envelope_resampled_mV)
plt.title("Smoothed and rescaled envelope in mV")
plt.xlabel("Rescaled samples")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.show()

# =========================
# SAVE RESAMPLED ENVELOPE IN mV
# =========================

duration = len(envelope) / FS
time_resampled = np.linspace(0, duration, len(envelope_resampled_mV))

df_out = pd.DataFrame({
    "time_s": time_resampled,
    "envelope_resampled_mV": envelope_resampled_mV
})

df_out.to_csv("envelope_resampled_mV.csv", index=False)

print("Resampled envelope saved to envelope_resampled_mV.csv")



# ============================================
# RAW vs DSP COMPARISON (SAME TIME SCALE)
# ============================================

# Time axis
time = np.arange(len(folded_raw_mV)) / FS

plt.figure(figsize=(12,5))

plt.plot(time, centered, label="Raw centered (mV)", alpha=0.6)
plt.plot(time, filtered, label="Filtered (mV)", alpha=0.7)
plt.plot(time, envelope, label="Envelope (mV)", linewidth=2.5)

plt.title("Raw vs DSP comparison")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.legend()
plt.show()


# Plots
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
