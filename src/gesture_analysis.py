import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch

def moving_average(signal, N):
    return np.convolve(signal, np.ones(N)/N, mode='same')

def bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, f0=50, Q=30):
    nyq = fs / 2
    b, a = iirnotch(f0 / nyq, Q)
    return filtfilt(b, a, signal)

FILE = "palma9.46db.csv"
FS = 1000

df = pd.read_csv(FILE, header=None)

time = df.iloc[:, 0].astype(float).values
raw  = df.iloc[:, 1].astype(float).values

# ============================================
# RAW SIGNAL PSD (BEFORE FILTERING)
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
plt.title("Raw signal PSD (before filtering)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 500)
plt.grid()
plt.legend()
plt.show()




raw_notched = notch_filter(raw, FS, f0=50, Q=10)
raw_notched_bandpassed = bandpass_filter(raw_notched, FS, lowcut=20, highcut=450, order=4)

print("Number of samples:", len(raw))
print("FS:", FS)
print("First raw values:", raw[:10])


# ============================================
# SIGNAL PSD AFTER NOTCH FILTER
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
plt.title("Signal PSD after notch filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 500)
plt.grid()
plt.legend()
plt.show()

# ============================================
# FILTERED SIGNAL PSD (TO VALIDATE BOTH NOTCH AND BANDPASS FILTERS)
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
plt.title("Filtered signal PSD (Notch + Bandpass)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.xlim(0, 500)
plt.grid()
plt.legend()
plt.show()

# ============================================
# ROBUST SEGMENTATION (ON RAW SIGNAL)
# ============================================

# RAW signal (use this directly)
signal_for_snr = raw

# Simple rectification (without heavy DSP)
baseline = np.median(raw)
rectified = np.abs(raw - baseline)

# Light smoothing for detection
smooth = moving_average(rectified, N=int(0.05 * FS))

# Robust threshold
threshold = np.mean(smooth) + 0.3 * np.std(smooth)

# Activity mask
active_mask = smooth > threshold

# Clean mask (remove short noise bursts)
mask_smooth = moving_average(active_mask.astype(float), N=int(0.2 * FS))
mask_smooth = mask_smooth > 0.3

# Detect segments
diff_mask = np.diff(mask_smooth.astype(int))
starts = np.where(diff_mask == 1)[0] + 1
ends = np.where(diff_mask == -1)[0] + 1

# Correct boundaries
if mask_smooth[0]:
    starts = np.insert(starts, 0, 0)
if mask_smooth[-1]:
    ends = np.append(ends, len(mask_smooth))

# Filter by minimum duration
min_len = int(0.5 * FS)

segments = []
for s, e in zip(starts, ends):
    if e - s >= min_len:
        segments.append((s, e))

print("Detected segments:", len(segments))



# ============================================
# MERGE NEARBY SEGMENTS
# ============================================

merge_gap = int(0.8 * FS)

merged_segments = []
if len(segments) > 0:
    current_start, current_end = segments[0]

    for s, e in segments[1:]:
        if s - current_end <= merge_gap:
            # merge
            current_end = e
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = s, e

    merged_segments.append((current_start, current_end))

segments = merged_segments

print("\n--- Segments after merging nearby segments ---")
for i, (s, e) in enumerate(segments, 1):
    print(f"{i}: start={s}, end={e}, length={e-s}")


# ============================================
# SELECT THE 5 REAL REPETITIONS
# ============================================

segment_lengths = []
for s, e in segments:
    segment_lengths.append((e - s, s, e))

segment_lengths.sort(reverse=True, key=lambda x: x[0])

top5 = segment_lengths[:5]
top5 = sorted(top5, key=lambda x: x[1])

active_segments_idx = [(s, e) for _, s, e in top5]

print("\n--- Final selected segments ---")
for i, (s, e) in enumerate(active_segments_idx, 1):
    print(f"Seg {i}: start={s}, end={e}, length={e-s}")


trim = int(0.2 * FS)

active_segments_trimmed_idx = []
for s, e in active_segments_idx:
    if e - s > 2 * trim:
        active_segments_trimmed_idx.append((s + trim, e - trim))


# ============================================
# FEATURE EXTRACTION
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

window_size = int(0.2 * FS)
step = int(0.1 * FS)

# -------------------------
# ACTIVATION FEATURES
# -------------------------
for rep_id, (s, e) in enumerate(active_segments_trimmed_idx, 1):
    seg = raw_notched_bandpassed[s:e]
    seg = seg - np.mean(seg)

    for start in range(0, len(seg) - window_size + 1, step):
        window = seg[start:start + window_size]
        window = window - np.mean(window)
        feat = extract_features(window)
        feat["label"] = "palma"
        feat["group"] = f"{FILE}_act_{rep_id}"
        features_list.append(feat)

# -------------------------
# REST INDICES
# -------------------------
rest_segments_idx = []
for i in range(len(active_segments_trimmed_idx) - 1):
    end_act = active_segments_trimmed_idx[i][1]
    start_next = active_segments_trimmed_idx[i + 1][0]

    if start_next > end_act:
        rest_segments_idx.append((end_act, start_next))

# -------------------------
# REST FEATURES
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
# OUTLIER FILTERING FOR NEUTRAL CLASS
# ---------------------------------
df_features = df_features[
    ~((df_features["label"] == "neutral") & (df_features["RMS"] > 5))
]

# -------------------------
# SAVE TO CSV / EXCEL
# -------------------------
df_features.to_csv("palmafeatures.csv", index=False)

print("\n--- EXTRACTED FEATURES ---")
print(df_features)
print("\nCount by class:")
print(df_features["label"].value_counts())
print("\nMean features by class:")
print(df_features.groupby("label")[["RMS", "MAV", "ZC", "VAR", "WL"]].mean())

# FOR SNR CALCULATION

# ============================================
# ACTIVE AND REST SEGMENTS (FOR SNR)
# ============================================

# Active segments on raw signal
act_segments = [raw[s:e] for s, e in active_segments_trimmed_idx]

# Rest segments between activations
rest_segments = []
for i in range(len(active_segments_trimmed_idx) - 1):
    end_act = active_segments_trimmed_idx[i][1]
    start_next = active_segments_trimmed_idx[i + 1][0]

    if start_next > end_act:
        rest_segments.append(raw[end_act:start_next])


epsilon = 1e-12

rms_act_list = [rms(seg - np.mean(seg)) for seg in act_segments if len(seg) > 0]
rms_rest_list = [rms(seg - np.mean(seg)) for seg in rest_segments if len(seg) > 0]

rms_act = np.mean(rms_act_list) if len(rms_act_list) > 0 else 0
rms_rest = np.mean(rms_rest_list) if len(rms_rest_list) > 0 else 0

snr_rms_db = 20 * np.log10((rms_act + epsilon) / (rms_rest + epsilon))

print("---- GLOBAL RMS SNR ----")
print(f"Mean activation RMS: {rms_act:.6f}")
print(f"Mean rest RMS:       {rms_rest:.6f}")
print(f"RMS SNR in dB:       {snr_rms_db:.2f} dB")

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
print(f"Notched activation RMS: {rms_act_notched:.6f}")
print(f"Notched rest RMS:       {rms_rest_notched:.6f}")
print(f"Raw+notch+bandpass SNR in dB:     {snr_notched_db:.2f} dB")


# ============================================
# FOLDING (AVERAGING REPETITIONS) WITH IMPROVED ALIGNMENT; REPETITIONS ARE ALIGNED BY THE ACTIVATION PEAK
# ============================================

window_before = int(3.5 * FS)
window_after  = int(3.5 * FS)
window_len = window_before + window_after

segments_equal = []

for i, (s, e) in enumerate(active_segments_trimmed_idx, 1):
    # Smoothed segment to locate the peak
    smooth_seg = smooth[s:e]

    # Local index of the maximum within the segment
    peak_local = np.argmax(smooth_seg)

    # Global peak index
    peak_global = s + peak_local

    # Window centered on the peak, but taken from the raw signal
    start_win = peak_global - window_before
    end_win   = peak_global + window_after

    # Check that the window fits within the signal
    if start_win >= 0 and end_win <= len(raw):
        seg = raw[start_win:end_win]
        if len(seg) == window_len:
            segments_equal.append(seg)
            print(f"Rep {i}: peak at {peak_global}, window [{start_win}:{end_win}]")
        else:
            print(f"Rep {i}: discarded due to unexpected length")
    else:
        print(f"Rep {i}: discarded for being too close to the boundary")

# Convert to matrix
segments_matrix = np.array(segments_equal)


# Folding
folded_signal = np.mean(segments_matrix, axis=0)
folded_signal_mV = folded_signal * 5000 / 1023

# Remove DC component to make the spectrum easier to interpret
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
plt.title("PSD of the folded raw signal in mV")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (mV²/Hz)")
plt.xlim(0, 500)
plt.grid()
plt.show()


print("Folding completed")
print("Number of repetitions used:", len(segments_equal))

# Plot aligned repetitions and average
plt.figure(figsize=(12,6))
for i in range(segments_matrix.shape[0]):
    plt.plot(segments_matrix[i], alpha=0.5, label=f"Rep {i+1}")
plt.plot(folded_signal, linewidth=3, label="Average")
plt.title("Raw repetitions aligned by activation peak")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

# ============================================
# SAVE THE 5 ALIGNED REPETITIONS
# ============================================
reps_df = pd.DataFrame(
    segments_matrix.T,
    columns=[f"rep_{i+1}" for i in range(segments_matrix.shape[0])]
)
reps_df.to_csv("aligned_reps_raw.csv", index=False)

print("Aligned repetitions saved to aligned_reps_raw.csv")


plt.figure(figsize=(12,4))
plt.plot(folded_signal_mV)
plt.title("Averaged signal (folding) in mV")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.show()

folded_df = pd.DataFrame({"folded_raw": folded_signal})
folded_df.to_csv("folded_signal.csv", index=False)

print("Folding saved to folded_signal.csv")



# To visualize the signal
plt.figure(figsize=(14,5))
plt.plot(rectified, label="rectified", alpha=0.6)
plt.plot(smooth, label="smooth", linewidth=2)
plt.axhline(threshold, linestyle="--", label="threshold")

for s, e in segments:
    plt.axvspan(s, e, alpha=0.25)

plt.title("Activation detection")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()



# To visualize how the segments were selected
plt.figure(figsize=(14,5))
plt.plot(raw, label="raw")

for i, (s, e) in enumerate(active_segments_idx):
    plt.axvspan(s, e, alpha=0.3, label=f"seg {i+1}" if i == 0 else None)

plt.title("Top 5 selected segments")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

print("\n--- Final selected segments ---")
for i, (s, e) in enumerate(active_segments_idx, 1):
    print(f"Seg {i}: start={s}, end={e}, length={e-s}")


# ============================================
# IDEAL SIGNAL FOR BENCHMARKING WITH NINAPRO
# ============================================

# 1. Create filtered + rectified + smoothed signal
filtered_rectified = np.abs(raw_notched_bandpassed)
filtered_envelope = moving_average(filtered_rectified, N=int(0.05 * FS))

# 2. Reuse the same folding parameters already used
window_before_bench = int(2.5 * FS)
window_after_bench  = int(2.5 * FS)
window_len_bench = window_before_bench + window_after_bench

segments_equal_bench = []

for i, (s, e) in enumerate(active_segments_trimmed_idx, 1):
    # Locate the peak using the filtered envelope
    smooth_seg = filtered_envelope[s:e]
    peak_local = np.argmax(smooth_seg)
    peak_global = s + peak_local

    # Window centered on the peak, now taken from the filtered envelope
    start_win = peak_global - window_before_bench
    end_win   = peak_global + window_after_bench

    if start_win >= 0 and end_win <= len(filtered_envelope):
        seg = filtered_envelope[start_win:end_win]
        if len(seg) == window_len_bench:
            segments_equal_bench.append(seg)
            print(f"[BENCH] Rep {i}: peak at {peak_global}, window [{start_win}:{end_win}]")
        else:
            print(f"[BENCH] Rep {i}: discarded due to unexpected length")
    else:
        print(f"[BENCH] Rep {i}: discarded for being too close to the boundary")

# 3. Convert to matrix
segments_matrix_bench = np.array(segments_equal_bench)

# 4. Average repetitions
if len(segments_matrix_bench) > 0:
    folded_benchmark_signal = np.mean(segments_matrix_bench, axis=0)

    # 5. Save final CSV for NinaPro
    benchmark_df = pd.DataFrame({"emg_filtrado": folded_benchmark_signal})
    benchmark_df.to_csv("palma_procesado.csv", index=False)

    print("Ideal signal for NinaPro saved to pulgar_procesado.csv")

    # 6. Also save all aligned repetitions in case you want to inspect them
    reps_bench_df = pd.DataFrame(
        segments_matrix_bench.T,
        columns=[f"rep_{i+1}" for i in range(segments_matrix_bench.shape[0])]
    )
    reps_bench_df.to_csv("palma_reps_benchmark.csv", index=False)

    # 7. Optional visualization
    plt.figure(figsize=(12, 5))
    for i in range(segments_matrix_bench.shape[0]):
        plt.plot(segments_matrix_bench[i], alpha=0.4, label=f"Rep {i+1}" if i == 0 else None)
    plt.plot(folded_benchmark_signal, linewidth=3, label="Benchmark average")
    plt.title("Benchmark signal for NinaPro")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()

else:
    print("The benchmark signal could not be built because there were no valid repetitions.")