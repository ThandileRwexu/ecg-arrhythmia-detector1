"""
ecg_pipeline.py
Raw ECG → Filtered → R-peaks → Beat extraction
Implements a simplified Pan-Tompkins approach.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360.0, order=3):
    """
    Band-pass filter to remove baseline wander (below 0.5 Hz)
    and high-frequency noise (above 40 Hz).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_r_peaks(signal, fs=360.0, min_distance_sec=0.25):
    """
    Detect R-peaks using a simplified Pan-Tompkins approach:
    1. Square the signal to enhance the QRS complex
    2. Apply moving average integration
    3. Find peaks with a minimum distance constraint
    """
    # Square the signal (enhances QRS)
    squared = signal ** 2

    # Moving average integration window (~150 ms)
    window_size = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')

    # Minimum distance between peaks (250 ms → 240 bpm max)
    min_distance = int(min_distance_sec * fs)

    # Find peaks
    peaks, _ = find_peaks(integrated, distance=min_distance, height=np.mean(integrated))

    return peaks


def extract_beats(signal, r_peaks, window_before=90, window_after=90):
    """
    Extract fixed-length beats around each R-peak.
    Returns an array of beats with shape (n_beats, window_before + window_after).
    """
    beats = []
    for peak in r_peaks:
        start = peak - window_before
        end = peak + window_after
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        if len(beat) == window_before + window_after:
            beats.append(beat)
    return np.array(beats)


def process_raw_ecg(raw_signal, fs=360.0):
    """
    End-to-end pipeline: raw ECG → filtered → R-peaks → beats.
    Returns (filtered_signal, r_peaks, beats).
    """
    filtered = bandpass_filter(raw_signal, fs=fs)
    r_peaks = detect_r_peaks(filtered, fs=fs)
    beats = extract_beats(filtered, r_peaks)
    return filtered, r_peaks, beats