"""
Test the raw ECG → beat pipeline using a real MIT-BIH record.
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from ecg_pipeline import process_raw_ecg

# Load a real MIT-BIH record
record = wfdb.rdrecord('data/100')
raw_signal = record.p_signal[:, 0]
fs = record.fs

# Take the first 10 seconds
segment = raw_signal[:int(10 * fs)]

# Process through the pipeline
filtered, r_peaks, beats = process_raw_ecg(segment, fs=fs)

print(f"✅ Raw samples:      {len(segment)}")
print(f"✅ Filtered samples: {len(filtered)}")
print(f"✅ R-peaks detected: {len(r_peaks)}")
print(f"✅ Beats extracted:  {beats.shape}")

# Visualise
fig, axes = plt.subplots(2, 1, figsize=(14, 6))

# Raw vs filtered
axes[0].plot(segment, label='Raw ECG', alpha=0.5)
axes[0].plot(filtered, label='Filtered', linewidth=1.5)
axes[0].scatter(r_peaks, filtered[r_peaks], color='red', s=50, zorder=5, label='R-peaks')
axes[0].set_title('Raw vs Filtered ECG with R-peak Detection')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].grid(True)

# Extracted beats
for i, beat in enumerate(beats[:5]):
    axes[1].plot(beat, alpha=0.7, label=f'Beat {i+1}')
axes[1].set_title('First 5 Extracted Beats (180 samples each)')
axes[1].set_xlabel('Sample within beat')
axes[1].set_ylabel('Amplitude')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()