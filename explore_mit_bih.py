import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a folder for the data if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download record 100 from MIT-BIH
print("📥 Downloading MIT-BIH record 100...")
wfdb.dl_database('mitdb', 'data/')

# Load the record
record = wfdb.rdrecord('data/100')
print(f"✅ Record loaded: {record.record_name}")
print(f"   Sampling frequency: {record.fs} Hz")
print(f"   Number of samples: {len(record.p_signal)}")
print(f"   Number of leads: {record.n_sig}")

# Get the first lead ECG signal
ecg_signal = record.p_signal[:, 0]
fs = record.fs

# Plot the first 5 seconds
time = np.arange(len(ecg_signal)) / fs
plt.figure(figsize=(12, 4))
plt.plot(time[:5*fs], ecg_signal[:5*fs], linewidth=1)
plt.title('MIT-BIH Record 100 — First 5 Seconds')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print basic stats
print(f"\n📊 Signal Statistics:")
print(f"   Mean: {np.mean(ecg_signal):.4f}")
print(f"   Std:  {np.std(ecg_signal):.4f}")
print(f"   Min:  {np.min(ecg_signal):.4f}")
print(f"   Max:  {np.max(ecg_signal):.4f}")