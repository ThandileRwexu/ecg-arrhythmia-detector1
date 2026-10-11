# Edge-Based Machine Learning for Real-Time Arrhythmia Detection
## Beta Version – IPJ527C

**Author:** Thandile Rwexu  
**Student Number:** 220015322  
**Course:** IPJ527C — Advanced Research Project B  
**Submission:** Beta Demonstration (Assessment 3)  
**Date:** October 2026

---

## 🔗 Quick Access

| Resource | Link |
|----------|------|
| **Source Code** | https://github.com/ThandileRwexu/ecg-arrhythmia-detector1 |
| **Beta Video** | Submitted on eThuto |

---

## 📌 Project Summary

This project develops a lightweight, edge-based machine learning system for real-time detection of abnormal heart rhythms (arrhythmias) from electrocardiogram (ECG) signals. The system uses a 1D Convolutional Neural Network trained on the MIT-BIH Arrhythmia Database, achieving **95% accuracy** on 3-class classification (Normal, SVEB, VEB). All inference runs locally on the edge device — no cloud dependency.

---

## ✅ Implemented Functions

### 1. Machine Learning Models

| Model | Purpose | Accuracy |
|-------|---------|----------|
| **1D-CNN (Beta)** | Primary model — 3-class AAMI classification | 95% |
| **Random Forest (Alpha)** | Baseline comparison model | 86% |

### 2. Input Workflows

| # | Workflow | Description |
|---|----------|-------------|
| 1 | Manual Entry (10 Values) | Enter 10 ECG amplitude values → interpolated to 180 samples → classified |
| 2 | CSV Upload | Upload real MIT-BIH CSV files (180 columns per beat) → classified in batch |
| 3 | Simulated Real-Time | Replays 20 real MIT-BIH test beats from the held-out DS2 set |
| 4 | Live Phone Stream | HTTP server on port 8765 receives authenticated ECG data from a remote client |

### 3. Signal Processing

- **Raw ECG → Beat Pipeline** (`ecg_pipeline.py`): band-pass filter, R-peak detection, beat extraction.
- **RR-Interval Features**: improved accuracy from 89% to 95%.
- **Signal Quality Check**: rejects flat, saturated, or noisy signals.

### 4. Classification & Alerting

- 3-Class Detection: Normal (N), Supraventricular (SVEB), Ventricular (VEB).
- Raw softmax probabilities (no clipping).
- "Uncertain" state when confidence < 60%.
- Rhythm-Aware Alerts: HIGH (3+ VEBs), MEDIUM (≥30% burden), or none.

### 5. Security Features

- Token Authentication (`X-Auth-Token` header).
- Rate Limiting (100 requests / minute).
- Payload Size Validation (20 KB max).
- Payload Format Validation (JSON + numeric checks).

---

## 🖥️ How to Run the System

### Prerequisites

- **Python 3.12**
- **VS Code** or any code editor
- **A modern web browser** (Chrome recommended)
- **Windows, Mac, or Linux**

### Step 1: Clone the Repository

```bash
git clone https://github.com/ThandileRwexu/ecg-arrhythmia-detector1.git
cd ecg-arrhythmia-detector1
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

You should see `(venv)` at the start of the terminal line.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- streamlit
- numpy
- pandas
- scikit-learn
- joblib
- plotly
- requests
- scipy
- tensorflow

### Step 4: Run the Streamlit App

In Terminal 1:

```bash
python -m streamlit run app.py
```

You should see:

```text
✅ HTTP server started on port 8765
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 5: Open the Dashboard

Open your browser and go to:

```text
http://localhost:8501
```

---

## 📱 How to Run the Phone Simulator

The phone simulator (`simulate_phone.py`) acts as a remote client that sends authenticated ECG data to the local server. It mimics the behaviour of a wearable device or smartphone.

### Step 1: Get the API Token

1. Open the Streamlit dashboard in your browser.
2. Look at the left sidebar → **🔐 API Token** panel.
3. Copy the token (e.g., `qBa7WAN8GnGFucf0A4-r2g`).

### Step 2: Open a Second Terminal

1. In VS Code, split the terminal (click the split icon next to the `+`).
2. Activate the virtual environment:

```bash
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### Step 3: Run the Simulator

```bash
python simulate_phone.py YOUR_TOKEN_HERE
```

Example:

```bash
python simulate_phone.py qBa7WAN8GnGFucf0A4-r2g
```

### Step 4: Watch the Output

You should see:

```text
📱 Simulating phone → http://localhost:8765/ecg
🔐 Token: qBa7WA...  Press Ctrl+C to stop.

📦 Packet 1 | HTTP 200 | {"prediction": "Normal (N)", "confidence": 0.95}
📦 Packet 2 | HTTP 200 | {"prediction": "Supraventricular (SVEB)", "confidence": 0.79}
📦 Packet 3 | HTTP 200 | ...
```

### Step 5: Check the Dashboard

The Streamlit dashboard updates in real time:

| Panel | Expected |
|-------|----------|
| Phone Connection | ✅ Receiving data — 20 packet(s) |
| Signal Quality | ✅ Good |
| ECG Visualization | Live waveform |
| Detection Result | Class + confidence |
| Detection History Log | Timestamped entries |
| Active Rhythm Alerts | 🔴 or 🟠 alerts (if patterns detected) |
