import os
import json
import hmac
import time
import secrets
import threading
import http.server
import socketserver
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# set_page_config must be called exactly once, and first
st.set_page_config(page_title="Edge ECG Arrhythmia Detector", layout="wide")

# ============================================================
# CONFIG
# ============================================================
HTTP_PORT = 8765
MAX_PAYLOAD_BYTES = 20_000
RATE_LIMIT_PER_MIN = 100
TFLITE_PATH = "ecg_cnn_focal_int8.tflite"
RF_MODEL_PATH = "ecg_model.pkl"
RF_SCALER_PATH = "scaler.pkl"


# ============================================================
# SHARED SERVER STATE
# One object, shared by every browser session AND the HTTP thread.
# (st.session_state is per-browser-tab and is NOT visible from the
#  HTTP thread, which is why the token/metrics were broken before.)
# ============================================================
@st.cache_resource
def get_server_state():
    return {
        "lock": threading.Lock(),          # protects the dict below
        "infer_lock": threading.Lock(),    # serialises model inference
        # Token is created ONCE per server process. Set ECG_API_TOKEN
        # in the environment if you want it to survive restarts.
        "token": os.environ.get("ECG_API_TOKEN") or secrets.token_urlsafe(16),
        "request_times": [],
        "latest_ecg": None,
        "latest_result": None,
        "packet_count": 0,
        "use_cnn": True,
        # live edge metrics
        "last_inference_ms": 0.0,
        "inference_times": [],
        "last_quality": ("Not yet assessed", "info"),
        # rhythm history (last 20 beats)
        "rhythm_history": [],
    }


SERVER = get_server_state()

# ============================================================
# SESSION STATE (UI-only things)
# ============================================================
if "detection_log" not in st.session_state:
    st.session_state["detection_log"] = []


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_tflite_model():
    from predict_tflite import TFLitePredictor
    return TFLitePredictor(TFLITE_PATH)


@st.cache_resource
def load_rf_model():
    try:
        return joblib.load(RF_MODEL_PATH), joblib.load(RF_SCALER_PATH)
    except Exception:
        return None, None


try:
    predictor = load_tflite_model()
    model_loaded = True
    USE_TFLITE = True
except Exception as e:
    st.error(f"⚠️ TFLite model not found: {e}")
    predictor = None
    model_loaded = False
    USE_TFLITE = False

rf_model, rf_scaler = load_rf_model()


# ============================================================
# SIGNAL QUALITY
# ============================================================
def assess_signal_quality(beat):
    """Return (is_good, reason)."""
    beat = np.array(beat, dtype=np.float32).flatten()

    amplitude = np.max(beat) - np.min(beat)
    if amplitude < 0.05:
        return False, "Signal is flat — check electrode contact."
    if amplitude > 8.0:
        return False, "Signal is saturated — electrode may be loose."

    if np.var(beat) < 0.005:
        return False, "Very low signal variance — possible electrode issue."

    hf_energy = np.mean(np.diff(beat) ** 2)
    total_energy = np.mean(beat ** 2) + 1e-8
    if hf_energy / total_energy > 0.6:
        return False, "Excessive high-frequency noise — movement artifacts likely."

    return True, "Signal quality OK."


# ============================================================
# INFERENCE (pure: no st.* calls, safe to call from the HTTP thread)
# ============================================================
def run_inference(features_array, use_cnn):
    start = time.perf_counter()
    beat = np.asarray(features_array, dtype=np.float32).flatten()

    is_good, reason = assess_signal_quality(beat)
    if not is_good:
        return {
            "pred": 0, "probs": np.array([0.5, 0.5]),
            "class_name": f"Bad signal — {reason}", "confidence": 0.0,
            "all_probs": [0, 0, 0], "quality": (reason, "bad"),
            "ms": (time.perf_counter() - start) * 1000,
        }

    all_probs = None
    with SERVER["infer_lock"]:
        if use_cnn and USE_TFLITE:
            class_name, confidence, all_probs = predictor.predict(beat)
            prediction = 0 if "Normal" in class_name else 1
            probs = (np.array([1 - confidence, confidence]) if prediction == 1
                     else np.array([confidence, 1 - confidence]))
        elif rf_model is not None:
            beat_10 = np.interp(np.linspace(0, 1, 10), np.linspace(0, 1, 180), beat)
            scaled = rf_scaler.transform([beat_10])
            prediction = int(rf_model.predict(scaled)[0])
            proba = rf_model.predict_proba(scaled)[0]
            confidence = float(proba[prediction])
            if confidence < 0.60:
                class_name = "Uncertain — needs review"
            else:
                class_name = "Normal (N)" if prediction == 0 else "Abnormal"
            probs = np.array([1 - confidence, confidence])
            all_probs = [float(1 - confidence), float(confidence)]
        else:
            return {
                "pred": 1, "probs": np.array([0.5, 0.5]), "class_name": "Error",
                "confidence": 0.5, "all_probs": None,
                "quality": ("No model loaded", "bad"),
                "ms": (time.perf_counter() - start) * 1000,
            }

    return {
        "pred": prediction, "probs": probs, "class_name": class_name,
        "confidence": float(confidence), "all_probs": all_probs,
        "quality": ("Good", "good"),
        "ms": (time.perf_counter() - start) * 1000,
    }


def record_metrics(ms, quality):
    with SERVER["lock"]:
        SERVER["last_inference_ms"] = ms
        SERVER["inference_times"].append(ms)
        del SERVER["inference_times"][:-200]   # rolling window
        SERVER["last_quality"] = quality


def predict_heartbeat(features_array, use_cnn=None):
    """UI-side wrapper: same 5-tuple as before, but metrics go to shared state."""
    if use_cnn is None:
        use_cnn = SERVER["use_cnn"]
    r = run_inference(features_array, use_cnn)
    record_metrics(r["ms"], r["quality"])
    return r["pred"], r["probs"], r["class_name"], r["confidence"], r["all_probs"]


# ============================================================
# RHYTHM ALERTS (shared history so phone + UI both feed it)
# ============================================================
def longest_veb_run(history):
    run = best = 0
    for v in history:
        run = run + 1 if v == 2 else 0
        best = max(best, run)
    return best


def abnormal_burden(history):
    valid = [v for v in history if v >= 0]
    if not valid:
        return 0.0, 0
    return sum(1 for v in valid if v in (1, 2)) / len(valid), len(valid)


def check_rhythm_alerts(class_name):
    if "Normal" in class_name:
        code = 0
    elif "SVEB" in class_name or "Supraventricular" in class_name:
        code = 1
    elif "VEB" in class_name or "Ventricular" in class_name:
        code = 2
    else:
        code = -1  # uncertain / bad signal

    with SERVER["lock"]:
        SERVER["rhythm_history"].append(code)
        del SERVER["rhythm_history"][:-20]
        history = list(SERVER["rhythm_history"])

    alerts = []
    max_run = longest_veb_run(history)
    if max_run >= 3:
        alerts.append(f"🔴 HIGH: {max_run} consecutive VEB beats — possible Ventricular Tachycardia")
    burden, n_valid = abnormal_burden(history)
    if n_valid >= 10 and burden >= 0.30:
        alerts.append(f"🟠 MEDIUM: High abnormal burden — {burden*100:.0f}% of last {n_valid} beats")
    return alerts


def log_detection(class_name, confidence):
    st.session_state["detection_log"].append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Result": class_name,
        "Confidence": f"{confidence:.2f}",
        "Alert": "⚠️" if "Normal" not in class_name else "✅",
    })
    st.session_state["detection_log"] = st.session_state["detection_log"][-20:]


def show_result(slot, class_name, confidence, pred):
    if "Bad signal" in class_name:
        slot.error(f"📵 {class_name}")
    elif "Uncertain" in class_name:
        slot.warning(f"❓ {class_name} | {confidence*100:.1f}%")
    elif pred == 0:
        slot.success(f"✅ {class_name} | {confidence*100:.1f}%")
    else:
        slot.error(f"⚠️ {class_name} | {confidence*100:.1f}%")


# ============================================================
# HTTP SERVER (phone → laptop)
# ============================================================
class ECGDataHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Auth-Token")

    def _send(self, code, body=None):
        payload = json.dumps(body if body is not None else {}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self._send(200, {})

    def do_POST(self):
        try:
            if self.path != "/ecg":
                return self._send(404, {"error": "Not found"})

            # 1. Authentication (constant-time compare, live token from shared state)
            token = self.headers.get("X-Auth-Token", "")
            with SERVER["lock"]:
                expected = SERVER["token"]
            if not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
                print("❌ Rejected: invalid or missing token")
                return self._send(401, {"error": "Unauthorised"})

            # 2. Payload size
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            if length <= 0:
                return self._send(400, {"error": "Empty body"})
            if length > MAX_PAYLOAD_BYTES:
                print("❌ Rejected: payload too large")
                return self._send(413, {"error": "Payload too large"})

            # 3. Rate limiting
            now = time.time()
            with SERVER["lock"]:
                recent = [t for t in SERVER["request_times"] if now - t < 60]
                limited = len(recent) >= RATE_LIMIT_PER_MIN
                if not limited:
                    recent.append(now)
                SERVER["request_times"] = recent
            if limited:
                print("❌ Rejected: rate limit exceeded")
                return self._send(429, {"error": "Rate limit exceeded"})

            # 4. Payload validation
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            ecg = data.get("ecg") if isinstance(data, dict) else None
            if not isinstance(ecg, list) or not (10 <= len(ecg) <= 1000):
                return self._send(400, {"error": "Invalid ECG data (need 10–1000 numbers)"})
            arr = np.asarray(ecg, dtype=np.float64)
            if arr.ndim != 1 or not np.all(np.isfinite(arr)):
                return self._send(400, {"error": "ECG values must be finite numbers"})

            # Use all samples: 180 are used as-is, anything else is resampled to 180.
            # (The old code kept only the first 10 samples of a 180-sample beat.)
            if len(arr) == 180:
                beat = arr
            else:
                beat = np.interp(np.linspace(0, 1, 180), np.linspace(0, 1, len(arr)), arr)

            # 5. Classification (no st.* calls in here)
            r = run_inference(beat, SERVER["use_cnn"])
            record_metrics(r["ms"], r["quality"])
            check_rhythm_alerts(r["class_name"])

            bad_or_unsure = "Bad signal" in r["class_name"] or "Uncertain" in r["class_name"]
            result = {
                "prediction": r["class_name"],
                "confidence": float(r["confidence"]),
                "pred": int(r["pred"]),
                "abnormal": bool(r["pred"] == 1 and not bad_or_unsure),
                "inference_ms": round(r["ms"], 3),
            }
            with SERVER["lock"]:
                SERVER["latest_ecg"] = [float(x) for x in beat]
                SERVER["latest_result"] = result
                SERVER["packet_count"] += 1

            print(f"📥 Authenticated packet ({len(arr)} samples) → {result}")
            self._send(200, result)

        except (ValueError, TypeError, json.JSONDecodeError):
            self._send(400, {"error": "Invalid payload"})
        except Exception as e:
            print(f"Error handling POST: {e}")
            self._send(500, {"error": "Server error"})


class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_http_server():
    try:
        with ReusableTCPServer(("0.0.0.0", HTTP_PORT), ECGDataHandler) as httpd:
            print(f"✅ HTTP server started on port {HTTP_PORT}")
            httpd.serve_forever()
    except OSError as e:
        print(f"❌ Could not start HTTP server: {e}")


@st.cache_resource
def get_server_thread():
    t = threading.Thread(target=start_http_server, daemon=True)
    t.start()
    return t


get_server_thread()


def get_snapshot():
    with SERVER["lock"]:
        return {
            "latest_ecg": list(SERVER["latest_ecg"]) if SERVER["latest_ecg"] else None,
            "latest_result": dict(SERVER["latest_result"]) if SERVER["latest_result"] else None,
            "packet_count": SERVER["packet_count"],
        }


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### 🧠 Active Model")
model_choice = st.sidebar.radio(
    "Choose the model for classification:",
    ["1D-CNN (Beta) — 89% accuracy", "Random Forest (Alpha) — 86% accuracy"],
)
use_cnn = "CNN" in model_choice
SERVER["use_cnn"] = use_cnn   # phone requests use the model chosen here

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Active Rhythm Alerts")
with SERVER["lock"]:
    history = list(SERVER["rhythm_history"])
if history:
    max_run = longest_veb_run(history)
    burden, _ = abnormal_burden(history)
    if max_run >= 3:
        st.sidebar.error(f"🔴 {max_run} consecutive VEBs")
    elif burden >= 0.30:
        st.sidebar.warning(f"🟠 Burden: {burden*100:.0f}%")
    else:
        st.sidebar.success("✅ Rhythm stable")
else:
    st.sidebar.info("No rhythm data yet.")

st.sidebar.markdown("### 📶 Signal Quality")
with SERVER["lock"]:
    q_label, q_status = SERVER["last_quality"]
if q_status == "good":
    st.sidebar.success(f"✅ {q_label}")
elif q_status == "uncertain":
    st.sidebar.warning(f"❓ {q_label}")
elif q_status == "bad":
    st.sidebar.error(f"📵 {q_label}")
else:
    st.sidebar.info(q_label)

st.sidebar.markdown("### 🔐 API Token")
st.sidebar.code(SERVER["token"], language=None)
st.sidebar.caption("Send this as the `X-Auth-Token` header. It stays the same across page "
                   "refreshes; it only changes if you restart the app or click below.")
if st.sidebar.button("♻️ Regenerate token"):
    with SERVER["lock"]:
        SERVER["token"] = secrets.token_urlsafe(16)
    st.rerun()

snap = get_snapshot()
st.sidebar.markdown("### 📱 Phone Connection")
if snap["packet_count"] > 0:
    st.sidebar.success(f"✅ Receiving data — {snap['packet_count']} packet(s)")
else:
    st.sidebar.warning("⚠️ No packets received yet from phone")
st.sidebar.caption(f"HTTP server listening on port {HTTP_PORT}")

if st.sidebar.button("🔄 Refresh live data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📊 Input Method")
input_method = st.sidebar.radio(
    "Choose how to provide ECG data:",
    ["📱 Live Phone Stream", "Simulated Real-Time", "Enter 10 ECG Values", "Upload CSV File"],
)

# ============================================================
# MAIN UI + LIVE EDGE METRICS
# ============================================================
st.title("🫀 Edge-Based Real-Time Abnormal Heart Rhythm Detection")
st.markdown(f"**Active Model:** {'1D-CNN (Beta)' if use_cnn else 'Random Forest (Alpha)'}")
st.markdown("*Prototype for Edge ML Deployment*")


def file_kb(*paths):
    try:
        return f"{sum(os.path.getsize(p) for p in paths) / 1024:.0f} KB"
    except OSError:
        return "n/a"


with st.expander("📊 Live Edge Deployment Metrics", expanded=True):
    metrics_slot = st.empty()


def render_metrics():
    """Redrawable: call it again after any inference to refresh the numbers."""
    with SERVER["lock"]:
        last = SERVER["last_inference_ms"]
        times = list(SERVER["inference_times"])
    with metrics_slot.container():
        c = st.columns(5)
        if SERVER["use_cnn"]:
            c[0].metric("Model", "1D-CNN (INT8)")
            c[1].metric("Size", file_kb(TFLITE_PATH))
            c[4].metric("Classes", "3 (N/SVEB/VEB)")
        else:
            c[0].metric("Model", "Random Forest")
            c[1].metric("Size", file_kb(RF_MODEL_PATH, RF_SCALER_PATH))
            c[4].metric("Classes", "2 (Normal/Abnormal)")
        c[2].metric("Last inference", f"{last:.2f} ms" if times else "—")
        c[3].metric(f"Avg (last {len(times)})" if times else "Avg", 
                    f"{np.mean(times):.2f} ms" if times else "—")
        extra = f" · p95 {np.percentile(times, 95):.2f} ms" if times else ""
        st.caption(f"{len(times)} inference(s) recorded{extra}. "
                   "All inference runs locally on the edge device — no cloud dependency.")


render_metrics()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📈 ECG Signal Visualization")
    ecg_placeholder = st.empty()
with col2:
    st.subheader("🩺 Detection Result")
    result_placeholder = st.empty()

# ============================================================
# LIVE PHONE STREAM
# ============================================================
if input_method == "📱 Live Phone Stream":
    st.info("🔴 Showing the most recent ECG packet received from your phone. "
            "Click 'Refresh live data' in the sidebar, or tick auto-refresh below.")
    auto_refresh = st.checkbox("Auto-refresh every 1s", value=False)

    snap = get_snapshot()
    ecg, res = snap["latest_ecg"], snap["latest_result"]

    if ecg is not None and res is not None:
        fig = go.Figure()
        pred_abnormal = res.get("prediction") == "Abnormal"
        fig.add_trace(go.Scatter(y=ecg, mode="lines",
                                 line=dict(color="red" if pred_abnormal else "green")))
        fig.update_layout(height=300, title=f"Latest packet ({snap['packet_count']} total received)")
        ecg_placeholder.plotly_chart(fig, use_container_width=True)
        if res.get("prediction") == "Normal":
            result_placeholder.success(f"✅ NORMAL | Confidence: {res['confidence']:.2f}")
        elif res.get("prediction") == "Abnormal":
            result_placeholder.error(f"⚠️ ABNORMAL | Confidence: {res['confidence']:.2f}")
        else:
            result_placeholder.info(f"Status: {res.get('prediction')}")

        if st.session_state.get("last_logged_packet") != snap["packet_count"]:
            log_detection(res["prediction"], res["confidence"])
            st.session_state["last_logged_packet"] = snap["packet_count"]
    else:
        ecg_placeholder.info("Waiting for the first packet from your phone...")
        result_placeholder.info("No data received yet")

    if auto_refresh:
        time.sleep(1)
        st.rerun()

# ============================================================
# SIMULATED REAL-TIME
# ============================================================
elif input_method == "Simulated Real-Time":
    st.info("🔴 Replaying real MIT-BIH test beats from the held-out test set (DS2).")

    if st.button("▶️ Start Real-Time Detection"):
        try:
            real_beats = np.load("real_test_beats.npy")
            real_labels = np.load("real_test_labels.npy")
            label_names = ["Normal (N)", "Supraventricular (SVEB)", "Ventricular (VEB)"]
        except FileNotFoundError:
            st.error("⚠️ real_test_beats.npy not found. Please download it from Colab.")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        log_placeholder = st.empty()
        log_entries = []

        for i in range(len(real_beats)):
            features = real_beats[i]
            true_label = label_names[int(real_labels[i])]

            pred, prob, class_name, confidence, all_probs = predict_heartbeat(features)
            log_detection(class_name, confidence)
            for a in check_rhythm_alerts(class_name):
                st.warning(a)
            render_metrics()   # live numbers update on every beat

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=features, mode="lines+markers",
                                     line=dict(color="red" if pred == 1 else "green")))
            fig.update_layout(height=300, title=f"Beat {i+1} — True label: {true_label}")
            ecg_placeholder.plotly_chart(fig, use_container_width=True)

            if "Bad signal" in class_name:
                log_entries.append(f"📵 Beat {i+1}: {class_name}")
            elif "Uncertain" in class_name:
                log_entries.append(f"❓ Beat {i+1}: {class_name} ({confidence:.2f})")
            elif pred == 0:
                log_entries.append(f"✅ Beat {i+1}: {class_name} ({confidence:.2f})")
            else:
                log_entries.append(f"⚠️ Beat {i+1}: {class_name} ({confidence:.2f})")
            log_placeholder.markdown("\n\n".join(log_entries))

            progress_bar.progress((i + 1) / len(real_beats))
            status_text.text(f"Analyzing real beat {i + 1}/{len(real_beats)}...")
            time.sleep(0.5)

        status_text.text("✅ Simulation complete!")
        st.success(f"🎯 All {len(real_beats)} real MIT-BIH beats processed successfully!")

# ============================================================
# MANUAL ENTRY
# ============================================================
elif input_method == "Enter 10 ECG Values":
    st.write("Enter 10 ECG amplitude values:")
    cols = st.columns(10)
    ecg_values = [col.number_input(f"P{i+1}", value=0.5, step=0.1, format="%.2f")
                  for i, col in enumerate(cols)]

    if st.button("🔍 Detect Arrhythmia"):
        if model_loaded or rf_model is not None:
            ecg_180 = np.interp(np.linspace(0, 1, 180), np.linspace(0, 1, 10), ecg_values)
            pred, prob, class_name, confidence, all_probs = predict_heartbeat(ecg_180)
            log_detection(class_name, confidence)
            for a in check_rhythm_alerts(class_name):
                st.warning(a)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=ecg_values, mode="lines+markers",
                                     line=dict(color="red" if pred == 1 else "green")))
            ecg_placeholder.plotly_chart(fig, use_container_width=True)
            show_result(result_placeholder, class_name, confidence, pred)
            render_metrics()
        else:
            st.error("No model is loaded.")

# ============================================================
# CSV UPLOAD
# ============================================================
else:
    st.write("Upload a CSV file with real MIT-BIH ECG data (180 columns per beat).")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded! Rows: {len(df)}, Columns: {len(df.columns)}")

        if df.shape[1] >= 180:
            st.info(f"📊 Real MIT-BIH data detected: {len(df)} beats × 180 samples")

            if st.button("Analyze All Heartbeats"):
                results = []
                progress_bar = st.progress(0)
                for n, (idx, row) in enumerate(df.iterrows(), start=1):
                    try:
                        features = row.values[:180].astype(float)
                        pred, prob, class_name, confidence, all_probs = predict_heartbeat(features)
                        log_detection(class_name, confidence)
                        check_rhythm_alerts(class_name)
                        results.append(f"{class_name} ({confidence:.2f})")
                    except Exception as e:
                        results.append(f"Error: {e}")
                    progress_bar.progress(n / len(df))
                render_metrics()

                df["Detection"] = results
                st.dataframe(df)

                # "SVEB" contains "VEB", so test SVEB first / exclude it from VEB
                normal_count = sum(1 for r in results if "Normal" in r)
                sveb_count = sum(1 for r in results if "SVEB" in r or "Supraventricular" in r)
                veb_count = sum(1 for r in results
                                if ("VEB" in r or "Ventricular" in r)
                                and "SVEB" not in r and "Supraventricular" not in r)
                c1, c2, c3 = st.columns(3)
                c1.metric("Normal", normal_count)
                c2.metric("SVEB", sveb_count)
                c3.metric("VEB", veb_count)

                st.download_button("📥 Download Results as CSV",
                                   data=df.to_csv(index=False).encode("utf-8"),
                                   file_name="ecg_detection_results.csv", mime="text/csv")

                # ---- Rhythm summary ----
                st.markdown("---")
                st.subheader("🫀 Rhythm Analysis Summary")
                with SERVER["lock"]:
                    hist = list(SERVER["rhythm_history"])
                max_run = longest_veb_run(hist)
                v = sum(1 for x in hist if x == 2)
                s = sum(1 for x in hist if x == 1)
                c1, c2, c3 = st.columns(3)
                c1.metric("VEB Beats (last 20)", v)
                c2.metric("SVEB Beats (last 20)", s)
                c3.metric("Longest VEB Run", max_run)
                if max_run >= 3:
                    st.error(f"🔴 HIGH ALERT: {max_run} consecutive VEBs — potential Ventricular Tachycardia")
                elif v / max(len(hist), 1) >= 0.30:
                    st.warning(f"🟠 MEDIUM ALERT: High VEB burden — {v}/{len(hist)} beats")
                else:
                    st.success("✅ No rhythm-level abnormalities detected.")

        elif df.shape[1] >= 10:
            st.warning("⚠️ 10-column CSV detected. For Beta, please use the real MIT-BIH CSV (180 columns).")
            if st.button("Analyze (Legacy 10-column)"):
                results = []
                for idx, row in df.iterrows():
                    f10 = row.values[:10].astype(float)
                    f180 = np.interp(np.linspace(0, 1, 180), np.linspace(0, 1, 10), f10)
                    _, _, class_name, confidence, _ = predict_heartbeat(f180)
                    results.append(f"{class_name} ({confidence:.2f})")
                render_metrics()
                df["Detection"] = results
                st.dataframe(df)
        else:
            st.error("CSV must have at least 180 columns (real MIT-BIH) or 10 columns (legacy).")

# ============================================================
# DETECTION HISTORY LOG
# ============================================================
st.markdown("---")
st.subheader("📜 Detection History Log")
if st.session_state["detection_log"]:
    st.dataframe(st.session_state["detection_log"], use_container_width=True)
    if st.button("🗑️ Clear Log"):
        st.session_state["detection_log"] = []
        st.rerun()
else:
    st.info("No detections yet. Run any workflow to populate the log.")

# Final redraw so the metrics always reflect inferences made during THIS run
# (including ones the phone thread made while the script was executing).
render_metrics()