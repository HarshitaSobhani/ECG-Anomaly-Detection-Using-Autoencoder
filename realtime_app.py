"""ECG Anomaly Detection -- real-time inference demo (Streamlit).

Run `python3 main.py` first (trains + saves ./artifacts/), then:
    streamlit run realtime_app.py

The app NEVER trains. It loads the saved Dense autoencoder, scaler (x_min,
x_max) and threshold, and processes heartbeats one at a time.
"""
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data import load_raw_test_set
from src.inference import (
    ARTIFACT_DIR, ArtifactsNotFoundError, detect, load_artifacts, parse_ecg_csv,
)

MODE_ECG5000 = "ECG5000 simulation"
MODE_CSV = "Upload ECG CSV"
CSV_WARNING = (
    "External ECG data must be compatible with the model's 140-sample heartbeat "
    "representation. This model was trained on ECG5000, so cross-dataset "
    "performance should be validated before drawing conclusions."
)

st.set_page_config(page_title="ECG Anomaly Detection", layout="wide")


@st.cache_resource(show_spinner="Loading saved model...")
def get_detector():
    return load_artifacts(ARTIFACT_DIR)


@st.cache_data(show_spinner="Loading ECG5000 test heartbeats...")
def get_ecg5000_test(seed: int, test_size: float):
    return load_raw_test_set(seed=seed, test_size=test_size)


def new_session(beats, labels, source_key):
    """(Re)start a session: reset position and statistics."""
    st.session_state.update(
        source_key=source_key, beats=beats, labels=labels, pos=0, running=False,
        last=None, processed=0, pred_anomaly=0, actual_abnormal=0, correct=0,
        labelled=0,
    )


def process_next():
    """Run inference on the next heartbeat only, and update statistics."""
    s = st.session_state
    if s.pos >= len(s.beats):
        s.running = False
        return False
    result = detect(get_detector(), s.beats[s.pos])
    actual = None if s.labels is None else int(s.labels[s.pos])
    result.update(index=s.pos + 1, actual=actual)
    s.last = result
    s.processed += 1
    s.pred_anomaly += result["prediction"] == "anomaly"
    if actual is not None:  # ground truth: statistics only, never model input
        s.labelled += 1
        s.actual_abnormal += actual == 0
        s.correct += (actual == 1) == (result["prediction"] == "normal")
    s.pos += 1
    return True


def plot_beat(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=result["original"], name="Original ECG",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(y=result["reconstruction"], name="Reconstructed ECG",
                             line=dict(color="#d62728", width=2, dash="dash")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Time step (normalized heartbeat)",
                      yaxis_title="Normalized amplitude",
                      legend=dict(orientation="h", y=1.1))
    return fig


# ----------------------------------------------------------------- header
st.title("ECG Anomaly Detection — Real-Time Inference")
st.caption("Dense Autoencoder trained on normal ECG5000 heartbeats")
st.info("Experimental / educational demo. A real-time inference simulation, "
        "not live physiological hardware and not a clinical diagnostic system.")

try:
    detector = get_detector()
except ArtifactsNotFoundError:
    st.error(f"Saved model artifacts not found in `{ARTIFACT_DIR}/`. "
             "Run `python3 main.py` first to train the model and create them.")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", [MODE_ECG5000, MODE_CSV])
    delay = st.slider("Seconds per heartbeat", 0.5, 1.0, 0.7, 0.1)
    shuffle = st.checkbox("Shuffle ECG5000 order (demo)", value=True,
                          help="Test set is ordered normal-then-abnormal; shuffling "
                               "mixes them. Fixed seed, does not affect predictions.")
    st.caption(f"Saved threshold: {detector.threshold:.4f}")
    st.caption(f"Saved scaler: x_min={detector.x_min:.3f}, x_max={detector.x_max:.3f}")

# ------------------------------------------------------------ data source
if mode == MODE_ECG5000:
    raw_test, test_labels = get_ecg5000_test(detector.seed, detector.test_size)
    order = (np.random.RandomState(0).permutation(len(raw_test)) if shuffle
             else np.arange(len(raw_test)))
    key = ("ecg5000", shuffle)
    if st.session_state.get("source_key") != key:
        new_session(raw_test[order], test_labels[order], key)
else:
    st.warning(CSV_WARNING)
    st.caption("CSV format: one heartbeat per row, exactly 140 ECG values; an optional "
               "141st column is a label (1 = normal, other = abnormal) used for "
               "evaluation display only. The scaler is NOT refitted.")
    upload = st.file_uploader("ECG CSV", type=["csv", "txt"])
    if upload is None:
        st.session_state.pop("source_key", None)
        st.stop()
    key = ("csv", upload.name, upload.size)
    if st.session_state.get("source_key") != key:
        try:
            beats, labels = parse_ecg_csv(upload)
        except ValueError as exc:
            st.session_state.pop("source_key", None)
            st.error(f"Could not read the CSV: {exc}")
            st.stop()
        new_session(beats, labels, key)

s = st.session_state

# --------------------------------------------------------------- controls
c1, c2, c3, c4 = st.columns(4)
if c1.button("▶ START REAL-TIME DETECTION", type="primary", width="stretch"):
    if s.pos >= len(s.beats):
        new_session(s.beats, s.labels, s.source_key)
    s.running = True
if c2.button("■ STOP DETECTION", width="stretch"):
    s.running = False
next_clicked = c3.button("NEXT HEARTBEAT", width="stretch")
if c4.button("RESET SESSION", width="stretch"):
    new_session(s.beats, s.labels, s.source_key)

if next_clicked and not s.running:
    process_next()
elif s.running:
    if not process_next():
        st.success("Reached the end of the data.")

# ---------------------------------------------------------------- display
last = s.last
if last is None:
    st.write(f"{len(s.beats)} heartbeats loaded. Press START or NEXT HEARTBEAT.")
else:
    st.subheader(f"Current Heartbeat: {last['index']} / {len(s.beats)}")
    st.plotly_chart(plot_beat(last), width="stretch")

    m1, m2 = st.columns(2)
    m1.metric("Reconstruction Error (MAE)", f"{last['reconstruction_error']:.4f}")
    m2.metric("Anomaly Threshold", f"{last['threshold']:.4f}")

    if last["prediction"] == "anomaly":
        st.error(f"STATUS: {last['status']}", icon="🚨")
    else:
        st.success(f"STATUS: {last['status']}", icon="✅")

    if last["actual"] is not None:
        actual = "Normal" if last["actual"] == 1 else "Abnormal"
        pred = "Normal" if last["prediction"] == "normal" else "Anomaly"
        st.caption(f"Ground Truth (evaluation only) — Actual: {actual} | Predicted: {pred}")

    st.divider()
    st1, st2, st3 = st.columns(3)
    st1.metric("Processed Beats", s.processed)
    st2.metric("Anomalies Detected", s.pred_anomaly)
    st3.metric("Predicted Normal", s.processed - s.pred_anomaly)
    if s.labelled:
        st.caption(f"Ground truth (evaluation only): {s.actual_abnormal} abnormal / "
                   f"{s.labelled - s.actual_abnormal} normal so far; running accuracy "
                   f"{s.correct / s.labelled:.1%}")

# Real-time pacing: wait, then rerun to process the next heartbeat.
if s.running:
    time.sleep(delay)
    st.rerun()
