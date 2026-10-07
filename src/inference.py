"""Real-time inference helpers: saved artifacts, validation, and detection.

Inference flow for one heartbeat:
    raw values -> validate (exactly 140 finite numbers) -> float32 (1, 140)
    -> normalize with the SAVED x_min/x_max (never refitted)
    -> Dense autoencoder reconstruction -> MAE -> compare to SAVED threshold.

The model, scaler and threshold are produced by `python3 main.py`.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import min_max_scale
from src.evaluate import classify, reconstruction_mae

ARTIFACT_DIR = "artifacts"
MODEL_FILE = "dense_autoencoder.keras"
SCALER_FILE = "scaler.json"
THRESHOLD_FILE = "threshold.json"
N_TIMESTEPS = 140


class ArtifactsNotFoundError(FileNotFoundError):
    """Raised when the saved model/scaler/threshold files are missing."""


@dataclass
class Detector:
    model: tf.keras.Model
    x_min: float
    x_max: float
    threshold: float
    seed: int = 42
    test_size: float = 0.2


def save_artifacts(model, x_min: float, x_max: float, threshold: float,
                   directory: str = ARTIFACT_DIR, seed: int = 42,
                   test_size: float = 0.2) -> None:
    """Save the trained model, scaler parameters and threshold."""
    os.makedirs(directory, exist_ok=True)
    model.save(os.path.join(directory, MODEL_FILE))
    with open(os.path.join(directory, SCALER_FILE), "w") as f:
        json.dump({"x_min": float(x_min), "x_max": float(x_max),
                   "seed": seed, "test_size": test_size}, f, indent=2)
    with open(os.path.join(directory, THRESHOLD_FILE), "w") as f:
        json.dump({"threshold": float(threshold)}, f, indent=2)


def load_artifacts(directory: str = ARTIFACT_DIR) -> Detector:
    """Load the saved model, scaler parameters and threshold."""
    paths = [os.path.join(directory, n) for n in (MODEL_FILE, SCALER_FILE, THRESHOLD_FILE)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise ArtifactsNotFoundError(
            f"Missing artifact file(s): {', '.join(missing)}. Run `python3 main.py` first."
        )
    model = tf.keras.models.load_model(paths[0])
    with open(paths[1]) as f:
        scaler = json.load(f)
    with open(paths[2]) as f:
        threshold = json.load(f)["threshold"]
    return Detector(model=model, x_min=scaler["x_min"], x_max=scaler["x_max"],
                    threshold=float(threshold), seed=scaler.get("seed", 42),
                    test_size=scaler.get("test_size", 0.2))


def validate_heartbeat(values) -> np.ndarray:
    """Return a float32 vector of exactly 140 finite values, else raise ValueError.

    Never truncates or pads.
    """
    try:
        arr = np.asarray(values, dtype="float32").reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Heartbeat contains non-numeric values.") from exc
    if arr.size != N_TIMESTEPS:
        raise ValueError(f"Heartbeat must have exactly {N_TIMESTEPS} values, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Heartbeat contains missing (NaN) or infinite values.")
    return arr


def detect(detector: Detector, heartbeat) -> dict:
    """Run inference on ONE raw (unscaled) heartbeat."""
    raw = validate_heartbeat(heartbeat)
    normalized = min_max_scale(raw, detector.x_min, detector.x_max)  # saved x_min/x_max
    x = normalized.reshape(1, N_TIMESTEPS).astype("float32")
    reconstruction = detector.model.predict(x, verbose=0)
    error = float(reconstruction_mae(x, reconstruction)[0])
    is_normal = bool(classify(np.array([error]), detector.threshold)[0])
    return {
        "original": normalized,
        "reconstruction": reconstruction.reshape(-1),
        "reconstruction_error": error,
        "threshold": detector.threshold,
        "prediction": "normal" if is_normal else "anomaly",
        "status": "NORMAL" if is_normal else "ANOMALY DETECTED",
    }


def parse_ecg_csv(source) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse an uploaded/path CSV of pre-segmented heartbeats.

    One heartbeat per row: 140 values (no labels), or 141 columns where the
    last is an optional label (ECG5000 convention: 1 = normal, other = abnormal).
    An all-text first row is treated as a header. Returns (beats, labels|None
    with 1 = normal, 0 = abnormal). Raises ValueError with a clear message.
    """
    try:
        df = pd.read_csv(source, header=None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Malformed CSV: {exc}") from exc
    if df.empty:
        raise ValueError("The file contains no rows.")

    # Header row: every cell of the first row is non-numeric text.
    first = pd.to_numeric(df.iloc[0], errors="coerce")
    if first.isna().all() and df.shape[0] > 1:
        df = df.iloc[1:].reset_index(drop=True)

    n_cols = df.shape[1]
    if n_cols not in (N_TIMESTEPS, N_TIMESTEPS + 1):
        raise ValueError(
            f"Expected {N_TIMESTEPS} ECG values per row (optionally +1 label column), "
            f"found {n_cols} columns."
        )

    numeric = df.apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna() & df.notna()
    if bad.to_numpy().any():
        row = int(np.argmax(bad.any(axis=1).to_numpy())) + 1
        raise ValueError(f"Non-numeric value found (first at data row {row}).")
    if numeric.isna().to_numpy().any():
        row = int(np.argmax(numeric.isna().any(axis=1).to_numpy())) + 1
        raise ValueError(f"Missing value found (first at data row {row}).")

    values = numeric.to_numpy(dtype="float64")
    labels = None
    if n_cols == N_TIMESTEPS + 1:
        labels = (values[:, -1] == 1).astype("int32")
        values = values[:, :-1]
    if not np.all(np.isfinite(values)):
        raise ValueError("ECG values must be finite numbers.")
    return values.astype("float32"), labels
