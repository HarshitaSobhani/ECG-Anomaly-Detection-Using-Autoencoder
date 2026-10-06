"""ECG Anomaly Detection -- Dense AE vs LSTM AE, end to end.

Run with: python3 main.py
Trains both autoencoders on normal-only ECG5000 heartbeats, evaluates
anomaly detection on a held-out mixed test set, and saves comparison
plots to ./output/.

This mirrors ecg_anomaly_detection.ipynb; see EXPLANATION.md for the
theory behind each step.
"""
import json
import os
import random

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_dataset
from src.models import build_dense_autoencoder, build_lstm_autoencoder
from src.evaluate import (
    reconstruction_error, pick_threshold, classify, compute_metrics, compute_roc,
)

# --- Configuration ---
SEED = 42
OUTPUT_DIR = "output"
TEST_SIZE = 0.2          # fraction of NORMAL beats held out for testing
VALIDATION_SPLIT = 0.1   # fraction of normal training data used for validation
DENSE_EPOCHS, DENSE_BATCH_SIZE = 100, 128
LSTM_EPOCHS, LSTM_BATCH_SIZE = 100, 64


def set_seeds(seed: int = SEED) -> None:
    """Seed Python/NumPy/TensorFlow and request deterministic TF ops.

    Exact numbers can still differ slightly across hardware/TF versions.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def print_report(name, threshold, metrics, roc_auc):
    print(f"\n--- {name} ---")
    print(f"Threshold: {threshold:.4f}")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k}: {metrics[k]:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred; order: abnormal, normal):")
    print(metrics["confusion_matrix"])


def main():
    set_seeds()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ECG5000 dataset...")
    normal_train, test_data, test_labels, normal_data, abnormal_data = load_dataset(seed=SEED, test_size=TEST_SIZE)
    n_timesteps = normal_train.shape[1]
    print(f"Train (normal only): {normal_train.shape}, Test: {test_data.shape}")

    # --- Dense autoencoder ---
    print("\nTraining Dense autoencoder...")
    dense_ae = build_dense_autoencoder(n_timesteps)
    dense_history = dense_ae.fit(
        normal_train, normal_train, epochs=DENSE_EPOCHS, batch_size=DENSE_BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, shuffle=True, verbose=0,
    )
    print(f"Final Dense AE val_loss: {dense_history.history['val_loss'][-1]:.4f}")

    # --- LSTM autoencoder ---
    print("\nTraining LSTM autoencoder...")
    normal_train_lstm = normal_train.reshape((*normal_train.shape, 1))
    test_data_lstm = test_data.reshape((*test_data.shape, 1))

    lstm_ae = build_lstm_autoencoder(n_timesteps)
    lstm_history = lstm_ae.fit(
        normal_train_lstm, normal_train_lstm, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, shuffle=True, verbose=0,
    )
    print(f"Final LSTM AE val_loss: {lstm_history.history['val_loss'][-1]:.4f}")

    # --- Thresholds (from NORMAL training reconstruction error only; no test labels) ---
    dense_train_errors = reconstruction_error(dense_ae, normal_train)
    dense_test_errors = reconstruction_error(dense_ae, test_data)
    dense_threshold = pick_threshold(dense_train_errors)

    lstm_train_errors = reconstruction_error(lstm_ae, normal_train, normal_train_lstm)
    lstm_test_errors = reconstruction_error(lstm_ae, test_data, test_data_lstm)
    lstm_threshold = pick_threshold(lstm_train_errors)

    # --- Classification + metrics ---
    dense_preds = classify(dense_test_errors, dense_threshold)
    lstm_preds = classify(lstm_test_errors, lstm_threshold)

    dense_metrics = compute_metrics(test_labels, dense_preds)
    lstm_metrics = compute_metrics(test_labels, lstm_preds)

    dense_fpr, dense_tpr, dense_auc = compute_roc(test_labels, dense_test_errors)
    lstm_fpr, lstm_tpr, lstm_auc = compute_roc(test_labels, lstm_test_errors)

    print_report("Dense Autoencoder", dense_threshold, dense_metrics, dense_auc)
    print_report("LSTM Autoencoder", lstm_threshold, lstm_metrics, lstm_auc)

    results = {}
    for name, thr, m, a_ in (("dense", dense_threshold, dense_metrics, dense_auc),
                             ("lstm", lstm_threshold, lstm_metrics, lstm_auc)):
        results[name] = {"threshold": thr, "auc": float(a_),
                         **{k: float(m[k]) for k in ("accuracy", "precision", "recall", "f1")},
                         "confusion_matrix": m["confusion_matrix"].tolist()}
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    plt.figure(figsize=(7, 4))
    plt.plot(dense_history.history["loss"], label="Train")
    plt.plot(dense_history.history["val_loss"], label="Val")
    plt.title("Dense AE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/dense_loss.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(lstm_history.history["loss"], label="Train")
    plt.plot(lstm_history.history["val_loss"], label="Val")
    plt.title("LSTM AE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/lstm_loss.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(dense_fpr, dense_tpr, label=f"Dense AE (AUC={dense_auc:.3f})")
    plt.plot(lstm_fpr, lstm_tpr, label=f"LSTM AE (AUC={lstm_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Dense AE vs LSTM AE")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plots to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
