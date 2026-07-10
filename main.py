"""ECG Anomaly Detection -- Dense AE vs LSTM AE, end to end.

Run with: python3 main.py
Trains both autoencoders on normal-only ECG5000 heartbeats, evaluates
anomaly detection on a held-out mixed test set, and saves comparison
plots to ./output/.

This mirrors ecg_anomaly_detection.ipynb; see EXPLANATION.md for the
theory behind each step.
"""
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

SEED = 42
OUTPUT_DIR = "output"


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    set_seeds()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ECG5000 dataset...")
    normal_train, test_data, test_labels, normal_data, abnormal_data = load_dataset(seed=SEED)
    n_timesteps = normal_train.shape[1]
    print(f"Train (normal only): {normal_train.shape}, Test: {test_data.shape}")

    # --- Dense autoencoder ---
    print("\nTraining Dense autoencoder...")
    dense_ae = build_dense_autoencoder(n_timesteps)
    dense_history = dense_ae.fit(
        normal_train, normal_train, epochs=100, batch_size=128,
        validation_split=0.1, shuffle=True, verbose=0,
    )
    print(f"Final Dense AE val_loss: {dense_history.history['val_loss'][-1]:.4f}")

    # --- LSTM autoencoder ---
    print("\nTraining LSTM autoencoder...")
    normal_train_lstm = normal_train.reshape((*normal_train.shape, 1))
    test_data_lstm = test_data.reshape((*test_data.shape, 1))

    lstm_ae = build_lstm_autoencoder(n_timesteps)
    lstm_history = lstm_ae.fit(
        normal_train_lstm, normal_train_lstm, epochs=100, batch_size=64,
        validation_split=0.1, shuffle=True, verbose=0,
    )
    print(f"Final LSTM AE val_loss: {lstm_history.history['val_loss'][-1]:.4f}")

    # --- Thresholds (from training-set reconstruction error only) ---
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

    print("\n--- Dense Autoencoder ---")
    print(f"Threshold: {dense_threshold:.4f}")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k}: {dense_metrics[k]:.4f}")
    print(f"AUC: {dense_auc:.4f}")

    print("\n--- LSTM Autoencoder ---")
    print(f"Threshold: {lstm_threshold:.4f}")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k}: {lstm_metrics[k]:.4f}")
    print(f"AUC: {lstm_auc:.4f}")

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
