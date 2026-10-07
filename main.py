"""ECG Anomaly Detection -- Dense autoencoder, end to end.

Run with: python3 main.py
Trains a Dense autoencoder on normal-only ECG5000 heartbeats, evaluates
anomaly detection on a held-out mixed test set, and saves plots and
metrics to ./output/.

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
from src.models import build_dense_autoencoder
from src.evaluate import (
    reconstruction_error, pick_threshold, classify, compute_metrics, compute_roc,
)

# --- Configuration ---
SEED = 42
OUTPUT_DIR = "output"
TEST_SIZE = 0.2          # fraction of NORMAL beats held out for testing
VALIDATION_SPLIT = 0.1   # fraction of normal training data used for validation
EPOCHS = 100
BATCH_SIZE = 128


def set_seeds(seed: int = SEED) -> None:
    """Seed Python/NumPy/TensorFlow and request deterministic TF ops.

    Exact numbers can still differ slightly across hardware/TF versions.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def main() -> None:
    set_seeds()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ECG5000 dataset...")
    # Min/max scaling is fitted on normal training data inside load_dataset.
    normal_train, test_data, test_labels, _scaler_params = load_dataset(
        seed=SEED, test_size=TEST_SIZE
    )
    n_timesteps = normal_train.shape[1]
    print(f"Train (normal only): {normal_train.shape}, Test: {test_data.shape}")

    print("\nTraining Dense autoencoder...")
    model = build_dense_autoencoder(n_timesteps)
    history = model.fit(
        normal_train, normal_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, shuffle=True, verbose=0,
    )
    print(f"Final val_loss: {history.history['val_loss'][-1]:.4f}")

    # Threshold from NORMAL training reconstruction errors only (no test labels).
    train_errors = reconstruction_error(model, normal_train)
    test_errors = reconstruction_error(model, test_data)
    threshold = pick_threshold(train_errors)

    predictions = classify(test_errors, threshold)
    metrics = compute_metrics(test_labels, predictions)
    fpr, tpr, roc_auc = compute_roc(test_labels, test_errors)

    print("\n--- Dense Autoencoder ---")
    print(f"Threshold: {threshold:.4f}")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k}: {metrics[k]:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred; order: abnormal, normal):")
    print(metrics["confusion_matrix"])

    results = {
        "threshold": threshold,
        "auc": float(roc_auc),
        **{k: float(metrics[k]) for k in ("accuracy", "precision", "recall", "f1")},
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Dense AE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/dense_loss.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Dense AE (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Dense AE")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plots and metrics to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

