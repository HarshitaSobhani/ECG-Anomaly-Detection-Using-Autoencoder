"""Reconstruction-error-based anomaly scoring and classification metrics."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)


def reconstruction_error(model, data_flat, model_input=None):
    """Mean absolute error per sequence between original and reconstruction.

    `model_input` lets LSTM models (which need a 3D input shape) be scored
    while errors are still computed against the original flat sequences.
    """
    model_input = data_flat if model_input is None else model_input
    reconstructions = model.predict(model_input, verbose=0).reshape(data_flat.shape)
    return np.mean(np.abs(data_flat - reconstructions), axis=1)


def pick_threshold(train_errors: np.ndarray) -> float:
    """threshold = mean + std of reconstruction error on normal training data."""
    return float(train_errors.mean() + train_errors.std())


def classify(errors: np.ndarray, threshold: float) -> np.ndarray:
    """1 = predicted normal, 0 = predicted abnormal (error > threshold)."""
    return (errors <= threshold).astype("int32")


def compute_metrics(y_true, y_pred) -> dict:
    """Metrics reported w.r.t. the abnormal class (pos_label=0)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=0),
        "recall": recall_score(y_true, y_pred, pos_label=0),
        "f1": f1_score(y_true, y_pred, pos_label=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def compute_roc(y_true_labels, errors):
    """ROC/AUC for detecting the abnormal class from raw reconstruction error."""
    fpr, tpr, _ = roc_curve(1 - y_true_labels, errors)
    return fpr, tpr, auc(fpr, tpr)
