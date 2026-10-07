import numpy as np

from src.evaluate import (
    reconstruction_error, pick_threshold, classify, compute_metrics, compute_roc,
)


class _PerfectModel:
    """Reconstructs its input exactly (zero error)."""
    def predict(self, x, verbose=0):
        return x


class _OffsetModel:
    """Reconstructs its input with a constant offset (fixed nonzero error)."""
    def __init__(self, offset):
        self.offset = offset

    def predict(self, x, verbose=0):
        return x + self.offset


def test_reconstruction_error_zero_for_perfect_model():
    data = np.random.rand(5, 8)
    errors = reconstruction_error(_PerfectModel(), data)
    assert np.allclose(errors, 0.0)


def test_reconstruction_error_matches_known_offset():
    data = np.zeros((4, 8))
    errors = reconstruction_error(_OffsetModel(0.5), data)
    assert np.allclose(errors, 0.5)


def test_pick_threshold_is_mean_plus_std():
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    threshold = pick_threshold(errors)
    assert threshold == errors.mean() + errors.std()


def test_classify_splits_on_threshold():
    errors = np.array([0.1, 0.2, 0.9, 1.0])
    preds = classify(errors, threshold=0.5)
    assert preds.tolist() == [1, 1, 0, 0]


def test_compute_metrics_perfect_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_compute_roc_perfect_separation_gives_auc_one():
    y_true = np.array([1, 1, 0, 0])
    errors = np.array([0.0, 0.1, 0.9, 1.0])  # abnormal (0) has strictly higher error
    _, _, roc_auc = compute_roc(y_true, errors)
    assert roc_auc == 1.0

