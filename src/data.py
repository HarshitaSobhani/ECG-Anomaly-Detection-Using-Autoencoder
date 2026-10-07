"""Download, split, normalize, and prepare the ECG5000 dataset.

Only *normal* heartbeats are used for training (unsupervised anomaly
detection setup) -- see EXPLANATION.md for the reasoning.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"


def load_raw(url: str = DATA_URL) -> pd.DataFrame:
    return pd.read_csv(url, header=None)


def to_sequences_and_labels(df: pd.DataFrame):
    """Split the last column (label) from the 140-step sequences.

    Label 1 = normal, anything else = abnormal. We collapse the abnormal
    subclasses (2-5) into a single binary "abnormal" class since we only
    care about normal-vs-not for this task.
    """
    raw = df.values
    sequences = raw[:, :-1].astype("float32")
    labels = raw[:, -1].astype("int32")
    binary_labels = (labels == 1).astype("int32")  # 1 = normal, 0 = abnormal
    return sequences, binary_labels


def fit_min_max(train: np.ndarray) -> tuple[float, float]:
    """Global min/max of the NORMAL TRAINING data only (no test information)."""
    return float(train.min()), float(train.max())


def min_max_scale(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Scale with already-fitted parameters. Never refit on test data."""
    return (x - x_min) / (x_max - x_min)


def load_dataset(seed: int = 42, test_size: float = 0.2, url: str = DATA_URL):
    """Returns (normal_train, test_data, test_labels, scaler_params).

    Order of operations (no leakage):
      1. separate features/labels, make binary labels, split normal vs abnormal
      2. split the raw NORMAL sequences into train / held-out normal test
      3. fit the min/max ONLY on the normal training split
      4. apply those same parameters to train, held-out normal and abnormal data

    normal_train: scaled normal sequences, used to train the autoencoder.
    test_data/test_labels: held-out normal (label 1) + all abnormal (label 0)
                           sequences, used only for evaluation.
    scaler_params: (x_min, x_max) fitted on the training data; reuse with
                   min_max_scale() to transform any new ECG.
    Test values may fall slightly outside [0, 1]; they are not clipped.

    A single global min/max is used (not one per time step) so the shape of
    each heartbeat is preserved exactly.
    """
    df = load_raw(url)
    sequences, binary_labels = to_sequences_and_labels(df)

    raw_normal = sequences[binary_labels == 1]
    raw_abnormal = sequences[binary_labels == 0]

    raw_train, raw_normal_test = train_test_split(
        raw_normal, test_size=test_size, random_state=seed
    )

    x_min, x_max = fit_min_max(raw_train)  # normal training data only

    normal_train = min_max_scale(raw_train, x_min, x_max)
    normal_test = min_max_scale(raw_normal_test, x_min, x_max)
    abnormal_test = min_max_scale(raw_abnormal, x_min, x_max)

    test_data = np.concatenate([normal_test, abnormal_test], axis=0)
    test_labels = np.concatenate([
        np.ones(len(normal_test), dtype="int32"),
        np.zeros(len(abnormal_test), dtype="int32"),
    ])

    return normal_train, test_data, test_labels, (x_min, x_max)
