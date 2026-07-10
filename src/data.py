"""Download, normalize, and split the ECG5000 dataset.

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


def min_max_scale(x, x_min=None, x_max=None):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max
    return (x - x_min) / (x_max - x_min)


def load_dataset(seed: int = 42, test_size: float = 0.2, url: str = DATA_URL):
    """Returns (normal_train, test_data, test_labels, normal_data, abnormal_data).

    normal_train:  normal sequences only, used to train the autoencoders.
    test_data/test_labels: held-out normal sequences + all abnormal sequences,
                            used only for evaluation.
    """
    df = load_raw(url)
    sequences, binary_labels = to_sequences_and_labels(df)
    scaled = min_max_scale(sequences)

    normal_data = scaled[binary_labels == 1]
    abnormal_data = scaled[binary_labels == 0]

    normal_train, normal_test = train_test_split(
        normal_data, test_size=test_size, random_state=seed
    )

    test_data = np.concatenate([normal_test, abnormal_data], axis=0)
    test_labels = np.concatenate([
        np.ones(len(normal_test), dtype="int32"),
        np.zeros(len(abnormal_data), dtype="int32"),
    ])

    return normal_train, test_data, test_labels, normal_data, abnormal_data
