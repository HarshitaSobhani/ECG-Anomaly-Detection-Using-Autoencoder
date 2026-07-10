import numpy as np
import pandas as pd

from src.data import to_sequences_and_labels, min_max_scale, load_dataset


def _fake_df(n_normal=8, n_abnormal=4, n_timesteps=5):
    rows = []
    for _ in range(n_normal):
        rows.append(list(np.random.rand(n_timesteps)) + [1])
    for _ in range(n_abnormal):
        rows.append(list(np.random.rand(n_timesteps)) + [2])
    return pd.DataFrame(rows)


def test_to_sequences_and_labels_splits_and_binarizes():
    df = _fake_df(n_normal=3, n_abnormal=2, n_timesteps=4)
    sequences, labels = to_sequences_and_labels(df)

    assert sequences.shape == (5, 4)
    assert labels.tolist() == [1, 1, 1, 0, 0]


def test_min_max_scale_bounds():
    x = np.array([0.0, 5.0, 10.0])
    scaled = min_max_scale(x)
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


def test_load_dataset_uses_normal_only_for_training(monkeypatch):
    df = _fake_df(n_normal=8, n_abnormal=4, n_timesteps=6)
    monkeypatch.setattr("src.data.load_raw", lambda url=None: df)

    normal_train, test_data, test_labels, normal_data, abnormal_data = load_dataset(
        seed=0, test_size=0.25
    )

    assert normal_data.shape[0] == 8
    assert abnormal_data.shape[0] == 4
    # 25% of the 8 normal sequences held out for test -> 2
    assert normal_train.shape[0] == 6
    assert test_data.shape[0] == 2 + 4
    assert test_labels.sum() == 2  # only the held-out normal sequences are labeled 1
