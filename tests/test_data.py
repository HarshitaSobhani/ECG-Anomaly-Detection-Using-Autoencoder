import numpy as np
import pandas as pd

from src.data import to_sequences_and_labels, load_dataset


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


def test_load_dataset_uses_normal_only_for_training(monkeypatch):
    df = _fake_df(n_normal=8, n_abnormal=4, n_timesteps=6)
    monkeypatch.setattr("src.data.load_raw", lambda url=None: df)

    normal_train, test_data, test_labels, _ = load_dataset(seed=0, test_size=0.25)

    # 25% of the 8 normal sequences held out for test -> 2
    assert normal_train.shape[0] == 6
    assert test_data.shape[0] == 2 + 4
    assert test_labels.sum() == 2  # only the held-out normal sequences are labeled 1


def test_scaler_fit_on_normal_train_only(monkeypatch):
    # Abnormal rows have huge values; they must not influence the scaling.
    rng = np.random.RandomState(0)
    rows = [list(rng.rand(6)) + [1] for _ in range(8)]
    rows += [list(rng.rand(6) * 100) + [2] for _ in range(4)]
    monkeypatch.setattr("src.data.load_raw", lambda url=None: pd.DataFrame(rows))

    normal_train, test_data, _, (x_min, x_max) = load_dataset(seed=0, test_size=0.25)

    assert normal_train.min() == 0.0 and normal_train.max() == 1.0
    assert test_data.max() > 1.0  # abnormal data scaled with train params, not refit
    assert x_max < 1.0 + 1e-9  # fitted on normal (0-1) data, not on abnormal (0-100)

