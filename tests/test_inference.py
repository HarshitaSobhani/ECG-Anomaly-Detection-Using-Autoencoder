import io

import numpy as np
import pytest

from src.data import fit_min_max, load_raw, to_sequences_and_labels, min_max_scale
from src.evaluate import pick_threshold, reconstruction_error
from src.inference import (
    ArtifactsNotFoundError, detect, load_artifacts, parse_ecg_csv,
    save_artifacts, validate_heartbeat,
)
from src.models import build_dense_autoencoder


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """Tiny real training run on synthetic 'normal' beats; saved + reloaded."""
    rng = np.random.RandomState(0)
    t = np.linspace(0, 2 * np.pi, 140)
    raw = (np.sin(t) * 3 + rng.normal(0, 0.1, (200, 140))).astype("float32")
    x_min, x_max = fit_min_max(raw)
    scaled = min_max_scale(raw, x_min, x_max)
    model = build_dense_autoencoder(140)
    model.fit(scaled, scaled, epochs=5, batch_size=32, verbose=0)
    threshold = pick_threshold(reconstruction_error(model, scaled))
    directory = str(tmp_path_factory.mktemp("artifacts"))
    save_artifacts(model, x_min, x_max, threshold, directory, seed=7, test_size=0.3)
    return directory, raw, (x_min, x_max, threshold)


def test_artifacts_roundtrip(trained_artifacts):
    directory, _, (x_min, x_max, threshold) = trained_artifacts
    det = load_artifacts(directory)
    assert (det.x_min, det.x_max) == (x_min, x_max)      # scaler loading
    assert det.threshold == pytest.approx(threshold)     # threshold loading
    assert (det.seed, det.test_size) == (7, 0.3)
    assert det.model.output_shape == (None, 140)         # model loading


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(ArtifactsNotFoundError, match="main.py"):
        load_artifacts(str(tmp_path / "nope"))


@pytest.mark.parametrize("bad", [np.zeros(139), np.zeros(141), np.zeros(0), ["a"] * 140])
def test_validate_rejects_wrong_length_or_type(bad):
    with pytest.raises(ValueError):
        validate_heartbeat(bad)


def test_validate_rejects_nan_and_accepts_140():
    beat = np.zeros(140)
    beat[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_heartbeat(beat)
    out = validate_heartbeat(list(range(140)))
    assert out.shape == (140,) and out.dtype == np.float32


def test_detect_uses_saved_scaler_and_threshold(trained_artifacts):
    directory, raw, (x_min, x_max, threshold) = trained_artifacts
    det = load_artifacts(directory)
    res = detect(det, raw[0])
    expected_norm = (raw[0] - x_min) / (x_max - x_min)       # normalization with saved params
    assert np.allclose(res["original"], expected_norm, atol=1e-6)
    # error equals the batch pipeline's MAE on the same normalized beat
    batch = reconstruction_error(det.model, expected_norm.reshape(1, 140).astype("float32"))[0]
    assert res["reconstruction_error"] == pytest.approx(batch, rel=1e-5)
    assert res["reconstruction"].shape == (140,)
    assert res["threshold"] == pytest.approx(threshold)
    assert res["prediction"] == ("anomaly" if res["reconstruction_error"] > threshold else "normal")


def test_threshold_classification_both_sides(trained_artifacts):
    directory, raw, _ = trained_artifacts
    det = load_artifacts(directory)
    det.threshold = 0.0                       # everything exceeds -> anomaly
    assert detect(det, raw[0])["prediction"] == "anomaly"
    det.threshold = 1e9                       # nothing exceeds -> normal
    assert detect(det, raw[0])["prediction"] == "normal"


def test_end_to_end_real_ecg5000_heartbeats(trained_artifacts):
    directory, _, _ = trained_artifacts
    try:
        df = load_raw()
    except Exception as exc:  # offline
        pytest.skip(f"ECG5000 unavailable: {exc}")
    seqs, labels = to_sequences_and_labels(df)
    det = load_artifacts(directory)
    res = detect(det, seqs[0])
    assert res["prediction"] in ("normal", "anomaly")
    assert res["reconstruction_error"] >= 0


def _csv(text):
    return io.StringIO(text)


def test_parse_csv_valid_with_and_without_labels():
    row = ",".join(["0.5"] * 140)
    beats, labels = parse_ecg_csv(_csv(row + "\n" + row))
    assert beats.shape == (2, 140) and labels is None
    beats, labels = parse_ecg_csv(_csv(row + ",1\n" + row + ",3"))
    assert beats.shape == (2, 140) and labels.tolist() == [1, 0]


@pytest.mark.parametrize("text,msg", [
    ("", "empty"),
    (",".join(["1"] * 139) + "\n", "columns"),
    (",".join(["1"] * 143) + "\n", "columns"),
    (",".join(["1"] * 139 + ["x"]) + "\n" + ",".join(["1"] * 140), "Non-numeric"),
    (",".join(["1"] * 139 + [""]) + "\n", "Missing"),
])
def test_parse_csv_rejects_bad_input(text, msg):
    with pytest.raises(ValueError, match=msg):
        parse_ecg_csv(_csv(text))
