# ECG Anomaly Detection using a Dense Autoencoder

Detects abnormal heartbeats in the ECG5000 dataset by training a Dense autoencoder **only on
normal ECG sequences**, then flagging anomalies via reconstruction error.

> Experimental / educational project. Not a clinical diagnostic system.

## Contents

- `ecg_anomaly_detection.ipynb` — lab notebook: theory, code, plots, evaluation, conclusion.
- `main.py` — runnable end-to-end script (same pipeline as the notebook).
- `src/data.py` — download, split, train-fitted min-max scaling, test-set construction.
- `src/models.py` — Dense autoencoder.
- `src/evaluate.py` — reconstruction error, thresholding, metrics, ROC/AUC.
- `src/inference.py` — saved artifacts, validation and single-heartbeat inference.
- `realtime_app.py` — Streamlit real-time demo.

## Dataset

[ECG5000](http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv) — 4998
heartbeats, 140 values each. Label `1` = normal, `2`-`5` = abnormal (merged into one binary
"abnormal" class). Downloaded automatically.

## Preprocessing (no leakage)

Binary labels (`1` = normal, `0` = abnormal) → separate normal/abnormal → split normal beats
80/20 into train / held-out test → min-max scaling fitted **only on normal training data** →
the same min/max applied to the held-out normal and abnormal data. Test set = held-out normal +
all abnormal. Test labels are only used for evaluation.

## Model

Dense autoencoder: `140 → 64 → 32 → 16 → 32 → 64 → 140`. ReLU hidden layers, sigmoid output,
MAE loss, Adam. Trained on normal beats only: 100 epochs, batch size 128, 10% validation split.

## Anomaly detection

Anomaly score = reconstruction error `mean(|original - reconstructed|)` per ECG.
`threshold = mean(normal training errors) + std(normal training errors)`.
`error <= threshold` → normal, `error > threshold` → anomaly. Abnormal is the positive class
for precision / recall / F1 / ROC.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

```bash
python3 main.py
```
Trains the model, prints metrics and saves the loss curve, ROC plot and `metrics.json` to
`./output/`. Notebook: `jupyter notebook ecg_anomaly_detection.ipynb` then Restart & Run All
(or upload it to Google Colab).

## Results (latest run of `python3 main.py`, seed 42)

| Metric | Dense AE |
|---|---:|
| Threshold | 0.0187 |
| Accuracy | 0.9756 |
| Precision (abnormal) | 0.9737 |
| Recall (abnormal) | 0.9957 |
| F1 (abnormal) | 0.9845 |
| AUC | 0.9892 |

Validation loss (final, MAE): 0.0128

Confusion matrix (rows = true labels, columns = predicted labels; order = abnormal, normal):

```
[[2070,   9],
 [  56, 528]]
```

True positives (abnormal caught) = 2070, false negatives = 9, false positives = 56,
true negatives = 528.

Saved inference artifacts from this run: `x_min = -7.090373992919922`,
`x_max = 4.966413974761963`, threshold `0.018679914996027946`.

Exact numbers may vary slightly with hardware / library version even with fixed seeds and
deterministic ops. These numbers come from the corrected train-fitted-scaling pipeline; older
figures from the previous pipeline are superseded. Metrics are saved to `output/metrics.json`.

## Real-Time Demo

An addition to the project: the trained Dense Autoencoder is used for real-time **inference**
in a Streamlit app. The methodology (architecture, MAE, mean + std threshold, train-fitted
scaling) is unchanged.

### Training

```bash
python3 main.py
```

Trains the existing Dense Autoencoder, prints the metrics, and saves the inference artifacts
to `artifacts/`: `dense_autoencoder.keras` (model), `scaler.json` (`x_min`, `x_max` fitted on
normal training data, plus the split seed/size) and `threshold.json` (the mean + std
threshold from `pick_threshold`). Nothing is hardcoded.

### Start the application

```bash
pip install -r requirements.txt
streamlit run realtime_app.py
```

The app loads the saved model, scaler and threshold and performs inference **without
retraining**. If `artifacts/` is missing it shows an error asking you to run `python3 main.py`.

### Real-Time Detection

```
ECG heartbeat -> preprocessing (saved x_min/x_max) -> Dense Autoencoder -> reconstruction
              -> MAE -> saved threshold -> NORMAL / ANOMALY
```

Each heartbeat is processed individually (`src/inference.py`). Controls: START / STOP
real-time detection, NEXT HEARTBEAT, RESET SESSION.

### Visualization

The app shows the original and reconstructed ECG waveform on the same axes, the
reconstruction error, the threshold, the prediction, and running statistics (processed beats,
anomalies detected). Ground truth labels, when available, are shown as "evaluation only" and
are never passed to the model.

### ECG5000 Simulation

The held-out ECG5000 test heartbeats (same split as `main.py`) are fed sequentially, roughly
one per second. This is a real-time inference **simulation**, not live physiological hardware.

### External CSV

Choose "Upload ECG CSV":

- one heartbeat per row, exactly 140 ECG values (an optional header row is skipped);
- an optional 141st column is a label (1 = normal, other = abnormal), used for display only;
- the scaler is **not** refitted: the saved `x_min`/`x_max` are applied;
- the model expects the same 140-value heartbeat representation as ECG5000.

Wrong column counts, empty files, non-numeric or missing values are rejected with a clear
message. Data is never truncated or padded.

### Limitations

- Experimental/educational project, not a clinical diagnostic system.
- The model was trained on ECG5000; external datasets can differ in distribution, so
  cross-dataset performance must be validated.
- Arbitrary raw ECG recordings are not converted into heartbeats by this version; only
  pre-segmented 140-value heartbeats are supported.
- The model is not clinically validated.

## Limitations and future work

The `mean + std` threshold is a simple heuristic computed on the training data; a
validation-based threshold would be better. One seed and split, merged abnormal subclasses, no
hyperparameter search, clean pre-segmented data. Future: validation threshold, multiple seeds,
per-class analysis, convolutional / variational autoencoders.

## License

MIT

