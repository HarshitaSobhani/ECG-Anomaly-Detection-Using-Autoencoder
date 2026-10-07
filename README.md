# ECG Anomaly Detection using a Dense Autoencoder

Detects abnormal heartbeats in the ECG5000 dataset by training a Dense autoencoder **only on
normal ECG sequences**, then flagging anomalies via reconstruction error.

> Experimental / educational project. Not a clinical diagnostic system.

See [`EXPLANATION.md`](EXPLANATION.md) for a full walkthrough.

## Contents

- `ecg_anomaly_detection.ipynb` — lab notebook: theory, code, plots, evaluation, conclusion.
- `build_notebook.py` — regenerates the notebook (regenerates the notebook without outputs).
- `main.py` — runnable end-to-end script (same pipeline as the notebook).
- `src/data.py` — download, split, train-fitted min-max scaling, test-set construction.
- `src/models.py` — Dense autoencoder.
- `src/evaluate.py` — reconstruction error, thresholding, metrics, ROC/AUC.

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

## Results (actual run of `python3 main.py`, seed 42)

| Metric | Dense AE |
|---|---|
| Threshold | 0.0187 |
| Accuracy | 0.9752 |
| Precision (abnormal) | 0.9714 |
| Recall (abnormal) | 0.9976 |
| F1 (abnormal) | 0.9843 |
| AUC | 0.9894 |

Confusion matrix (rows = true, columns = predicted; order abnormal, normal): `[[2074, 5], [61, 523]]`.

Exact numbers may vary slightly with hardware / library version even with fixed seeds and
deterministic ops. These numbers come from the corrected train-fitted-scaling pipeline; older
figures from the previous pipeline are superseded. Metrics are saved to `output/metrics.json`.

## Limitations and future work

The `mean + std` threshold is a simple heuristic computed on the training data; a
validation-based threshold would be better. One seed and split, merged abnormal subclasses, no
hyperparameter search, clean pre-segmented data. Future: validation threshold, multiple seeds,
per-class analysis, convolutional / variational autoencoders.

## License

MIT
