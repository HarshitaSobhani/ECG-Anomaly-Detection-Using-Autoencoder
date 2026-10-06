# ECG Anomaly Detection using Autoencoders and LSTM Networks

Detects abnormal heartbeats in the ECG5000 dataset by training autoencoders **only on normal
ECG sequences**, then flagging anomalies via reconstruction error. Includes a plain Dense
autoencoder baseline and a sequence-aware LSTM autoencoder.

See [`EXPLANATION.md`](EXPLANATION.md) for a full walkthrough of how the code and the detection
method work.

## Contents

- `ecg_anomaly_detection.ipynb` — full lab notebook: theory, code, plots, evaluation, conclusion.
- `main.py` — runnable end-to-end script version (same pipeline as the notebook).
- `src/data.py` — download, split, normalize (train-fitted), and prepare the ECG5000 dataset.
- `src/models.py` — Dense and LSTM autoencoder architectures.
- `build_notebook.py` — regenerates the notebook (outputs are not stored; run it to see results).
- `src/evaluate.py` — reconstruction error, thresholding, metrics, ROC/AUC.

## Dataset

[ECG5000](http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv) — 4998
heartbeats, 140 time steps each. Label `1` = normal, `2`-`5` = abnormal (collapsed to a single
binary "abnormal" class here). Downloaded automatically by the code, no manual steps needed.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

**As a script:**
```bash
python3 main.py
```
Trains both autoencoders, prints metrics, and saves loss curves, ROC plot and metrics.json
to `./output/`.

**As a notebook:**
```bash
jupyter notebook ecg_anomaly_detection.ipynb
# then: Kernel -> Restart & Run All
```
Or open it directly in [Google Colab](https://colab.research.google.com) (File -> Upload
notebook) and Runtime -> Run all — no local setup required.

## Method (short version)

1. Train an autoencoder to reconstruct ECG sequences, using **only normal heartbeats**.
2. At test time, run both normal and abnormal heartbeats through the model.
3. Abnormal heartbeats reconstruct poorly (the model never learned their shape) → high
   reconstruction error.
4. Threshold the error (`mean + std` of normal training error) to classify anomalies.

## Preprocessing and thresholds (no leakage)

Normal beats are split 80/20 (train / held-out). Min-max scaling is fitted on the normal
training split only and reused for all other data. The threshold per model is
`mean + std` of that model's reconstruction error on the normal training data; test labels are
never used for preprocessing or thresholding. Label convention: `1` = normal, `0` = abnormal
(positive class for precision/recall/F1/ROC).

## Results (actual run of `python3 main.py`, seed 42)

| Metric | Dense AE | LSTM AE |
|---|---|---|
| Threshold | 0.0187 | 0.0392 |
| Accuracy | 0.9752 | 0.8306 |
| Precision (abnormal) | 0.9714 | 0.9700 |
| Recall (abnormal) | 0.9976 | 0.8081 |
| F1 (abnormal) | 0.9843 | 0.8817 |
| AUC | 0.9894 | 0.9241 |

Dense outperforms LSTM in this run. Earlier reported numbers (≈97.7% for both) came from a
pipeline that scaled before splitting and were not reproduced after the fix; see
`EXPLANATION.md` Section 5. Exact numbers may vary slightly with hardware / library version
even with fixed seeds and deterministic ops, and LSTM results can vary notably with the seed.
Metrics are saved to `output/metrics.json`.

## Limitations and future work

Single seed and split; threshold is a simple heuristic; abnormal subclasses are merged; no
hyperparameter search; ECG5000 is clean and pre-segmented. Future: multiple seeds, per-class
analysis, validation-based thresholds, convolutional/variational autoencoders.

**Disclaimer:** experimental/educational project, not a clinical diagnostic system.

## License

MIT
