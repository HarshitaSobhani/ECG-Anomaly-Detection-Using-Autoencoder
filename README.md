# ECG Anomaly Detection using Autoencoders and LSTM Networks

Detects abnormal heartbeats in the ECG5000 dataset by training autoencoders **only on normal
ECG sequences**, then flagging anomalies via reconstruction error. Includes a plain Dense
autoencoder baseline and a sequence-aware LSTM autoencoder.

See [`EXPLANATION.md`](EXPLANATION.md) for a full walkthrough of how the code and the detection
method work.

## Contents

- `ecg_anomaly_detection.ipynb` — full lab notebook: theory, code, plots, evaluation, conclusion.
- `main.py` — runnable end-to-end script version (same pipeline as the notebook).
- `src/data.py` — download, normalize, and split the ECG5000 dataset.
- `src/models.py` — Dense and LSTM autoencoder architectures.
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
Trains both autoencoders, prints metrics for each, and saves loss curves + a combined ROC plot
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

## Results (example run)

| Metric | Dense AE | LSTM AE |
|---|---|---|
| Accuracy | 0.9775 | 0.9771 |
| Precision (abnormal) | 0.9742 | 0.9755 |
| Recall (abnormal) | 0.9976 | 0.9957 |
| F1 (abnormal) | 0.9857 | 0.9855 |
| AUC | 0.9905 | 0.9740 |

Numbers vary slightly run to run despite fixed seeds (TensorFlow's GPU/CPU nondeterminism in
some ops). See `EXPLANATION.md` for why these two architectures perform the way they do on this
dataset, and the notebook's Conclusion section for limitations.

## License

MIT
