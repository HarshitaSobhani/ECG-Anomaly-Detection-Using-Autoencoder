# Explanation: How the Code Works & How Detection Works

This document walks through the codebase (`src/`, `main.py`, `ecg_anomaly_detection.ipynb`)
and explains the mechanics behind ECG anomaly detection with a Dense autoencoder.

> Experimental / educational project. Not a clinical diagnostic system.

## 1. The dataset and preprocessing (`src/data.py`)

ECG5000 is a CSV of 4998 heartbeats. Each row is 140 values (one heartbeat waveform) plus a
label in the last column: `1` = normal, `2`-`5` = different kinds of abnormal beats.

`load_dataset()` does, in this order:
1. Downloads the CSV from Google's storage bucket.
2. Separates the 140 values from the label, and converts the label to binary
   (`1` = normal, `0` = abnormal; abnormal subtypes 2-5 are merged).
3. Separates normal and abnormal sequences.
4. Splits the **normal** sequences (raw, unscaled) into 80% train / 20% held-out normal test.
   Abnormal sequences are **never** part of training.
5. Fits min-max scaling (`fit_min_max`) on the **normal training split only**. One global
   min and one global max are used, so the shape of each beat is preserved.
6. Applies those same min/max values (`min_max_scale`) to train, held-out normal and abnormal
   data. Test data is only transformed, never used to fit. Test values can fall slightly
   outside `[0, 1]`; they are not clipped. (An earlier version computed min/max on the whole
   dataset before splitting; this was corrected.)
7. Builds the test set: held-out normal (label 1) + all abnormal (label 0).

The fitted `(x_min, x_max)` is returned so the same scaling can be applied to any new ECG.

## 2. The model (`src/models.py`)

An **autoencoder** is a neural network trained to reconstruct its own input:
`decoder(encoder(x)) ≈ x`. The encoder compresses the input into a small **bottleneck**, and
the decoder expands it back. Because the bottleneck is much smaller than the input, the
network must learn the general pattern of what it was trained on.

`build_dense_autoencoder()` builds a fully-connected encoder/decoder:

`140 → 64 → 32 → 16 → 32 → 64 → 140`

- Hidden layers: ReLU. Output layer: sigmoid (inputs are scaled to about `[0, 1]`).
- Loss: MAE. Optimizer: Adam.
- Input and output shape: `(samples, 140)`.

The Dense Autoencoder treats the ECG as a fixed-length feature vector and does not explicitly
model temporal dependencies between consecutive time steps.

## 3. Why training on normal data only enables anomaly detection

- No abnormal heartbeat is shown to the model during training.
- Its weights therefore encode what normal heartbeats look like.
- At test time, abnormal beats are typically reconstructed worse, giving a higher
  **reconstruction error**: `mean(|input - reconstruction|)` per sequence
  (`src/evaluate.py: reconstruction_error`).
- Held-out normal beats come from the same distribution as training data, so they usually
  reconstruct well (low error).

This gives anomaly detection without labeled abnormal training examples. Labels are only used
to evaluate.

## 4. Threshold and classification (`src/evaluate.py`)

`pick_threshold()` computes `threshold = mean(train_errors) + std(train_errors)`, using only
the reconstruction errors of the **normal training data**. No test data, test reconstruction
errors or test labels are used, and the threshold is not tuned on the test set.

`classify()` labels a sequence anomaly (0) if `error > threshold`, else normal (1).

`compute_metrics()` reports accuracy, precision, recall, F1 and the confusion matrix with the
**abnormal class as positive** (`pos_label=0`).

`compute_roc()` uses the raw reconstruction error as the anomaly score (higher = more
anomalous), with abnormal as the positive class, and reports ROC/AUC, a threshold-independent
measure of separation.

## 5. Training configuration (`main.py`)

Seed 42 (Python, NumPy, TensorFlow, deterministic ops enabled), 100 epochs, batch size 128,
`validation_split=0.1` (the last 10% of the normal training data; no abnormal data).
Exact numbers can still vary slightly across hardware / library versions.

## 6. Actual results from a full run

From the executed `ecg_anomaly_detection.ipynb` (seed 42). A `python3 main.py` run on one CPU gave
very slightly different values (accuracy 0.9752, AUC 0.9894, saved to `output/metrics.json`), which
is normal hardware / library variation. Abnormal is the
positive class.

| Metric | Dense AE |
|---|---|
| Threshold | 0.0187 |
| Accuracy | 0.9756 |
| Precision (abnormal) | 0.9737 |
| Recall (abnormal) | 0.9957 |
| F1 (abnormal) | 0.9845 |
| AUC | 0.9892 |

Confusion matrix (rows = true, columns = predicted; order abnormal, normal): `[[2070, 9], [56, 528]]` (derived from the reported metrics and the 2663-beat test set, not printed by the notebook).

These come from the pipeline with scaling fitted on normal training data only. Earlier
documented numbers (Dense accuracy ≈ 0.9775, AUC ≈ 0.9905) came from the previous pipeline that
scaled before splitting; they are superseded by the table above.

## 7. Limitations

- The threshold is a simple heuristic computed from errors on the same normal data the model
  was fitted on, which can be slightly optimistic. A validation-based threshold would estimate
  the normal error distribution on data not used to fit the model.
- The test set is held-out normal beats plus all abnormal beats, so accuracy and precision
  depend on that mix. Recall, F1 and AUC are more informative.
- One seed and one split; no confidence intervals; no hyperparameter search.
- Abnormal subtypes are merged; per-class detection is not analysed.
- ECG5000 is clean and pre-segmented. Results are not clinical-grade.

## 8. Future improvements

Validation-based threshold, multiple seeds / cross-validation, per-class analysis,
convolutional or variational autoencoders, noisier real-world ECG.

## 9. `main.py` — orchestration

Seeds → load/split/scale → build Dense AE → train on normal data → reconstruction errors on
train and test → threshold from train errors → classify → metrics, ROC/AUC → save loss curve,
ROC plot and `metrics.json` to `./output/`.

