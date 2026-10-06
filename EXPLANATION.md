# Explanation: How the Code Works & How Detection Works

This document walks through the codebase (`src/`, `main.py`, `ecg_anomaly_detection.ipynb`)
and explains the mechanics behind ECG anomaly detection with autoencoders.

## 1. The dataset (`src/data.py`)

ECG5000 is a CSV of 4998 heartbeats. Each row is 140 time-step values (one heartbeat waveform)
plus a label in the last column: `1` = normal, `2`-`5` = different kinds of abnormal beats.

`load_dataset()`:
1. Downloads the CSV directly from Google's storage bucket.
2. Splits each row into a 140-value sequence + a binary label (`1` if original label was `1`,
   else `0`) — we collapse all abnormal subtypes into one class since the goal is binary
   normal-vs-anomaly detection, not multi-class diagnosis.
3. Splits the **normal** sequences (raw, unscaled) into 80% train / 20% held-out normal test.
   Abnormal sequences are **never** included in training — they only appear in `test_data`.
4. Fits min-max scaling (`fit_min_max`) on the **normal training split only**, then applies those
   same min/max values to train, held-out normal and abnormal data (no leakage). Test values can
   therefore fall slightly outside `[0, 1]`; they are not clipped. (An earlier version computed
   min/max on the whole dataset before splitting; this was corrected.)

This last point is the core trick of the whole project, explained in Section 3.

## 2. The models (`src/models.py`)

An **autoencoder** is a neural network trained to reconstruct its own input:
`decoder(encoder(x)) ≈ x`. The encoder compresses the input down to a small **bottleneck**
(a low-dimensional latent vector), and the decoder expands that bottleneck back out. Because
the bottleneck is much smaller than the input, the network is forced to learn only the most
important, generalizable patterns in the data — it physically cannot memorize every input
value independently.

### Dense autoencoder
`build_dense_autoencoder()` builds a plain fully-connected encoder/decoder:
`140 → 64 → 32 → 16` (encoder) and `16 → 32 → 64 → 140` (decoder). The Dense
Autoencoder treats the ECG as a fixed-length feature vector and does not explicitly model
temporal dependencies between consecutive time steps.

### LSTM autoencoder
`build_lstm_autoencoder()` instead treats the input as a genuine time series:
- **Encoder**: two stacked `LSTM` layers read the sequence step by step, maintaining a hidden
  state that gets updated at every time step. The final hidden state (size `latent_dim=16`) is
  the compressed representation — but unlike the dense model's bottleneck, this vector was built
  by processing the sequence *in order*.
- **RepeatVector**: copies that single latent vector 140 times, producing an input sequence the
  decoder LSTMs can consume one step at a time.
- **Decoder**: two more `LSTM` layers unroll the repeated latent vector back into a 140-step
  sequence, and a `TimeDistributed(Dense(1))` layer maps each time step's LSTM output to a
  single reconstructed value.

Both models are compiled with `loss="mae"` (mean absolute error) so their reconstruction errors
are directly comparable.

## 3. Why training on normal data only enables anomaly detection

This is the central idea of the whole project (also called "one-class" or "novelty" learning):

- We never show the model a single abnormal heartbeat during training.
- The autoencoder's weights therefore only encode *what normal heartbeats look like and how
  to reconstruct them accurately*.
- At test time, we feed the model heartbeats it may have never seen the shape of (abnormal
  ones). Since its internal representation has no capacity for those patterns, its
  reconstruction of an abnormal input will typically be worse — it "falls back" toward
  something resembling a normal heartbeat, producing a visible mismatch from the true abnormal
  input.
- We measure that mismatch as **reconstruction error**: `mean(|input - reconstruction|)` per
  sequence (`src/evaluate.py: reconstruction_error`).
- Normal test sequences (which the model never trained on directly, but which come from the same
  distribution as training data) still reconstruct well → low error.
- Abnormal test sequences → higher error.

This means we get anomaly detection **without ever needing labeled abnormal training examples**
— only enough labeled test data to validate the approach.

## 4. Picking a threshold and classifying (`src/evaluate.py`)

`pick_threshold()` computes `threshold = mean(train_errors) + std(train_errors)`, using only
the reconstruction errors of the **normal training data**. This keeps the threshold decision
free of any information from the labeled test set (no leakage) while still being grounded in
what "normal" reconstruction error looks like.

`classify()` then labels any test sequence with `error > threshold` as an anomaly.

`compute_metrics()` reports accuracy/precision/recall/F1 with the **abnormal class as positive**
(`pos_label=0`), since detecting anomalies is the actual goal — a model that misses abnormal
beats is the failure mode we care about most.

`compute_roc()` sweeps every possible threshold and plots true-positive-rate vs.
false-positive-rate for detecting abnormal beats, summarized by AUC — a threshold-independent
measure of how well each model's error distributions separate normal from abnormal.

## 5. Actual results from a full run

Produced by `python3 main.py` (seed 42, TensorFlow deterministic ops, CPU), saved in
`output/metrics.json`. Abnormal is the positive class.

| Metric | Dense AE | LSTM AE |
|---|---|---|
| Threshold | 0.0187 | 0.0392 |
| Accuracy | 0.9752 | 0.8306 |
| Precision (abnormal) | 0.9714 | 0.9700 |
| Recall (abnormal) | 0.9976 | 0.8081 |
| F1 (abnormal) | 0.9843 | 0.8817 |
| AUC | 0.9894 | 0.9241 |

Confusion matrices (rows = true, columns = predicted; order abnormal, normal):
Dense `[[2074, 5], [61, 523]]`, LSTM `[[1680, 399], [52, 532]]`.

**Dense vs LSTM.** In this run the Dense AE clearly outperforms the LSTM AE. The LSTM's final
validation MAE (0.0305) is also much higher than the Dense AE's (0.0130), i.e. the LSTM
reconstructs normal beats less well, so its `mean + std` threshold (0.0392) is high and it
misses many abnormal beats (recall 0.81). Earlier README numbers (LSTM accuracy
≈ 0.977, AUC ≈ 0.974) came from an older version (global scaling, non-deterministic run) and
were **not** reproduced after the leakage fix and determinism changes. LSTM training with `relu`
activations is sensitive to initialization, so results from a single seed are not conclusive; no
tuning was done to improve the numbers. We therefore make **no claim that LSTM beats Dense**.

## 6. `main.py` — orchestration

`main.py` strings the above pieces together end to end: load data → train Dense AE → train
LSTM AE → compute thresholds/metrics/ROC for both → save loss curves and ROC comparison plots to
`./output/`. It's the `.py`-script equivalent of running the notebook top to bottom.
