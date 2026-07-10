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
3. Min-max scales every value into `[0, 1]` using the global min/max across the whole dataset,
   so the autoencoder doesn't have to deal with an arbitrary raw amplitude range.
4. Splits the **normal** sequences into a train/test portion. Abnormal sequences are **never**
   included in training — they only appear in the returned `test_data`/`test_labels`.

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
`140 → 64 → 32 → 16` (encoder) and `16 → 32 → 64 → 140` (decoder). It treats the 140 values as
an unordered feature vector — it has no built-in concept of "this value comes right after that
one." It can only learn statistical relationships between fixed positions in the vector.

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

| Metric | Dense AE | LSTM AE |
|---|---|---|
| Threshold | 0.0137 | 0.0234 |
| Accuracy | 0.9775 | 0.9771 |
| Precision (abnormal) | 0.9742 | 0.9755 |
| Recall (abnormal) | 0.9976 | 0.9957 |
| F1 (abnormal) | 0.9857 | 0.9855 |
| AUC | 0.9905 | 0.9740 |

On this particular dataset/run, the Dense AE actually edges out the LSTM AE slightly. ECG5000's
heartbeats are short, pre-segmented, and denoised, so the "shape" of a normal beat is largely
captured by *which values occur*, not just *their order* — reducing the LSTM's usual advantage.
On noisier, more temporally complex signals, the ordering-aware LSTM would be expected to pull
ahead more clearly (see the notebook's Conclusion section for the full theoretical argument).

## 6. `main.py` — orchestration

`main.py` strings the above pieces together end to end: load data → train Dense AE → train
LSTM AE → compute thresholds/metrics/ROC for both → save loss curves and ROC comparison plots to
`./output/`. It's the `.py`-script equivalent of running the notebook top to bottom.
