# Rail model experiments — 19 September 2026

These experiments keep the active models and the original submission unchanged.
Research outputs are under `weights/Rail Corrugation/runs/2026-09-19-model-experiments/`.
Only candidates passing the qualification gates are packaged in separate
`weights/submissions/2026-09-19-<candidate>/predictions.zip` files.

## Protocol

- Use the 272 labelled training recordings for model development; no test labels
  or hidden-test batch statistics are used.
- Screening: the same five contiguous file-number blocks as Experiment 1, with
  offsets 0, 27 and 54. Both side rows remain in the same fold.
- Within each outer training set, three inner contiguous blocks produce
  out-of-fold probabilities for threshold selection. Normal-reference statistics,
  MiniROCKET biases, feature scaling and classifiers are fitted inside their
  respective training folds.
- Incumbent: 800-tree ExtraTrees on legacy features, fixed threshold 0.25.
  The harness reproduces Experiment 1's 0.771522 mean and 0.011345 repeat SD.
- Confirmation: reserved block offsets 14, 41 and 68, plus a five-fold stratified
  check with seed 20260919. A shortlist is recorded before confirmation.
- Scores are macro-F1 on all pooled out-of-fold recordings per rotation, with all
  three classes always included. Per-block absent classes are not silently
  omitted from the metric.

The three rotations reuse the same recordings. Their standard deviation is a
partition-sensitivity measure, not a confidence interval or standard error.
Reserved partitions also reuse recordings and are not an independent holdout.
Adaptive candidate search introduces selection optimism even with nested
threshold selection. Filename order is a proxy for acquisition grouping, not
proof of true run boundaries or of the hidden distribution.

The first confirmation round qualified three models, but the SG Random Forest
with minimum leaf size two produced the same test labels as an existing candidate.
A second screening round therefore tested six additional classifier/settings
variants. Its shortlisted models reused the fixed confirmation partitions; these
partitions must not be described as untouched for that second round. Acceptance
gates were not relaxed. `followup_rationale.json` and
`confirmation_shortlists.json` record this sequence.

## Representations

1. **Full SG:** the complete feature extractor from local Git reference
   `origin/sg-experiments`, commit `fa07703e84d8148acff3543bcafa654d3bf63e4f`,
   copied into `sg_features.py`. Includes averaged-waveform PSD bands, ratios,
   differences, peak frequency, centroid, amplitudes and per-car bilateral
   measurements. Both side orientations are extracted so a shared side detector
   learns fault evidence without requiring separate minority-class models.
2. **Cross-car spectral consistency:** per-car mean channel powers, bilateral
   log-power differences, adjacent-car differences, spectral cosine similarity
   and dominant frequency/wavelength dispersion. Channel powers are averaged
   before car comparisons to avoid waveform phase cancellation.
3. **MiniROCKET:** 32 vibration channels per side, anti-aliased from 10 kHz to
   2.5 kHz, 1,680 requested kernels, training-fold standardization and a
   regularized logistic classifier (C=0.1). A second variant standardizes each
   recording/channel independently. This initial representation removes content
   above approximately 1.25 kHz and does not include shock channels; a negative
   result does not reject all possible learned representations.

The documentation gives no car spacing or sensor timing offsets. No delay-aligned
cross-car correlation is calculated, and no shared track-section overlap within
the one-second recording is assumed. Spectral consistency is an empirical
feature, not evidence of causal propagation.

## Qualification and submission format

Qualification requires a screening gain above max(0.01, incumbent repeat SD),
improvement on every screening rotation, a confirmation mean gain above 0.01,
improvement on every confirmation rotation, stratified regression no greater
than 0.02, and neither fault-class confirmation F1 dropping more than 0.03.
An ensemble must additionally beat its strongest component by 0.005 in both
screening and confirmation. Blend weights and thresholds are chosen using inner
predictions only. The small ensemble weight grid is 0.25, 0.5, 0.75.

Final models fit all training recordings. The final threshold (and blend weight)
is the median of the screening inner-selected settings. It is not fitted to
hidden scores. Each accepted folder contains the model, inference source
snapshot, validation/checksum manifest, individual CSVs and a flat ZIP with:

```text
door_predictions.csv
acv_predictions.csv
rail_predictions.csv
shm_predictions.csv
```

Door, ACV and SHM bytes come directly from the archived original submission.
Duplicate Rail label vectors, including the original predictions, are not
counted as new candidates. No candidate is uploaded or made active automatically.

## Running and reproducing

```powershell
.venv/Scripts/python.exe "Rail Corrugation/code/experiment_models.py" --stage screen
.venv/Scripts/python.exe "Rail Corrugation/code/experiment_models.py" --stage confirm --names <shortlist>
.venv/Scripts/python.exe "Rail Corrugation/code/experiment_models.py" --stage package
.venv/Scripts/python.exe "Rail Corrugation/code/predict_candidate.py" --model <candidate-folder>/rail_candidate.joblib --input data/Rail_Corrugation/Test --output <output.csv>
.venv/Scripts/python.exe -m pytest tests app/tests -q
```

Intermediate predictions and metrics are cached per candidate/protocol. Do not
change an existing candidate's configuration and reuse its name; use a new name
or a new run directory. Keep packaged `C151/` tests out of source test collection
because their duplicate module names conflict with the live app tests.
