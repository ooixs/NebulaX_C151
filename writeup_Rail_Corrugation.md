# Rail Corrugation classification

## Summary

The Rail Corrugation subsystem classifies each one-second axle-box recording as `Normal`,
`Side I`, or `Side II`. The active product uses the September 19 forest ensemble:
75% legacy spectral plus SG random forest and 25% SG random forest. Each component has 800
trees fitted with seed 0. The primary forest uses `max_features=0.5` and minimum leaf size 2;
the global decision threshold is 0.20. The selected run is
`2026-09-19-legacy_sg_mf50_leaf2_ensemble-forest`, with user-reported leaderboard macro F1
**0.8261904762**. All 68 saved Rail predictions match that scored submission exactly.

The previous ExtraTrees product model achieved macro F1 **0.8232 +/- 0.0167** in repeated nested validation.
A later leakage-focused campaign evaluated the same model family under duplicate-safe blocks and
speed-shift checks. Its promoted configuration scored 0.809 in repeated stratified validation,
0.782 in duplicate-safe blocks, and 0.481 when transferred between speed bins. These results are
reported separately because they measure different generalisation conditions.

## Problem and data

There are 272 labelled training recordings and 68 unlabelled test recordings. Each CSV contains
10,000 samples collected over one second at 10 kHz:

- one digital speed-pulse channel;
- 64 vibration channels; and
- 64 shock channels.

The channels cover eight cars and eight sensor positions. Odd positions belong to Side I and even
positions to Side II. The training labels are strongly imbalanced: 234 Normal, 14 Side I, and 24
Side II.

The official metric is macro F1 over the three classes. It gives each class equal weight, which
prevents the large Normal class from dominating the evaluation.

## Data audit and assumptions

Every audited recording has the expected 10,000 by 129 shape, the same ordered headers, and finite
numeric values. Two exact duplicate families were found: `Train107.csv`/`Train115.csv` and
`Train165.csv`/`Train187.csv`. Duplicate-safe validation keeps the members of each family in the
same fold.

The speed pulse has two transitions per wheel tooth. With a 90-tooth wheel and diameter 0.85 m,
speed is decoded as:

```text
speed = transitions / 180 * pi * 0.85 metres per second
```

Speeds range from stationary to about 19.5 m/s. Every labelled fault occurs at 9.7 m/s or above,
while the Normal class also contains stationary and slow recordings. Speed is therefore a strong
confounder: a model can partly separate the classes from operating condition instead of rail
condition.

## Validation design

The independent unit is a complete recording. Both side rows derived from one file always stay in
the same fold. Normal-reference statistics and decision thresholds are fitted without the outer
validation labels.

Four checks are used:

| Check | Question answered |
|---|---|
| Repeated stratified folds | How well does the model interpolate across similarly distributed files? |
| Contiguous file-number blocks | Does performance survive possible acquisition adjacency and exact duplicates? |
| High-speed matched validation | Can the model separate classes when the large speed difference is reduced? |
| Speed-bin transfer | Can it generalise to speed regions not represented for a class during training? |

The first check is closest to the primary model-selection protocol. The others are stress tests,
not estimates of the organisers' test distribution.

## Exploratory findings and features

Corrugation produces repeated vibration patterns, so frequency and wavelength are more useful than
a raw time-series average. Wavelength features use `wavelength = speed / frequency` to describe a
spatial pattern consistently across speeds.

For each vibration channel, the pipeline extracts:

- Welch log power in eight frequency bands and seven wavelength bands;
- dominant wavelength and its prominence;
- spectral entropy and centroid; and
- RMS, kurtosis, crest factor, and peak-to-peak range.

Shock channels contribute RMS, kurtosis, and peak information. Channel values are compared with a
Normal reference from the same speed region, fitted inside the training fold. Features are then
aggregated by rail side using their mean, maximum, upper quartile, spread, and number of unusually
large channels.

Cross-rail features retain the difference between Side I and Side II. This is important because a
fault is localised by asymmetry as well as absolute vibration. In the raw training data, the mean
Side I/Side II RMS ratio is about 1.02 for Normal, 1.13 for Side I, and 0.78 for Side II.

## Methods tested

### Initial model comparison

| Method | Historical macro F1 | Finding |
|---|---:|---|
| Three-class Random Forest, probability argmax | 0.652 | Weak Side I recall |
| Three-class Random Forest with OOF probability scaling | 0.815 | Better, but used two class adjustments |
| Shared per-side Random Forest | 0.819 | Side formulation improved rare-class detection |
| Shared per-side ExtraTrees | 0.834 | Best initial result |

The historical 0.834 score tuned and evaluated the decision threshold on the same out-of-fold
probabilities. It is useful development history but is optimistically selected. It is superseded
by the nested results below.

The shared per-side design creates two rows per file. Each row describes one side and its contrast
with the other side. One binary detector is trained across both sides. At inference, scores below
the threshold produce `Normal`; otherwise the higher-scoring side is returned. This shares scarce
fault examples and uses one threshold for the three-class decision.

### Challenger findings

ExtraTrees was the only clear improvement over the first per-side Random Forest. LightGBM,
logistic regression, deeper or larger tree ensembles, probability averages, and window-level
augmentation did not produce a stable gain. Halving each recording into two windows reduced
performance because it shortened the spectral observation.

Ablations supported the final feature design. Removing cross-rail features reduced macro F1 by
about 0.055; removing own-side features reduced it by about 0.15. Both absolute evidence and
side-relative evidence are therefore useful.

## Current validation evidence

### Active product run

Run `2026-09-18-autoresearch` fitted references and thresholds through inner folds for each outer
split. Its repeated 5-fold by 3-repeat macro F1 was **0.823227 +/- 0.016735**. Fresh nested
partitions scored 0.803706, and contiguous file-number blocks scored 0.765362. The final
all-training threshold is 0.25.

### Leakage-focused robustness campaign

A separate experimental branch added immutable data manifests, exact-duplicate grouping,
inner-training threshold selection, and speed-specific diagnostics. The promoted `RC-E02`
configuration uses the same shared-side ExtraTrees approach and retains raw speed because every
tested removal or weighting alternative weakened grouped validation.

| Protocol for RC-E02 | Macro F1 |
|---|---:|
| Repeated stratified 5-fold, 3 repeats | **0.8085 +/- 0.0195** |
| High-speed matched | **0.7997** |
| Duplicate-safe contiguous blocks | **0.7817** |
| Speed-bin transfer | **0.4806** |

The repeated-stratified pooled per-class F1 values were 0.970 for Normal, 0.638 for Side I, and
0.818 for Side II. In duplicate-safe blocks they were 0.970, 0.552, and 0.824. The rare Side I
class remains the main difficulty.

Some challengers achieved a slightly higher stratified score but a lower blocked score. They were
not promoted. This prevents model selection from following the easiest split while ignoring the
more deployment-relevant grouping test.

## Final product method

The app currently loads the `2026-09-18-autoresearch` artifact, not the later experimental-branch
artifact. Its prediction path is:

1. Validate the recording shape and channel order.
2. Decode speed from the pulse transitions.
3. Extract vibration, shock, frequency, and wavelength features.
4. Compare the channels with speed-aware Normal references.
5. Aggregate features for each side and compute cross-rail contrasts.
6. Score both sides with the shared ExtraTrees detector.
7. Return Normal or the higher-scoring side using the fixed threshold.

The current test export contains 57 Normal, 5 Side I, and 6 Side II predictions. Their labels are
not public, so no test macro F1 is claimed.

## Limitations

- Only 14 Side I and 24 Side II recordings are labelled.
- Fault and speed are strongly associated in the training set. The low speed-transfer score shows
  that this remains a deployment risk.
- Exact duplicates and possible acquisition adjacency reduce the effective sample diversity.
- Repeatedly evaluating ideas on the same 272 files can still create model-selection optimism.
- A one-second classification does not locate the recording along the track. Operators must use
  their acquisition records to map a flagged file to a physical location.
- `Normal` means that this classifier did not detect its learned corrugation pattern. It does not
  rule out other rail defects.

## Reproduction pointers

- Feature pipeline: `Rail Corrugation/code/pipeline.py`
- Current nested research protocol: `Rail Corrugation/code/train.py`
- Prediction: `Rail Corrugation/code/predict.py`
- Spectral exploration: `Rail Corrugation/notebooks/explore.py`
- Historical adjacency check: `Rail Corrugation/notebooks/leakage_check.py`
- Leakage-safe campaign: branch `rail-corrugation/exp-20260918-reproducible-campaign`
- Official metric: `common/metrics.py`

## References

- Rail Corrugation Info Kit,
  `docs/references/Rail_Corrugation/Rail_Corrugation_Info_Kit.md`.
- Baasch et al. (2025), distance-wavenumber analysis of axle-box acceleration,
  [TRA 2024 proceedings](https://link.springer.com/chapter/10.1007/978-3-031-85578-8_42).
- De Rosa et al. (2024), separating corrugation from other axle-box excitations,
  [Applied Sciences](https://www.mdpi.com/2076-3417/14/19/8920).
- Hassanieh et al. (2023), rail-defect classification using axle-box acceleration,
  [International Journal of Rail Transportation](https://doi.org/10.1080/23248378.2023.2220112).
