# Structural Health Monitoring fatigue estimation

## Summary

The Structural Health Monitoring (SHM) task estimates cumulative fatigue damage from one dynamic
stress recording. The label-generation process is based on rainflow counting and Miner's rule, so
we used that physical structure before testing learned corrections.

The product counts stress cycles, applies a slope-five S-N damage model, and corrects the remaining
log residual with a small Huber regressor. Its selection validation score was **0.9797 +/- 0.0011**
under the official `1 - MAPE` metric. A fresh set of fold seeds scored 0.9807, compared with 0.9743
for the physics-only model on the same splits.

## Problem and data

The dataset contains 64 labelled training files and 16 unlabelled test files. Each file is one
headerless stress channel with 581,120 samples. Training damage labels range from about 0.029 to
0.928 and are all positive.

The required output contains one positive cumulative-damage estimate per file. The official score
is `max(0, 1 - MAPE)`, where MAPE is the mean absolute percentage error. Percentage error gives
small and large damage values comparable influence, so all calibration and model comparison use
the same objective.

The files are described as independent recording segments from two load conditions, AW0 and AW4.
File numbers are identifiers rather than a known time sequence.

## Validation design

Complete files are the independent units. Validation uses eight-fold cross-validation repeated
with three shuffle seeds. Rainflow conventions, S-N parameters, scale constants, and learned
residual models are selected using training files only.

The residual model was accepted only after it exceeded the physics baseline by more than the
predeclared 0.002 practical gain and repeated that advantage using fold seeds not used for its
selection. The confirmation still uses the same 64 labelled files; it reduces split-specific
luck but is not an independent dataset.

## Physical baseline

### Rainflow counting and Miner's rule

The stress sequence is converted into cycles with ASTM E1049 rainflow counting. Each cycle has a
stress range and a count of 0.5 or 1.0. With amplitude `a_i = range_i / 2`, the accumulated damage
statistic is:

```text
S_m = sum(count_i * a_i^m)
predicted damage = S_m / C
```

The exponent `m` controls how strongly large stress cycles dominate. The scale `C` is fitted to
minimise MAPE, not ordinary squared error. For a fixed `m`, this can be obtained from the weighted
median of `S_m / damage` with weights `S_m / damage`.

### Exploratory findings

The labels closely follow slope-five damage: `log(S_5)` and `log(damage)` have correlation 0.9994
and an approximately unit slope. The MAPE curve is sharp around `m = 5`; changing the exponent to
4.9 or 5.1 costs about 0.015-0.017 in score.

Across all 24 original validation folds, the selected convention was the same:

- `m = 5`;
- ASTM half-cycle treatment of the residue;
- no endurance cut-off; and
- no mean-stress correction.

The corrected physics-only validation score is **0.974120 +/- 0.000252**. The all-file scale is
approximately `C = 7.352e8`.

## Methods tested

| Method family | Validation result | Decision |
|---|---:|---|
| Slope-five Miner's rule | 0.9741 | Physics baseline |
| Fine S-N exponent search | 0.9751 | Gain below 0.002 floor |
| Bilinear S-N or power-law calibration | About 0.9747 in earlier runs | Extra complexity not justified |
| Goodman/Gerber mean-stress corrections | About 0.9735 | Rejected |
| Alternative residue handling | At or below baseline | Rejected |
| Ridge residual on a small feature set | 0.9696 in earlier runs | Rejected |
| Replicated non-physical SHM ensemble | 0.9605 | Rejected |
| Miner's rule plus Huber log-residual | **0.9797 +/- 0.0011** | Selected |

The replicated alternative SHM pipeline originally used a custom rainflow counter. Its cycle
energy disagreed substantially with the validated ASTM implementation, so its feature ideas were
recomputed with the standard counter before comparison. This kept the model comparison focused on
the feature and estimator choices rather than inconsistent cycle extraction.

## Learned residual correction

The physical baseline leaves small, structured multiplicative errors. The final model predicts
`log(true damage / physics damage)` from 32 per-file features, including:

- rainflow energy at several S-N exponents;
- stress-range distribution summaries;
- mean-stress variants;
- time-series percentiles and spread; and
- compact spectral summaries.

A Huber regressor with `alpha = 1` was selected. Huber loss limits the influence of the largest
residuals, which is appropriate for only 64 files. The correction is applied multiplicatively:

```text
final damage = physics damage * exp(predicted log residual)
```

Selection validation improved from 0.974120 to **0.979701 +/- 0.001083**. Median out-of-fold
absolute percentage error fell from 1.65% to 1.39%; the 90th percentile fell from 5.77% to 4.82%,
and the worst file fell from 13.90% to 7.78%.

Using three fresh fold seeds, the hybrid scored **0.980676** versus **0.974258** for the physical
baseline on the same splits. The hybrid was better in every repeat, and its gain of 0.006418
exceeded the 0.002 acceptance floor. Nearby Huber strengths, Ridge, LightGBM, ExtraTrees, feature
subsets, and probability-style averages did not produce a further accepted improvement.

## Final product method

The active artifact is from run `2026-09-18-sg-branch3`:

1. Read the first numeric stress channel without standardising it per file.
2. Extract ASTM rainflow cycles.
3. Calculate slope-five Miner damage with the calibrated scale.
4. Extract the fixed 32-feature residual vector.
5. Apply the Huber log-residual correction.
6. Return one positive damage estimate for each file.

The current test predictions range from 0.027894 to 0.823098. These values are unlabelled and do
not provide a test MAPE.

## Why this fits maintenance work

The output preserves the physical interpretation of accumulated fatigue: larger values represent
more estimated damage over a compatible recording period. Operators can use the estimates to
prioritise further engineering review among comparable recordings. The app deliberately does not
turn them into remaining-life estimates because the dataset does not provide material limits,
asset histories, or future loading.

## Limitations

- The Huber correction is fitted on only 64 files and is less interpretable than the Miner's rule
  baseline.
- Fresh-seed confirmation changes the partitions, not the underlying labelled files. Some
  model-selection optimism can remain.
- The two load conditions do not include structural fault examples.
- Damage values should not be added or compared across unrelated assets or recording durations
  without engineering context.
- The sampling frequency and material-specific S-N design class are not supplied, limiting a
  direct engineering interpretation of the calibrated constant.

## Reproduction pointers

- Rainflow and damage pipeline: `SHM/code/pipeline.py`
- Validation and residual research: `SHM/code/train.py`
- Prediction: `SHM/code/predict.py`
- Physical-model alternatives: `SHM/notebooks/autoresearch.py`
- Exploratory analysis: `SHM/notebooks/explore.py`
- Official metric: `common/metrics.py`

## References

- SHM Info Kit, `docs/references/SHM/SHM_Info_Kit.md`.
- ASTM E1049-85 rainflow counting through the
  [`rainflow` package](https://github.com/iamlikeme/rainflow/).
- Marsh et al. (2016), rainflow residue processing,
  [International Journal of Fatigue](https://doi.org/10.1016/j.ijfatigue.2015.10.007).
- de Myttenaere et al. (2016), optimisation under MAPE,
  [Neurocomputing](https://arxiv.org/abs/1605.02541).
