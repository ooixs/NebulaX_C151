# Rail Corrugation validation report

The immutable audit covered all 272 training and 68 test recordings. Every recording is
10,000 rows by 129 columns with identical ordered headers and finite numeric values. Two exact
duplicate training families were found (`Train107`/`Train115` and `Train165`/`Train187`) and kept
together in blocked validation. No source file was changed or excluded.

All decision thresholds below were selected on inner-training out-of-fold predictions and frozen
before outer-fold inference. This removes the optimistic threshold reuse in the earlier 0.834
claim. The reproduced nested result is therefore the current comparator.

| Experiment | Controlled change | Stratified 5x3 | High-speed matched | Duplicate-safe blocks | Decision |
|---|---|---:|---:|---:|---|
| RC-E01 | Speed only | 0.394 | 0.350 | 0.403 | Diagnostic |
| RC-E02 | Current shared-side features | 0.809 | 0.800 | 0.782 | Promote |
| RC-E03C | Remove raw speed | 0.825 | 0.817 | 0.736 | Reject |
| RC-E03D | Remove raw speed and bin | 0.811 | 0.802 | 0.767 | Reject |
| RC-E03E | Nearest speed reference | 0.811 | 0.802 | 0.767 | Reject |
| RC-E03F | Speed-matched weighting | 0.820 | 0.809 | 0.754 | Reject |
| RC-E04A | Position-level reference | 0.814 | 0.803 | 0.779 | Reject |

RC-E02 repeated-stratified macro F1 is 0.809 ± 0.019 across repeats. Its pooled confusion matrix
(three predictions per file) is `[[671,14,17],[10,30,2],[1,8,63]]` in the order Normal, Side I,
Side II. Per-class F1 is 0.970, 0.638, and 0.818. The duplicate-safe blocked confusion matrix is
`[[225,4,5],[5,8,1],[0,3,21]]`, with per-class F1 0.970, 0.552, and 0.824.

Speed-bin transfer is much harder (macro F1 0.481), especially for rare faults outside their
observed speed support. This is a deployment limitation, not an estimate of the official test
distribution. Raw speed remains in the frozen model because every removal/weighting alternative
regressed under grouped validation; it is explicitly documented as a confounding risk.

The campaign stopped after the confounding and reference-granularity rounds produced no eligible
challenger. This avoids repeatedly tuning to only 14 Side I and 24 Side II examples.
