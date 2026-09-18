# Hidden-test feedback and Experiment 1 (decision layer) — 19 September 2026

**Submission 1 (best active models) hidden scores:** Door 1.000, ACV 1.000, Rail **0.7448**, SHM
0.9725 → overall **0.9293**. The Rail nested CV (0.8232) was optimistic; the contiguous
file-number block validation (0.7654) tracked the hidden score far better, so block validation is
now the primary Rail selection protocol. The submitted ZIP, member checksums, active-model
manifests and scores are archived in `weights/submissions/2026-09-18-submission1-best/`.

**Experiment 1 — re-select only the decision layer of the unchanged ExtraTrees detector**
(run `2026-09-19-exp1-decision-layer`, `Rail Corrugation/code/experiment1.py`). Protocol: 5
contiguous file-number blocks × 3 rotations (offsets 0/27/54 files), references and detector
refit per outer block, decision parameters selected only on 3 inner contiguous blocks of each
training set, pooled macro-F1 over the 272 files per rotation.

| Decision rule (same model, same features) | Block-CV macro-F1 |
|---|---|
| Incumbent fixed global τ = 0.25 | **0.7715 ± 0.0113** |
| Global τ re-selected on inner blocks | 0.7730 ± 0.0051 (+0.0015, within noise) |
| Per-side thresholds (τ₁, τ₂) | 0.7544 ± 0.0302 |
| Global τ + side-margin δ | 0.7695 ± 0.0064 |

**Conclusion: the decision layer is not the bottleneck; no submission was spent on it.**
Diagnosis from the block-validation confusion matrix (first rotation): Normal 225/5/4, Side I
6/**7**/1, Side II 0/3/**21**. Side I recall is 7/14 and there are 4 wrong-side calls, while
Normal false alarms are only 9/234. The detector's maximum side probability for true Side I
files is p10 0.10 / p50 0.29 / p90 0.45, against Normal p90 0.13 — many Side I recordings simply
look Normal to the current representation, so lowering τ trades Side I recall directly for
Normal false alarms. Side II is detected confidently (p50 0.57). Side I files also cluster in
the middle of the file-number range (two of five blocks contain none), which is consistent with
the hidden test set being harder for Side I than stratified CV suggested.

Implications for the next experiments: gains must come from the **representation/model**, aimed
at (a) Side I sensitivity and (b) side attribution (3 Side II files were called Side I). The
decision rules and block protocol live in `rail_corrugation.decision`; inference honours an
optional `decision` entry in the artifact, and artifacts without one behave exactly as before.
