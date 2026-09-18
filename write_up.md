# Team C151 — PS3 Train Condition Monitoring: Write-up

This single write-up covers all four subsystems (Door, ACV, Rail Corrugation, SHM). Per the
submission spec, one copy sits at the top level of `Optional_Items/`.

## 1. Summary

| Subsystem | Task | Model shipped | Validation | Score on our validation |
|---|---|---|---|---|
| Door | Segment a continuous stream into door cycles, label each Normal / Abnormal resistance | Timestamp-gap segmentation → per-cycle current-profile features → RF + logistic-regression ensemble | 5 contiguous time blocks with inner blocked threshold selection | IoU-weighted F1 **1.000** (segmentation alone 1.000) |
| ACV | Rank 8 cars by refrigerant-leak likelihood | Peer-relative cabin-temperature deviation, plus a fixed-weight discharge-pressure tie-breaker where pressure telemetry exists | Leave-one-case-out over 6 cases | Rank-decay **1.000** (0.979 from temperature alone) |
| Rail Corrugation | Normal / Side I / Side II per 1-second recording | Speed-normalised spectral features → shared per-side ExtraTrees detector | 5-fold × 3 repeats with nested reference/threshold fitting, 272 files | Macro F1 **0.823227 ± 0.016735** |
| SHM | Cumulative fatigue damage per file | ASTM rainflow + Miner's rule `D = S₅/C` with a Huber(α=1) log-residual correction on 32 rainflow/spectral features | 8-fold × 3 seeds, 64 files; fresh-seed confirmation | 1 − MAPE **0.979701 ± 0.001083** (confirmation 0.980676 vs 0.974258 physics-only) |

These are the results of the `2026-09-18-autoresearch` rerun. Thresholds, reference profiles and
calibration constants were fitted without the corresponding outer validation labels. These are
training-data validation estimates, not independent test scores; standard deviations across
repeats are not confidence intervals. The fixed ACV physics rule was previously developed on
these same six cases, including just one rich-format pressure case.

Door and ACV stopped at the score ceiling. Rail and SHM each stopped after five consecutive
challengers failed to exceed both the incumbent CV standard deviation and the practical gain
floor (0.005 macro-F1 for Rail, 0.002 score for SHM). All four retained models were refitted on
all training data, with previous artefacts preserved under `weights/<Subsystem>/runs/`.
The README records the current trial results and distinguishes them from the earlier experiments
summarised below.

## 2. Working principles

1. **Judge metrics re-implemented exactly** (`common/metrics.py`) and unit-tested against the
   worked examples in each Info Kit. All model selection used the official metric end-to-end.
2. **No random-row splits.** Door: contiguous time blocks. ACV: leave-one-case-out. Rail:
   stratified K-fold on files. SHM: K-fold on files.
3. **Baseline first, challenger second; challenger ships only if it beats the baseline
   out-of-fold** by a margin larger than the fold-to-fold noise. Where a tuned variant only won
   in-sample (ACV weighted sum, 1.000 in-sample vs 0.958 nested-LOO) the baseline was kept.
4. **Physics before capacity.** The datasets are small (6 ACV cases, 14 Side I files, 64 SHM
   files); low-parameter, physically-motivated models were the default everywhere.
5. **An improvement phase per subsystem**, stopping after five consecutive candidates without a
   meaningful gain or when the result was already at its ceiling.

## 3. Door

**Data facts.** 17 columns exactly as listed in the Info Kit (header names differ slightly, so
columns are mapped by position). No car/door identifier columns exist despite the headers doc.
Within a cycle the controller logs every 20 ms; between cycles there is a 10–59 s silence.

**Segmentation.** Cutting the stream wherever Δt > 1 s reproduces all 110 labelled cycles exactly
(segmentation-only IoU-F1 = 1.000; every row of both Train and Test is assigned to a cycle). The
state-machine and change-point alternatives in the Methodology were therefore not needed. The
Info Kit's hint — "think about what changes at a cycle boundary" — pointed at exactly this.

**Classification.** Per cycle we compute basic current / voltage / back-EMF / position
statistics, the mean current in the uniform-speed phase (30–80 % of the cycle, "mid-travel"),
and excess-current z-scores against a median ± IQR reference profile of *Normal* cycles for the
same operation (Open / Close), fitted inside each fold. This follows Wei et al. (2020), who show
that resistance faults manifest as phase-specific current excess rather than peak current.

The data confirm it: mid-travel current is 186–195 mA (Close) / 203–221 mA (Open) for every
Normal cycle and ≥ 313 mA / ≥ 235 mA for every Abnormal one, while peak (locking) current,
duration and the controller's own opening/closing-time columns do **not** separate the classes.
RandomForest, LightGBM and an RF + logistic-regression ensemble all reach OOF IoU-F1 = 1.000 with
wide probability margins.

**Earlier improvement phase.** With CV saturated, effort went into robustness on the test stream. A margin
probe found 7 test cycles inside the gap between the training classes (Open 229–236 mA, Close
215–220 mA) on which LightGBM said Abnormal and RF / LR / SVM said Normal. Their excess-current
profiles show a single-bin start-up spike (z ≈ 12–38, also present in clearly-normal test cycles)
rather than the sustained z ≈ 60–175 excess across acceleration and travel that every training
abnormal has, and the test Open cycles form a continuum 213 → 236 mA before jumping to 335. We
judged them Normal and shipped the RF + LR ensemble with the decision threshold placed at the
centre of the zero-error OOF interval (0.33). **Test: 30 Normal / 8 Abnormal.**

The Normal call was then corroborated on an independent axis the classifier never used:
**switch-actuation timing**. In training, the DLS→DCS release gap separates Open classes
disjointly (Normal −0.14…−0.13 s vs Abnormal −0.20…−0.16 s) and time/position at DCS actuation
separates Close classes; the clear-abnormal test cycles reproduce the abnormal signature
(−0.20/−0.18 s; 3.35–3.41 s), while all 7 borderline cycles sit inside the Normal ranges — the
borderline Closes reach their switches faster (3.16–3.22 s) than any training cycle at all. Two
independent physical axes now support the same labels.

**Assumptions stated.** (a) Gaps > 1 s delimit cycles in the held-out stream as they do in Train
(verified: 37 gaps, 38 cycles, all rows assigned). (b) The abnormal signature is *sustained*
excess current in the travel phase, not a start-up spike.

## 4. ACV

**Data facts.** Five of six training cases (and the test case) carry 8 parameters per car; case 04
carries 63 including refrigeration pressures, and only cars 01–04 have any data in it. Columns are
parsed with the regex `Car (\d{2}) - (.+)` and mapped to roles by keyword, so both formats load
without hard-coded column lists. Readings of 0 °C and rows flagged `Invalid` are masked.

**Scoring.** At every timestamp, among cars in a cooling running mode, each car's indoor
temperature is compared with the median of its peers; the per-car mean of this deviation ranks
the cars. In every 8-parameter training case the faulty car sits +0.46 to +1.31 °C above its
peers while all healthy cars are within ±0.35 °C — the peer-relative frame recommended by the
FDD literature (Du, Domanski & Payne 2015) works because the other seven cars share the ambient
conditions and act as contemporaneous fault-free references. Leave-one-case-out: 0.979 (true car
1st in five cases, 2nd in case 04).

**What did not help.** A six-component weighted sum (setpoint error, persistence, a Ridge
healthy-response residual, etc.) reached 1.000 with weights chosen on all six cases but only 0.958
when the weights were chosen on the other five — classic small-sample overfitting, so it was not
shipped. Eight further temperature-only scorers (setpoint-adjusted, trimmed mean, Borda
aggregation, hottest-car fraction, daily-max deviation, high-load-only, leave-one-out median,
Full-Cooling fraction) tied or lost. Case 04 is genuinely ambiguous from cabin temperature: car 04
is warmer than the faulty car 01, running Full Cooling 91 % of the time with *elevated* pressures
on both sides — a different problem.

**What did help.** Refrigerant undercharge lowers discharge (high-side) pressure. Where the rich
telemetry exists, a fixed-weight (1.0) peer-relative pressure-deficit term is added: in case 04
the faulty car's high-side pressure is ~250 kPa *below* its peers and car 04's is 120 kPa
*above*, so the true car ranks first. The term has no fitted parameters, is insensitive to its
weight (0.5–2.0 all give 1.000) and is a no-op on 8-parameter files, so it cannot hurt the common
format. LOO 1.000. Caveat: validated on one rich-format case.

**Test prediction.** `01|04|03|07|08|06|02|05`. Car 01 leads on every indicator (mean deviation
+0.105 °C, p90 +1.0 °C, 18 min longest run above +1 °C) but the margin is smaller than in any
training case.

## 5. Rail Corrugation

**Data facts.** 272 one-second files at 10 kHz, 129 columns; 234 / 14 / 24 class split. The
speed pulse has symmetric high/low run lengths, i.e. two transitions per tooth:
`v = transitions / 180 · π · 0.85 m/s`. Speeds span 0–19.5 m/s, and **every fault file is at
≥ 9.7 m/s while Normal includes stationary and slow runs** — speed is a strong confounder.

**Features.** Per channel: Welch log band power in 8 frequency bands *and* 7 wavelength bands
(λ = v/f, 25 mm – 1 m), dominant wavelength and prominence, spectral entropy/centroid, RMS,
kurtosis, crest factor, shock RMS/kurtosis/peak count. Excess-over-reference z-scores are computed
against a Normal reference for the same speed bin, fitted inside each fold (Baasch et al. 2025,
De Rosa et al. 2024). Aggregated per side (positions 1,3,5,7 vs 2,4,6,8): mean, max, p75, std,
count of channels > +2 IQR; plus Side I − Side II differences, same-side dominant-wavelength
agreement, and per-car maxima to preserve local evidence. Raw asymmetry is already clear: RMS
ratio Side I / Side II is 1.02 (Normal), 1.13 (Side I), 0.78 (Side II).

**Earlier models.** A 3-class RandomForest reached 0.652 macro-F1 by argmax (Side I F1 0.235) and
0.815 with OOF-tuned class-probability scaling. The **shared per-side detector** — one binary
model trained on 544 (file, side) rows with own-side and side-relative features, decoded to
Normal / Side I / Side II with one threshold τ — reached 0.819 with RF and **0.834 ± 0.014 with
ExtraTrees** (Side I recall 10/14, Side II 20/24, Normal 228/234). These threshold-tuned OOF
scores include selection optimism and are superseded by the nested validation in the summary.

**Earlier improvement phase (20 variants on identical folds).** ExtraTrees was the only accepted gain.
Ablations were informative: dropping cross-rail (relative) features costs 5.5 points and dropping
own-side features costs 15, confirming both halves of the design (Hassanieh et al. 2023 on
left–right coupling). The legacy mirroring trial misaligned swapped rows and their binary
labels, so its degradation is not evidence of physical asymmetry and it is excluded from the
current research pipeline. LightGBM, logistic regression,
excess-only features, deeper trees, more trees, RF/ET/LGBM averages, 5-seed averaging, per-side
decision thresholds (+0.007, within noise) and window-level augmentation (2 × 0.5 s, −6.5 points:
halving the spectral record costs more than doubling the sample count gains) all failed the
acceptance rule; the stop criterion was met twice over.

**Current rerun and robustness checks.** References and thresholds are now refitted in an inner
3-fold loop for every outer fold: macro F1 is **0.823227 ± 0.016735**. Ratios/side identity,
paired-channel contrasts, compact features, compact RF and compact RBF-SVM all failed the
predeclared improvement rule, so the original ExtraTrees architecture was retained and retrained
with a final threshold of 0.25. Fresh nested partitions scored **0.803706**; contiguous
file-number blocks scored **0.765362**. Both side-rows always stay with their source file.
The spread demonstrates validation sensitivity, not a guaranteed interval for test performance.

**Test predictions.** 57 Normal / 5 Side I / 6 Side II of 68 (training prior implies ≈ 3.5 / 6).

## 6. SHM

**Data facts.** Each file is one headerless stress channel of 581,120 samples (~180 k rainflow
cycles, ≤ 26 residue half-cycles). Labels 0.029–0.928, no zeros.

**Model.** The Info Kit says the labels come from rainflow counting + Miner's rule with
σ^m · N = C, so this is a physics-recovery problem. ASTM E1049 rainflow → `S_m = Σ countᵢ ·
(rangeᵢ/2)^m` → `D̂ = S_m / C`. Because the judge metric is MAPE, `C` is fitted as the
MAPE-optimal scalar (weighted median of `S_m / D` with weights `S_m / D`; de Myttenaere et al. 2016),
not by log-space least squares. The corrected all-file fit is `C = 735168784.8379046`.

**Earlier calibration.** A grid of 1,968 conventions — m ∈ 3.0…7.0 step 0.1, residue as half / full / dropped cycles,
endurance cut-off ∈ {0, 0.5, 1, 2}, Goodman correction with σ_u ∈ {none, 100, 200, 400} — was
scored inside 8-fold CV × 3 seeds, choosing the simplest configuration within 0.002 of the best.
**All 24 folds selected m = 5.0, ASTM half-cycle residue, no cut-off, no mean-stress
correction.** CV score 0.9736 ± 0.0000; in-sample 0.9743; log(S₅) vs log(D) has correlation
0.9994 and slope 0.9965. The MAPE curve is sharply peaked: m = 4.9 or 5.1 costs 1.5–1.7 points,
m = 4.5 costs 10.6. m = 5 is the standard slope for welded steel details, which is reassuring.

**Earlier improvement phase (7 refinements).** Residual ratios spanned 0.95–1.16 and correlated
only weakly with series skewness. A finer m grid, bilinear S-N, Goodman/Gerber mean-stress,
closing the largest residue loop, a ridge residual model and a power-law recalibration all
landed within +0.0011 of that baseline, below the practical improvement floor.

**Current rerun.** Fixing the MAPE calibration weights raised like-for-like CV from **0.973615**
to **0.974120 ± 0.000252**. Five further families tested range-bin counts, finer bin counts,
alternative bin centres, fine S-N exponents and residue conventions. The strongest, fine
exponents, reached **0.975100**, but its +0.000980 gain did not clear the 0.002 floor. At that
point the retained model was m = 5, ASTM half cycles and no binning, cut-off or mean-stress
correction, with per-file CV APE median 1.65%, p90 5.77%, worst 13.90%.

**sg-experiments evaluation (shipped).** The sg-experiments branch contributed feature ideas
(multi-exponent log rainflow energies, Goodman variants, percentile spreads, spectral moments,
an AW0/AW4 load proxy). Its own custom rainflow disagrees with the ASTM `rainflow` package by up
to ~99% on range-energy sums, so all ideas were recomputed with the validated counter; the
branch's app ensemble replicated on corrected features scores 0.960466, below physics. Grouped
calibrations failed, but **S₅/C plus a Huber(α=1) log-residual on 32 such features** reached CV
**0.979701 ± 0.001083** and passed a fresh-seed confirmation (0.980676 vs 0.974258 physics-only,
better in every repeat, gain +0.0064 > the 0.002 floor). Per-file CV APE: median 1.39%, p90
4.82%, worst 7.78%. Shipped as `SHM/model/shm_model.joblib`; the physics-only JSON model remains
in the run archive and `predict.py` falls back to it when no joblib is present.

**Test predictions.** 16 values in 0.027894–0.823098 (training label range 0.029–0.928).

## 7. Deliverables and reproducibility

- `predictions/{door,acv,rail,shm}_predictions.csv` — produced by each subsystem's
  `code/predict.py` (`--input … --output …`), which the app also calls.
- `<Subsystem>/model/` — the single shipped artefact per subsystem; `weights/<Subsystem>/runs/`
  — versioned run folders with CV reports, trial results, model backups and prediction exports.
- `common/metrics.py`, `common/research.py` and `tests/` — judge metrics, plateau tracking,
  fold-isolation tests, archived-score replay and all four inference CLI smoke tests.
  Run `& ".\.venv\Scripts\python.exe" -m pytest tests -q` on Windows.
- Use each training CLI's `--research --ship --run-id <new-id>` mode to reproduce the current
  protocol. See the README for complete commands; never reuse an existing run directory.
- `README.md` — full Methodology (implementation spec) and Results (all tables above with
  per-variant numbers), reproduction commands, environment.
- Environment: Python 3.12 via `uv`, `requirements.txt`; CPU-only (an RTX 4090 was available but
  unnecessary at these data sizes).

## 8. Honest limitations

- **Door:** 7 borderline test cycles cannot be validated against ground truth; the Normal call
  rests on two independent physical axes (profile shape + switch timing) rather than CV evidence.
- **ACV:** the pressure tie-breaker rests on one rich-format case; the test case has the common
  format, where the temperature signal is weaker than in any training case.
- **Rail:** Side I has only 14 training examples. Nested CV is 0.8232 ± 0.0167, fresh-split
  confirmation is 0.8037, and contiguous-block validation is 0.7654. None is an independent
  labelled test score; repeatedly searching these files can still bias model selection.
- **SHM:** the shipped hybrid leaves about 2.03% validation MAPE. The Huber correction is fitted
  on only 64 files and is not interpretable physics; its fresh-seed confirmation reused the same
  64 files, so residual selection optimism cannot be fully excluded.
- **sg-experiments rail ideas** (per-car bilateral asymmetry features) were evaluated under the
  nested protocol and rejected: all five variants scored below the incumbent's 0.823227 mean.

## References

Wei et al. 2020 (*Urban Rail Transit* 6); Shimizu et al. 2022 (*IEEE Aerospace*); Dempster et
al. 2021 (MiniRocket, *KDD*); Du, Domanski & Payne 2015 (*Appl. Therm. Eng.* 98); Mattera et
al. 2018 (*Sensors* 18); Baasch et al. 2025 (*TRA 2024*); De Rosa et al. 2024 (*Appl. Sci.* 14);
Hassanieh et al. 2023 (*Int. J. Rail Transp.* 12); ASTM E1049-85 via `rainflow`; Marsh et al.
2016 (*Int. J. Fatigue* 82); de Myttenaere et al. 2016 (*Neurocomputing* 192). Full links in the
README.
