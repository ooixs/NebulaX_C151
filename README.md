# NebulaX-Hackathon — PS3: Train Condition Monitoring

Solution repo for [Problem Statement 3](https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/tree/main/PS3)
(rail vehicle condition monitoring). Four independent subsystems, each scored on its own
held-out test set and weighted 25% of the Overall Score:

| # | Subsystem | Task | Metric | Output file |
|---|---|---|---|---|
| 1 | Door | Segment a continuous stream into open/close cycles, label each `Normal` / `Abnormal resistance` | IoU-weighted F1 | `door_predictions.csv` |
| 2 | ACV | Rank the 8 cars from most- to least-likely to have a refrigerant leak | Linear rank-decay | `acv_predictions.csv` |
| 3 | Rail Corrugation | 3-class: `Normal` / `Side I` / `Side II` per 1 s recording | Macro F1 | `rail_predictions.csv` |
| 4 | SHM | Regress cumulative fatigue damage per file | max(0, 1 − MAPE) | `shm_predictions.csv` |

## Repository layout

Subsystem folders use the **exact names the submission spec requires** (`Door`, `ACV`,
`Rail Corrugation`, `SHM`, each with `code/` and `model/`) so they can be dropped into the
submission folder unchanged.

```
NebulaX/
├── write_up.md                  # Optional item: single write-up covering all subsystems
├── Door/              {code, model, notebooks}
├── ACV/               {code, model, notebooks}
├── Rail Corrugation/  {code, model, notebooks}      # space, not underscore - per spec
├── SHM/               {code, model, notebooks}
├── app/                         # Compulsory deliverable: single web app covering all subsystems
│   ├── pages/                   # one page per subsystem (upload -> predict -> view -> download)
│   └── assets/
├── common/                      # Shared code: data loaders, local re-implementation of the four
│                                # judge metrics, train/val split helpers, CSV writers
├── data/                        # Raw datasets (git-ignored). Mirror PS3/02_Datasets here:
│   ├── Door/                    #   Train.csv, Train_Segments_Answer.csv, Test.csv
│   ├── ACV/{Train,Test}/        #   acv_case_01..06.xlsx, Train_Labels.csv, acv_test_case.xlsx
│   ├── Rail_Corrugation/{Train,Test}/  # Train1..272.csv, Train_Labels.csv, Test1..68.csv
│   └── SHM/{Train,Test}/        #   train01..64.csv, Train_Labels.csv, test01..16.csv
├── docs/
│   ├── 01_Problem_Statement_3_Specifications.md   # top-level spec (deliverables, scoring)
│   ├── references/<Subsystem>/  # Info Kits + supporting docs (authoritative task definitions)
│   └── example_submission/      # the 4 sample *_predictions.csv showing the exact output schema
└── tests/
```

All 446 files of the upstream `PS3/` folder are mirrored here (`02_Datasets` → `data/`,
`03_References` → `docs/references/`, `04_Example_Submission` → `docs/example_submission/`);
verified byte-identical up to CRLF line endings.

## Environment

- **GPU:** NVIDIA GeForce RTX 4090, 24 GB, driver 595.79 / CUDA 13.2 — available but not
  required; every model in the Methodology runs on CPU at these data sizes.
- **Python is not installed** on the dev machine (no `python`, `python3` or `py`). Install Python
  3.11/3.12 and create a venv before starting; add a CUDA build of `torch` only if GPU MiniROCKET
  is wanted.

## Submission packaging (team name: `C151`)

The hand-in folder is assembled from this repo at packaging time and is **not** versioned
(`C151/`, `predictions.zip` and `demo_video.*` are git-ignored). Mapping:

```
C151/
├── demo_video.<mp4|mov>          <- recorded separately
├── predictions.zip               <- zip of the *_predictions.csv files produced by app/ (top level, no subfolders)
├── app/                          <- copy of app/  (+ common/ and the four */code folders it imports)
└── Optional_Items/
    ├── write_up.md               <- copy of write_up.md
    ├── Door/{code, model}        <- copy of Door/code, Door/model      (notebooks/ omitted)
    ├── ACV/{code, model}         <- copy of ACV/code, ACV/model
    ├── Rail Corrugation/{code, model}
    └── SHM/{code, model}
```

Omit any subsystem folder we did not attempt.

Conventions:
- `data/` is never committed; copy the `02_Datasets/<Subsystem>` folders from the problem-statement repo into it.
- Each `<Subsystem>/code/` exposes a `predict(input_path) -> DataFrame` entry point that the app calls,
  so the app is the only thing that generates submission CSVs (as the brief requires).
- Metric implementations in `common/` must match the Info Kit formulas exactly so local validation
  scores are comparable to the leaderboard.

## Methodology

Implementation spec for the four models. Read the subsystem's Info Kit in `docs/references/`
first — it is the authoritative task definition. Each subsystem section below gives: inputs and
outputs, the modelling pipeline, what to validate and how, and the acceptance criteria for
adopting a challenger over the baseline.

### Ground rules (apply to every subsystem)

1. **Metrics.** Implement the four judge formulas in `common/metrics.py` exactly as written in the
   Info Kits and unit-test them against the worked examples there. Every model-selection decision
   uses the official metric, computed end-to-end (e.g. Door is scored on the full held-out stream,
   not on pre-cut cycles).
2. **Splits.** Never split on random rows. Door: contiguous time blocks. ACV: leave-one-case-out.
   Rail: repeated stratified K-fold, grouped by file. SHM: file-level K-fold. Anything fitted from
   data (reference profiles, thresholds, weights, S-N constants, class thresholds) is refit inside
   each training fold.
3. **Baseline → challenger.** Build the baseline first, record its CV score, then build the
   challenger. Adopt the challenger only if it beats the baseline on the official metric across
   folds. Otherwise ship the baseline.
4. **Capacity.** Data is small everywhere (6 ACV cases, 14 Side I files, 64 SHM files). Default to
   low-capacity, physically-motivated models. Add capacity only where validation residuals show a
   systematic pattern.
5. **Interface.** `<Subsystem>/code/predict.py` exposes `predict(input_path: str | Path) ->
   pandas.DataFrame` returning the exact output schema for that subsystem. Trained artefacts live
   in `<Subsystem>/model/` and are loaded lazily. `app/` calls `predict` and writes the CSV.
   The same file also has an `argparse` entry point — `python predict.py --input <path> --output
   <csv>` — because the Info Kits reference a `predict.py --input/--output` CLI even though the
   top-level spec only mandates the app; the wrapper covers both readings. For Rail and SHM,
   `--input` may be a directory (one output row per file).
6. **Stack.** `pandas`, `numpy`, `scipy`, `scikit-learn`; `lightgbm` optional; `sktime` (for
   MiniROCKET); `ruptures`; `rainflow`; `openpyxl`; `streamlit` for the app.

---

### 1. Door — segmentation + per-cycle classification

**Input.** One CSV stream, 17 columns per row: timestamp (`Y-M-D-h-m-s-ms`, hyphen-separated, not
zero-padded), motor current (mA), motor voltage (10 mV), back-EMF, door opening time (0.1 s),
door closing time (0.1 s), close command, open command, DCSR, DCSL, DLSR, DLSL, door opened, door
locked, opening, closing, door position. Actual header names differ slightly from the Info Kit
(e.g. `Motor electrodynamic force` is the back-EMF column; `Door leaf position` is position) — map
by position, not by exact name. **There are no car / door identifier columns** in `Train.csv` or
`Test.csv`, despite the headers doc listing them. `Train_Segments_Answer.csv` gives `start_time`,
`end_time`, `operation` (Open/Close), `status` (Normal / Abnormal resistance), `n_rows` per true
cycle. Sizes: Train 18,036 rows / 110 cycles (80 Normal, 30 Abnormal resistance); Test 6,253 rows.

**Output.** `door_predictions.csv` with columns `start_time`, `end_time`, `prediction`; one row per
predicted cycle; timestamps in the native format or ISO.

**Metric.** IoU-weighted F1: predicted segments match true segments one-to-one (same label, IoU > 0,
greedy by IoU); credit per match = IoU; soft precision/recall → harmonic mean.

#### 1.1 Parsing

- Parse the timestamp into a `datetime64[ms]` column; keep the original string for output.
- Sort by time; assert monotonicity. Record the distribution of Δt between consecutive rows.

#### 1.2 Segmentation

Build a **segmentation harness** first: given predicted `(start, end)` pairs and the answer file,
compute IoU-F1 with labels ignored. Target ≈ 1.0 before touching classification.

Implement three segmenters and pick by harness score:

- **A — timestamp gaps.** Cut wherever Δt exceeds a threshold (scan thresholds; the Δt histogram
  should be bimodal if the controller only logs during activity). Cheapest; try first.
- **B — motion state machine.** A cycle is active while any of: |Δposition| > ε, current above
  its idle level, open/close command asserted, DCS*/DLS* switches transitioning. Start = first
  active row, end = last active row before ≥ N idle rows. Tune ε, idle level, N on the harness.
- **C — change-point detection.** `ruptures` PELT (`model="rbf"` or `"l2"`) on `[current,
  position]`, penalty tuned on the harness. Fallback if A and B are unreliable.

Boundary rule: do **not** trim start-up or locking activity aggressively — check on the answer file
whether the labelled segments include them, and match that convention exactly. Sloppy boundaries
lose IoU on every match.

Record `operation` (Open vs Close) per predicted segment using the sign of the net position change
or the command flags — needed for per-operation references below, not for output.

#### 1.3 Features per segment

Compute from Normal training cycles only a **reference current profile** per operation (Open and
Close separately; there are no door identifiers, so no per-door references): resample current onto
~20 bins of normalised position (fallback: normalised time), take median and IQR per bin.

Per-segment feature vector:

| Group | Features |
|---|---|
| Excess over reference | per-bin `(current − median) / IQR`; max, mean, integral of positive excess; bin index of max excess; longest run of bins above +2 IQR |
| Current stats | peak, mean, RMS, std, integral (charge), time above 50 / 75 / 90 % of peak |
| Motion | duration (rows and seconds), `door opening time` / `door closing time` columns, mean and min |velocity|, stall indicators (rows with |Δposition| ≈ 0 while current high), number of direction reversals |
| Electrical | mean voltage, mean back-EMF, electrical energy (Σ V·I·Δt), current-vs-back-EMF slope |
| Context | operation (Open/Close), door identifier if present |

Keep absolute current values and raw durations; do not amplitude-normalise a cycle to itself.

#### 1.4 Classifiers

- **Baseline.** `RandomForestClassifier(class_weight="balanced")` or shallow LightGBM on the
  feature table. Tune the decision threshold on the end-to-end metric, not on accuracy.
- **Challenger.** MiniROCKET (`sktime.transformations.panel.rocket.MiniRocketMultivariate`) on the
  resampled `[current, position, back-EMF]` profiles (fixed length, e.g. 128 samples) →
  `RidgeClassifierCV`. Compare on the same folds.

#### 1.5 Validation

- Split `Train.csv` into contiguous time blocks (e.g. 5 blocks of ~22 cycles); rotate one out.
  With only 30 abnormal cycles, check each block holds at least ~5 so per-fold scores are
  meaningful.
- Score the **full pipeline** (segment → classify) on the held-out block with `common.metrics`.
- Report: segmentation-only IoU-F1, classification precision/recall per class on true segments, and
  end-to-end IoU-F1. Acceptance: challenger must beat baseline on end-to-end IoU-F1 in ≥ 4/5 folds.

---

### 2. ACV — per-car anomaly ranking

**Input.** One `.xlsx` per case: 3 id columns (car model, train number, time) + per-car columns
named `Car <NN> - <parameter>`, sampled every 30 s. Most files have 8 parameters per car (setting
mode, running mode, control temp for cooling and heating, indoor / outdoor average temp,
load-halved status, information-valid status); one file has 60+. `Train_Labels.csv` gives the
faulty car per training case.

**Output.** `acv_predictions.csv` with `file_id`, `ranked_cars` — all cars in the file, most → least
likely faulty, `|`-separated, using the two-digit id exactly as in the headers (e.g. `03`).

**Metric.** `(n − (r − 1)) / n` where `r` is the rank of the true faulty car.

#### 2.1 Loading

- Read headers; regex `^Car (\d{2}) - (.+)$` to get `(car_id, parameter)`. Build a per-car long
  table. Never hard-code the column list.
- Map parameter names to canonical roles by keyword (`indoor`→`t_in`, `outdoor`→`t_out`,
  `cooling`+`control`→`t_set_cool`, `running mode`→`mode_run`, `setting mode`→`mode_set`,
  `load`→`load_halved`, `valid`→`info_valid`). Use only roles present in the file.
- Rows where `info_valid` indicates invalid data are masked for that car (not treated as a fault
  signal).

#### 2.2 Condition-matched peer features

For each timestamp and car, define the **peer set** = other cars in the same `mode_run` (and
`mode_set` if present). Compute robust z-scores against the peer median / MAD. Aggregate per car
over the file:

| Feature | Definition |
|---|---|
| `setpoint_err_mean`, `setpoint_err_p95` | `t_in − t_set_cool` while in cooling mode |
| `peer_dev_mean`, `peer_dev_p95` | robust z of `t_in` vs peer median, cooling mode |
| `cool_duty` | fraction of timestamps in cooling running mode |
| `cooldown_rate` | slope of `t_in` over the first K samples after entering cooling mode, averaged over episodes |
| `excess_persist` | longest run (in samples) with `peer_dev` > +2 |

#### 2.3 Healthy-response residual model

Fit on the **healthy cars of the training cases** (all cars except the labelled faulty one):

```
Δt_in(t→t+1) ~ Ridge(t_in, t_out, t_set_cool, mode_run one-hot, load_halved, time_since_mode_entry)
```

For every car, residual = observed Δt_in − predicted Δt_in. Positive residual in cooling mode =
cooling weaker than expected. Aggregate `resid_mean_cool`, `resid_p90_cool` per car.

#### 2.4 Scoring and ranking

Score per car = weighted sum of standardised components:
`setpoint_err_p95`, `peer_dev_p95`, `excess_persist`, `resid_p90_cool`, `−cooldown_rate`.
Start with equal weights. Sort descending → `ranked_cars`.

Weight tuning: grid over a small simplex (e.g. weights ∈ {0, 0.5, 1}) evaluated by
leave-one-case-out mean rank-decay. Keep the number of components ≤ 5.

#### 2.5 Validation

- Leave-one-case-out over the 6 training cases: fit the residual model and choose weights on 5,
  rank the 6th.
- Report the true car's rank in each case and the mean rank-decay score. Acceptance: mean score
  from condition-matched + residual features ≥ that of plain peer z-scores; otherwise ship the
  plain version.
- Sanity-check on the 60+-parameter file that the role mapping still yields the core roles.

---

### 3. Rail Corrugation — spectral features, side-relative, 3-class

**Input.** CSV, 10 000 rows (1 s at 10 kHz), 129 columns. Column 1: speed pulse (90-tooth wheel,
0/1 toggling, wheel diameter 0.85 m). Columns 2–129: `Car c, Position p` vibration then shock for
c = 1..8, p = 1..8, i.e. column index `1 + 2·(8·(c−1) + (p−1)) + {0: vib, 1: shock}`. Positions
1,3,5,7 = Side I; 2,4,6,8 = Side II. Labels in `Train_Labels.csv` (`filename`, `label`):
234 Normal / 14 Side I / 24 Side II.

**Output.** `rail_predictions.csv` with `file_id`, `prediction` ∈ {`Normal`, `Side I`, `Side II`}.

**Metric.** Macro F1 over the three classes.

#### 3.1 Speed decode

- Count transitions in column 1. Verify on a few files whether one tooth produces one or two
  transitions (plot the pulse train). Speed
  `v = transitions / (edges_per_tooth × 90) × π × 0.85 / Δt` [m/s].
- Compute both a file-level mean speed and a sliding-window local speed (e.g. 0.1 s windows). If
  local speed varies by more than ~5 % within the file, resample the acceleration channels to the
  distance domain (`x = ∫v dt`) before spectral analysis.

#### 3.2 Per-channel features (64 vibration + 64 shock channels)

Vibration channels:

| Domain | Features |
|---|---|
| Frequency | Welch PSD (nperseg 1024) → log band power in fixed bands (e.g. 20–50, 50–100, 100–200, 200–400, 400–800, 800–1600 Hz); spectral centroid; spectral flatness / entropy |
| Wavelength | map frequency to wavelength `λ = v / f`; log band power in third-octave-like λ bands across ~20–500 mm; dominant λ and its peak prominence |
| Time | RMS, kurtosis, crest factor, peak-to-peak |

Shock channels: RMS, peak count above k·σ, kurtosis, max.

#### 3.3 Normal-response reference

From **Normal training files only** (inside each fold): for every channel and speed bin (e.g.
5 km/h bins), compute the median and IQR of each log band power. Add **excess features** =
`(value − median) / IQR` for every band. Keep the raw features too.

#### 3.4 Side aggregation and cross-rail features

For each side (Side I = positions 1,3,5,7 across 8 cars = 32 vibration + 32 shock channels):
mean, max, 75th percentile, std, and count of channels with excess > 2, for every feature. Then:

- `side_diff_<band>` = Side I aggregate − Side II aggregate; `side_ratio_<band>` likewise.
- `same_side_agreement` = fraction of same-side channels whose dominant λ falls within ±10 % of the
  side's modal dominant λ.
- Per-car max excess (to keep local evidence — not every car need cross the defect within 1 s).

#### 3.5 Models

- **Baseline.** `RandomForestClassifier(class_weight="balanced", n_estimators=500)` or LightGBM
  with class weights on the file-level feature table. Tune per-class decision thresholds on
  out-of-fold probabilities to maximise macro F1.
- **Challenger.** Shared per-side detector: build one row per (file, side) with side-relative
  features (own side minus other side), label = 1 if that side is corrugated. Train one binary
  model on 544 rows. Decode per file: if both side scores < τ → `Normal`; else the higher-scoring
  side. Tune τ on out-of-fold macro F1.
- Optional augmentation for the challenger only: swap Side I ↔ Side II channels to double the
  rare classes, **after** verifying on Normal files that the two sides' feature distributions are
  statistically indistinguishable. Augmented copies stay in the same fold as their source file.

#### 3.6 Validation

- `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` on files. Reference statistics (3.3) and
  thresholds are computed inside each training fold.
- Report macro F1 (mean ± std), per-class precision / recall, confusion matrix. Acceptance:
  challenger must improve mean macro F1 by more than one std of the baseline.

---

### 4. SHM — rainflow counting + calibrated Miner's rule

**Input.** One CSV per equal-length time segment of dynamic stress from a measurement point.
Each file is a **single headerless column of 581,120 samples** (one channel, no timestamp; the
sampling rate is not stated). `Train_Labels.csv` gives `filename`, `damage`; training labels range
0.0286–0.928 with no zeros, so MAPE is well-defined. File numbers are random identifiers, not
chronology.

**Output.** `shm_predictions.csv` with `file_id`, `prediction` (a single positive number).

**Metric.** `max(0, 1 − MAPE)`, `MAPE = mean(|true − pred| / |true|)`.

#### 4.1 Inspect first

Confirm on all 80 files that the shape is uniform (1 column × 581,120 rows) and record the value
range and units (stress values are O(1), sign-alternating). Check whether the 16 test files share
the same length — if not, damage must be treated as per-file, not per-unit-time, exactly as the
labels imply.

#### 4.2 Damage model

For each file, run rainflow on the single channel (`rainflow.extract_cycles`, ASTM E1049-85; yields
`range, mean, count∈{0.5, 1.0}`). With amplitude `a_i = range_i / 2`:

```
S_m(file) = Σ_i count_i · a_i^m
D̂(file)   = S_m / C
```

Never standardise stress per file; never concatenate files.

#### 4.3 Calibration (minimise MAPE directly)

Parameters: `m` (grid 2–10, step 0.25), `C` (closed form given `m`: the MAPE-optimal scalar is the
weighted median of `S_m / D` with weights `1/D`), plus these **discrete conventions**, each a
switch evaluated in the grid:

| Switch | Options |
|---|---|
| Residue treatment | half cycles (ASTM default) / count residue as full cycles / repeat-and-recount |
| Cycle magnitude | amplitude (`range/2`) / range |
| Mean-stress correction | none / Goodman (`a_eq = a / (1 − mean/σ_u)` with σ_u fitted) |
| Endurance cut-off | none / ignore cycles with `a < a_0` (a_0 fitted) |

Objective for every candidate: `mean(|D − D̂| / D)` on the training fold. Use log-space least
squares only to initialise `m`. Select the simplest configuration within one std of the best.

#### 4.4 Residual model (only if 4.3 leaves systematic error)

If out-of-fold residuals correlate with an observable (e.g. RMS, kurtosis, mean stress level), fit a
small `Ridge` or shallow GBM predicting `log(D / D̂)` from rainflow-derived features
(`S_m` for m ∈ {3, 4, 5}, amplitude-histogram moments, RMS, peak count), with sample weights
`1/D`. Apply as a multiplicative correction.

#### 4.5 Fallback

If no physics configuration reaches a usable CV score, train LightGBM on the same rainflow-derived
features with `objective="regression_l1"` and sample weights `1/D`, target `D`.

#### 4.6 Validation

- `KFold(n_splits=8)` on files (repeat with 3 seeds). If a line / load-condition (AW0 / AW4) label
  can be inferred from the data, add a `GroupKFold` on it to check generalisation across
  conditions.
- Report `1 − MAPE` per fold and the per-file relative-error distribution. Acceptance for the
  residual model: it must reduce out-of-fold MAPE in every fold.

---

### References (implementation pointers)

- Wei S. et al. (2020). Subhealth diagnosis of door resistance from motor current — phase-wise
  envelope features. *Urban Rail Transit* 6. <https://link.springer.com/article/10.1007/s40864-020-00133-4>
- Dempster A. et al. (2021). MiniRocket. *KDD*. <https://arxiv.org/abs/2012.08791>
- Truong C. et al. (2020). Offline change-point detection (`ruptures`). <https://github.com/deepcharles/ruptures/>
- Du Z., Domanski P.A., Payne W.V. (2015). Fault effects vs operating conditions in vapour-compression systems. *Appl. Therm. Eng.* 98. <https://www.nist.gov/publications/effect-common-faults-performance-different-vapor-compression-systems>
- Mattera C.G. et al. (2018). Virtual-sensor residuals for HVAC fault detection. *Sensors* 18(11). <https://www.mdpi.com/1424-8220/18/11/3931>
- Baasch B. et al. (2025). ABA → distance–wavenumber domain with system-response removal. *TRA 2024*. <https://link.springer.com/chapter/10.1007/978-3-031-85578-8_42>
- De Rosa A. et al. (2024). Separating corrugation from other ABA excitations. *Appl. Sci.* 14(19). <https://www.mdpi.com/2076-3417/14/19/8920>
- Hassanieh W. et al. (2023). RF on ABA features; nullifying cross-rail coupled vibration. *Int. J. Rail Transp.* 12(4). <https://doi.org/10.1080/23248378.2023.2220112>
- iamlikeme/rainflow — ASTM E1049-85 rainflow counting. <https://github.com/iamlikeme/rainflow/>
- Marsh G. et al. (2016). Rainflow residue processing and its effect on damage. *Int. J. Fatigue* 82. <https://doi.org/10.1016/j.ijfatigue.2015.10.007>
- de Myttenaere A. et al. (2016). MAPE-optimal regression = weighted MAE with weights 1/|y|. *Neurocomputing* 192. <https://arxiv.org/abs/1605.02541>
