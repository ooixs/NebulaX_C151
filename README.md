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
├── weights/<Subsystem>/         # Training outputs (git-ignored): checkpoints, per-fold CV models,
│                                # calibration tables, reference profiles. Same 4 subsystem names.
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
- **Python:** the existing `.venv` uses Python 3.12.14. On Windows, run it as
  `& ".\.venv\Scripts\python.exe"`; the system `python` command is not required.
  The current pipelines use CPU-based NumPy/scikit-learn and do not need PyTorch.

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

**`python package.py`** assembles a new `C151/` directory (or use `--dest <new-directory>`).
It refuses to overwrite an existing directory. The flat predictions zip, active model artefacts,
checksummed model manifests, shared `common/` modules and `rail_corrugation/` import bridge are
included. Both `app/` and `Optional_Items/` contain the dependencies needed by their prediction
CLIs. Only the selected model is packaged, not inactive checkpoints or raw/training data.
`requirements-runtime.txt` records the installed versions of the direct project dependencies;
`requirements.txt` is also copied. Missing app sources and demo video are still reported as
warnings: assembling an inference bundle does not make those compulsory deliverables complete.

**`pytest tests`** checks metrics, individual prediction CLIs, all four predictors sharing one
Python process, active-model switching and integrity checks, full-test-set prediction parity,
and packaged app/optional-code inference in isolated Python processes outside the checkout.
Packaged CLI and shared-backend outputs are compared with existing CSV exports, and zip members
are checked byte-for-byte. These tests exercise the inference backend, not a graphical app
(the app UI is still a separate deliverable).

### Shared inference and active models

An app should call the shared backend rather than load the SG branch's bundled models:

```python
from common.inference import predict

result = predict("Rail Corrugation", input_path)
```

The subsystem names are `Door`, `ACV`, `Rail Corrugation`, and `SHM`. Existing prediction CLIs
remain supported. Rail is imported as `rail_corrugation.pipeline` / `rail_corrugation.predict`,
not `code.pipeline`, which conflicts with Python's standard `code` module. The original
`Rail Corrugation/code/` submission directory is unchanged. A scoped legacy joblib reader maps
old `code.pipeline` references during loading without replacing any standard-library module
or rewriting trained models. Joblib is pinned to the tested 1.5.3 version because that reader
uses its numpy-aware unpickler internals. Model files must be trusted; this is not a sandbox
for arbitrary uploaded pickle/joblib files.

`<Subsystem>/model/active_model.json` selects the model by filename and SHA-256, with its research
run ID where available. All four retained models are explicitly activated. Publishing a model
updates this manifest atomically and preserves inactive alternatives, so switching SHM between
physics JSON and residual joblib cannot silently leave the wrong model active. Readers reject
missing, ambiguous or checksum-mismatched selections; caches refresh when the selected artifact
changes. Without a manifest, exactly one legacy candidate is accepted; two candidates require
explicit activation. One model snapshot is used throughout each batch.

To activate an existing, trusted checkpoint without retraining:

```powershell
& ".\.venv\Scripts\python.exe" -m common.artifacts --activate "SHM/model/shm_model.joblib" --run-id "2026-09-18-sg-branch3"
```

Activating or republishing does not regenerate CSV exports. Regenerate them through the app
before submission whenever weights change, then rerun the parity tests. The integration fixes
alone do not change any retained weights or prediction values.

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
   pandas.DataFrame` returning the exact output schema for that subsystem. Training writes
   everything it produces (per-fold models, checkpoints, calibration grids, reference profiles,
   CV reports) to `weights/<Subsystem>/`. Publishing selects the final artefact using
   `<Subsystem>/model/active_model.json`; inactive alternatives may remain locally, but packaging
   includes only the selected artefact and its manifest. `predict` loads it lazily, and the app
   calls the shared inference interface and writes the CSV.
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
weighted median of `S_m / D` with weights `S_m / D`), plus these **discrete conventions**, each a
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

## Results

### Current validated run — 18 September 2026

All four models were retrained under run ID `2026-09-18-autoresearch`. Model selection used
training data only; test CSVs were regenerated after the models were fixed.

| Subsystem | Validation used in this run | Retained score | Stopping reason |
|---|---|---|---|
| Door | 5 contiguous time blocks; thresholds selected by inner blocked CV | IoU-F1 **1.000000** in every outer fold | Metric ceiling |
| ACV | 6 leave-one-case-out cases; fixed physics weights, fold-local healthy-response fitting | Rank-decay **1.000000** | Metric ceiling |
| Rail Corrugation | 5-fold × 3 repeats; references and threshold calibration nested inside each outer training fold | Macro F1 **0.823227 ± 0.016735** | 5 consecutive non-improving challengers |
| SHM | 8-fold × 3 seeds; convention and scale selection confined to training files | 1 − MAPE **0.974120 ± 0.000252** | 5 consecutive non-improving candidate families (superseded below) |

These are validation estimates, not independent test scores. The standard deviations describe
variation between CV repeats, not confidence intervals. In particular, ACV has only six cases,
and its fixed pressure rule was previously developed using the single rich-format case.

**Stopping rule.** A candidate must improve the incumbent mean by more than the larger of its
CV standard deviation and a practical floor: 0.005 for Rail, 0.002 for SHM. Five consecutive
failures stop the search; a score of 1.0 stops it immediately. Small gains do not automatically
justify extra parameters, and this stopping rule does not establish a global optimum.

**Rail.** Retained the original 800-tree per-side ExtraTrees architecture. New ratio/side-identity
features, paired-channel contrasts, compact physical features, compact RF, and compact RBF-SVM
all failed the acceptance rule. Fresh nested splits (seed 2026, two repeats) scored **0.803706**;
contiguous-file block validation scored **0.765362**. The same saved probabilities reproduce the
old **0.833610** score when thresholds are tuned and scored on those same OOF predictions: the
lower current estimate reflects corrected validation, not evidence of a worse classifier.
The final all-training-data threshold is 0.25.

**SHM.** Corrected the MAPE-optimal scale: the weighted median of `S_m / D` needs weights
`S_m / D`, not `1/D`. On identical folds this raised the baseline from **0.973615** to
**0.974120**. Range-bin counts, finer bin counts, alternative bin centres, fine S-N exponents,
and residue conventions were tested. The best challenger, fine exponents, reached **0.975100**:
its +0.000980 gain is below the predeclared 0.002 floor. Retained `m = 5`, ASTM half cycles,
no binning/cut-off/mean-stress correction, and refitted `C = 735168784.8379046`.

**Artefacts.** Each `weights/<Subsystem>/runs/2026-09-18-autoresearch/` contains the trial log,
summary, configuration/version manifest, final model and prediction CSV. `before/` preserves the
previous models and reports. Rail additionally saves OOF predictions, nested split indices,
confirmation/block checks and fold feature tables; SHM saves per-trial damage matrices and OOF
predictions. Older top-level `weights/<Subsystem>/cv_*.json` reports remain historical records.
The retained model is copied to `<Subsystem>/model/`. Door, ACV and Rail prediction CSVs are
unchanged; SHM predictions were refreshed after recalibration (range 0.029057–0.821870).

Reproduce with a **new** run ID; existing run directories are never overwritten:

```powershell
$py = ".\.venv\Scripts\python.exe"
$run = Get-Date -Format "yyyyMMdd-HHmmss"
& $py "Door/code/train.py" --research --ship --run-id $run
& $py "ACV/code/train.py" --research --ship --run-id $run
& $py "Rail Corrugation/code/train.py" --research --ship --run-id $run --jobs 8
& $py "SHM/code/train.py" --research --ship --run-id $run --jobs 6
& $py -m pytest tests -q
```

Omit `--ship` to keep candidate artefacts without updating the models used by inference. Rail
and SHM build missing feature caches from raw training files; reused caches are fingerprinted
and sample-checked against raw files. Rebuild caches when training data or feature extraction
changes. Generate CSVs with the existing `predict.py --input ... --output ...` interfaces only
after selection.
Tests cover MAPE calibration, fold isolation, stopping rules, archived-score replay, model/export
consistency, full prediction coverage, and all four prediction CLIs. Use `--research` for the
current validation protocol; the older exploratory scripts and tables below are historical.

### sg-experiments branch evaluation — 18 September 2026

`origin/sg-experiments` (commits `8f17643`, `fa07703`) contributes per-subsystem feature
extractors, app-oriented pipeline wrappers, a 1,737-line Streamlit app with its own pre-trained
models, and an architecture PDF. Its modelling ideas were re-evaluated under the same harness,
folds, and stopping rules as the main run (run IDs `2026-09-18-sg-branch*`). The branch itself
was not merged; the ideas were re-implemented against the validated pipelines.

**Data-quality finding.** The branch's hand-rolled rainflow counter disagrees with the ASTM
E1049 `rainflow` package by up to ~99% on range-energy sums (S₃/S₅) over the training files, so
its app models were trained on unreliable fatigue features. All evaluations below recompute the
branch's feature ideas with the validated rainflow. The branch's Streamlit SHM ensemble
(Huber + shallow ExtraTrees on 13 features, log-damage target), replicated on corrected
features, scores **0.960466** — below the physics baseline.

**Rail (run `2026-09-18-sg-branch`, unpublished).** The branch's per-car bilateral asymmetry
statistics (per-car side diff/ratio, dominance counts, speed-normalised RMS) were added as
side-row features (`aggregate(paired=True)`, feature mode `cardom`). All five challengers lost
to the incumbent 0.823227 ± 0.016735: ET 0.815584, RF 0.807338, LightGBM 0.808575, ET
max_features 0.3 0.799603, ET min_samples_leaf 2 0.810536. The retained Rail model is unchanged.
The branch's remaining rail ideas (band-power ratios, 3-class ensembles with class-probability
multipliers) duplicate features or approaches that already lost in earlier phases.

**SHM (runs `2026-09-18-sg-branch{,2,3}`; final run published).** Fifteen candidates: the
replicated SG ensemble, per-load-condition and per-skew-sign grouped calibration (the branch's
AW0/AW4 `mean > 0` proxy — grouped C reached only 0.974511), and a family of physics-plus-residual
hybrids on 32 SG-style features (multi-exponent log rainflow energies, Goodman variants,
percentile spreads, spectral moments) computed with the validated rainflow. Champion:

- **`m = 5` Miner's rule + Huber(α = 1) log-residual correction** — selection CV
  **0.979701 ± 0.001083** vs physics-only 0.974120; per-file CV APE p50 1.39% / p90 4.82% /
  worst 7.78% (physics-only: 1.65% / 5.77% / 13.90%).
- Neighbouring candidates (α ∈ {0.3, 3, 10}, ridge, LightGBM, ExtraTrees, averages, core-13
  subset) did not clear the acceptance rule; the search stopped at the plateau.
- **Fresh-seed confirmation** (seeds 100–102, same protocol): champion **0.980676** vs baseline
  0.974258 on identical splits — better in every repeat, gain +0.006418 > 0.002 floor. Shipped.

The shipped artefact is now `SHM/model/shm_model.joblib` (`ResidualDamageModel`: rainflow →
S₅/C base × exp(Huber correction) on 32 per-file features); the previous `shm_model.json` is
preserved under the run's `before/` folder. `predict.py` supports both formats, with the active
artifact selected explicitly by `active_model.json`. `predictions/shm_predictions.csv` was regenerated (range
0.027894–0.823098). Caveats: the correction is fitted on 64 files; the confirmation used fresh
splits but the same 64 files, so some selection optimism can remain; the residual model is not
interpretable physics — the physics-only model remains available in the run archives.

**Door / ACV.** Both are at their validation ceilings; the branch adds app wrappers around
equivalent scoring ideas (its ACV scorer similarly combines tracking error, fleet-relative
deviation, and a heavily weighted pressure asymmetry), so nothing was adopted.

**Not merged (out of scope for results):** the Streamlit app (compulsory deliverable `app/` is
still empty on main — the branch's app predicts with its own bundled models, including the
miscalibrated SHM ensemble, so integration should rewire it to the `<Sub>/code/predict.py`
interfaces), the app model binaries, the feature CSV exports, and the architecture PDF.

### Hidden-test feedback and Experiment 1 (decision layer) — 19 September 2026

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

### Earlier Door experiments

**Segmentation.** Inter-row Δt within a cycle is a constant 20 ms; between cycles it is 10–59 s.
Cutting at Δt > 1 s reproduces all 110 labelled cycles exactly — **segmentation-only IoU-F1 =
1.000** — so the final score is entirely down to classification. Test.csv splits into 38 cycles
(18 Open / 20 Close) the same way, with every row assigned.

**Classification.** Features per cycle: basic current/voltage/back-EMF/position statistics,
uniform-speed-phase ("mid-travel", 30–80 % of the cycle) current, and excess-over-reference
z-scores against per-operation Normal profiles fitted inside each fold. Threshold chosen on
out-of-fold end-to-end IoU-F1.

| Model | CV (5 contiguous time blocks) | OOF end-to-end IoU-F1 | OOF margin (normal max / abnormal min) | Threshold |
|---|---|---|---|---|
| RandomForest (500 trees, balanced) | per-fold 1.000 × 5 | 1.000 | 0.09 / 0.75 | 0.43 |
| LightGBM (300 × 7 leaves, balanced) | per-fold 1.000 × 5 | 1.000 | 0.00 / 1.00 | 0.50 |
| RF + logistic-regression ensemble — **shipped** | per-fold 1.000 × 5 | **1.000** | 0.10 / 0.55 | 0.33 |

Threshold = centre of the zero-error OOF interval (not the lowest grid value that scores 1.0).

Sanity check (not leakage): the separating physics is the mid-travel (uniform-speed phase)
current: 186–195 mA (Close) / 203–221 mA (Open) for every Normal cycle vs ≥ 313 mA (Close) /
≥ 235 mA (Open) for every Abnormal one. Peak (locking) current, cycle duration and the
controller's own opening/closing-time columns do **not** separate the classes, so a naive "peak
current threshold" would fail here.

**Improvement phase.** CV is saturated (1.000 with wide margins), so research went into
robustness on the *test* stream instead of the metric:

1. *LightGBM vs RF* — identical CV; no change.
2. *4-feature logistic regression* (`mid_mean`, `trav_mean`, `exc_mid_mean`, operation) — OOF
   0.982; not better alone, but a useful second opinion.
3. *Test-stream margin probe* — 7 test cycles sit in the gap between the training classes
   (Open at 229–236 mA, Close at 215–220 mA) where the models disagree: LightGBM flags all 7,
   RF@0.5 / LR / SVM flag none. Their excess-current profiles show a start-up spike in one
   bin only (z ≈ 12–38, also present in the clearly-normal test cycles) rather than the sustained
   z ≈ 60–175 excess across the acceleration and travel phases that every training abnormal
   cycle has, and the test Open cycles form a continuum 213→236 mA with the next value at 335 —
   so these are judged Normal.
4. *RF + LR ensemble with centred threshold* — same CV, agrees with the profile evidence on the
   contested cycles, less sensitive to any single model's extrapolation. **Shipped.**

Stopped early: the metric cannot improve and the remaining uncertainty (7 borderline test
cycles) is irreducible from the current-profile features alone.

**Switch-timing corroboration** (`Door/notebooks/switch_timing.py`). The DCS/DLS switch actuation
times — an axis the classifier does not use — separate the training classes cleanly and validate
on the clear-abnormal test cycles, giving independent evidence on the borderline 7:

| Feature | Train Normal | Train Abnormal | Clear-abnormal test | Borderline 7 |
|---|---|---|---|---|
| Open: DLS→DCS release gap (s) | −0.14…−0.13 | −0.20…−0.16 (disjoint) | −0.20…−0.18 | **−0.14…−0.13** |
| Close: time to DCS actuation (s) | 3.22–3.38 | 3.34–3.43 | 3.35–3.41 | **3.16–3.22** |
| Close: position at DCS actuation | 8–11 | 11–19 | 16–19 | **7–12** |

All 7 borderline cycles carry the Normal switch signature — the borderline Closes actually reach
their switches faster than any training cycle — so the Normal call stands on two independent
physical axes, not on profile-shape reasoning alone.

**Test predictions.** `predictions/door_predictions.csv`: 38 rows, 30 Normal / 8 Abnormal
resistance (3 Open at 335–378 mA, 5 Close at 441–542 mA).

### Earlier ACV experiments

**Loading.** Header regex `Car (\d{2}) - (.+)` + keyword role mapping handles both formats: the
8-parameter files (`Indoor Average Temperature`, `ACV Control Temperature (Cooling)`, `ACV Running
Mode`, …) and the 63-parameter `acv_case_04.xlsx` (`Passenger Cabin Temperature Detected Value`,
`Target Temperature Value`, plus refrigeration pressures). Readings of 0 °C and rows flagged
`Invalid` are masked. In case 04 only cars 01–04 carry any data; 05–08 are empty and rank last.

**Scoring.** Per timestamp, among cars in a cooling running mode, each car's indoor temperature is
compared with the median of its peers; per car this is aggregated to `peer_dev_mean`. Cars are
ranked by that single number (robust-z standardised). Six candidate components (setpoint error
p95, persistence, positive-deviation fraction, a Ridge healthy-response residual, and a
discharge-pressure deficit available only in the rich format) were evaluated as a weighted sum
over a 3^6 grid.

| Scorer | Leave-one-case-out rank-decay (6 cases) | True car rank per case |
|---|---|---|
| `peer_dev_mean` only (baseline) | 0.979 | 1, 1, 1, **2**, 1, 1 |
| `peer_dev_mean` + fixed-weight pressure deficit (see improvement phase) — **shipped** | **1.000** | 1, 1, 1, 1, 1, 1 |
| Weighted 6-component sum, weights chosen on the other 5 cases (nested LOO) | 0.958 | 1, 1, 1, 2, 2, 1 |
| Same, weights chosen on all 6 cases (in-sample, optimistic) | 1.000 | — |

The tuned combination does not beat the untuned baseline once weight selection is held out
(`ACV/code/train.py` encodes this acceptance rule). The one temperature-only miss is case 04,
where car 04 is genuinely warmer than the faulty car 01 (car 04 runs Full Cooling 91 % of the time
with *elevated* pressures on both sides — a different problem), so the labelled car is 2nd.

**Improvement phase.** Nine alternative scorers, all with no fitted weights, evaluated LOO:

| # | Scorer | LOO |
|---|---|---|
| 1 | Setpoint-adjusted peer deviation (`t_in − t_set` vs peers) | 0.979 (tie) |
| 2 | Peer deviation during high outdoor temperature only | 0.813 |
| 3 | Borda rank aggregation of 4 indicators | 0.958 |
| 4 | Trimmed-mean peer deviation | 0.979 (tie) |
| 5 | Deviation of daily maximum temperature | 0.896 |
| 6 | Fraction of timestamps as the hottest car | 0.979 (tie) |
| 7 | Deviation vs median of the *other* cars | 0.979 (tie) |
| 8 | Peer deviation + 0.5 × Full-Cooling fraction | 0.792 |
| 9 | Peer deviation + **discharge-pressure deficit vs peers** (fixed weight 1.0; active only when the rich-format pressure columns exist) | **1.000** |

Cabin temperature alone cannot separate case 04 (five consecutive non-improvements → stop rule
met). Scorer 9 adds the direct physical signature of undercharge — the faulty car's high-side
pressure is ~250 kPa *below* its peers while car 04's is 120 kPa *above* — and is insensitive to
its weight (0.5–2.0 all give 1.000). It has no fitted parameters and is a no-op on 8-parameter
files, so it cannot hurt the common format. **Shipped**, with the caveat that it is validated on a
single rich-format case.

**Signal strength.** In every 8-parameter training case the faulty car's mean deviation from its
peers is +0.46 to +1.31 °C while all healthy cars sit within ±0.35 °C.

**Test prediction.** `predictions/acv_predictions.csv`: `01|04|03|07|08|06|02|05`. Car 01 leads on
every indicator (mean deviation +0.105 °C, p90 +1.0 °C, longest run above +1 °C = 36 samples =
18 min, highest setpoint error) but the margin is smaller than in any training case.

### Earlier Rail Corrugation experiments

**Speed decode.** The pulse train has symmetric high/low run lengths, so each tooth produces two
transitions: `v = transitions / 180 · π · 0.85 m/s`. Speeds span 0–19.5 m/s. **Every fault file
is at ≥ 9.7 m/s, while Normal files include stationary and slow runs** — speed is a strong
confounder, which is why all excess features are computed against a Normal reference in the same
speed bin (0–3, 3–6, 6–9, 9–12, 12–15, 15+ m/s), fitted inside each fold.

**Features.** Per channel (64 vibration + 64 shock): Welch log band power in 8 frequency bands and
7 wavelength bands (λ = v/f, 25 mm–1 m), dominant wavelength and prominence, spectral entropy and
centroid, RMS, kurtosis, crest factor, shock RMS/kurtosis/peak count, and excess-over-reference
z-scores. Aggregated per side (mean, max, p75, std, count of channels > +2 IQR), Side I − Side II
differences, same-side dominant-wavelength agreement, per-car max excess. Side asymmetry is clean
in the raw data: RMS ratio Side I / Side II is 1.02 (Normal), 1.13 (Side I), 0.78 (Side II).

**Models.** 5-fold stratified CV × 3 repeats (272 files), reference + thresholds refit per fold.

| Model | OOF macro F1 (mean ± std over repeats) | Per-class F1 (Normal / Side I / Side II) |
|---|---|---|
| RF 3-class, argmax | 0.652 ± 0.015 | 0.961 / 0.235 / 0.810 |
| RF 3-class, minority-probability scaling tuned on OOF | 0.815 ± 0.018 | — |
| Shared per-side RF detector (544 side-rows, side-relative features), τ = 0.33 | 0.819 ± 0.009 | 0.972 / 0.615 / 0.851 |
| Shared per-side **ExtraTrees** detector (800 trees), τ = 0.28 — **shipped** | **0.834 ± 0.014** | 0.981 / 0.690 / 0.840 |

Confusion (ExtraTrees side detector, OOF): Normal 228/2/4, Side I 3/**10**/1, Side II 2/2/**20**.
The per-side design clears the acceptance rule against the argmax baseline (> 1 std) and beats the
probability-scaled 3-class baseline while having one tuned parameter instead of two. Side I recall
went from 2/14 (3-class) to 10/14.

**Improvement phase.** 13 variants on identical folds (5 × 3 repeated stratified):

| # | Variant | Macro F1 |
|---|---|---|
| 0 | RF per-side detector (initial) | 0.819 ± 0.009 |
| 1 | **ExtraTrees per-side detector** | **0.834 ± 0.014** |
| 2 | LightGBM per-side | 0.794 |
| 3 | RF on side-*relative* features only | 0.685 |
| 4 | RF on *own-side* features only (no cross-rail) | 0.779 |
| 5 | Logistic regression | 0.737 |
| 6 | RF, min_samples_leaf 3, max_features 0.3 | 0.818 |
| 7 | RF on excess-over-reference features only | 0.758 |
| 8 | RF + LightGBM average | 0.810 |
| 9 | ET max_features 0.3 | 0.823 |
| 10 | ET min_samples_leaf 2 | 0.821 |
| 11 | ET 2000 trees | 0.829 |
| 12 | ET + RF average | 0.828 |
| 13 | Legacy row-swapping experiment (incorrect label alignment; invalid comparison) | 0.784 |

| 14 | ET rerun (reference for phase 3) | 0.834 ± 0.014 |
| 15 | Per-side thresholds (τ₁, τ₂) | 0.840 ± 0.011 |
| 16 | 5-seed averaged ET | 0.834 ± 0.017 |
| 17 | 5-seed ET + per-side thresholds | 0.839 ± 0.012 |
| 18 | Window augmentation (2 × 0.5 s, window-averaged inference) | 0.769 ± 0.004 |
| 19 | Windows + per-side thresholds | 0.797 |
| 20 | (File + window probabilities)/2 + per-side thresholds | 0.825 |

Variants 9–12 and 15–20 fail the historical acceptance rule (> baseline + 1 std = 0.848); the best
Side I-targeted idea (per-side thresholds, +0.007) is within fold noise, and window augmentation
*hurts* — halving the spectral record costs more than doubling the sample count gains. Ablations
support retaining both feature groups: dropping cross-rail features costs 5.5 points, dropping
own-side features 15. The legacy mirroring experiment swapped rows without consistently swapping
their binary labels; its degradation does **not** establish physical side asymmetry. That
experiment is not used by the current `--research` pipeline.

**Adjacency-leakage check** (`notebooks/leakage_check.py`). File numbers may encode acquisition
order, so a 5-fold CV over *contiguous file-number blocks* was run as a stress test:
**block-CV macro F1 = 0.778** (0.965 / 0.545 / 0.824 per class) vs 0.834 stratified. Part of the
gap is mechanical — fault files cluster mildly in numbering (two blocks contain zero Side I, so
the hardest folds train on 8–10 of the 14 Side I examples), and τ is a single global tune — but
some may be genuine adjacency correlation. Honest held-out expectation is therefore
**≈ 0.75–0.85**, not a point estimate at 0.83.

**Test predictions.** `predictions/rail_predictions.csv`: 57 Normal / 5 Side I / 6 Side II
(training prior implies ≈ 3.5 / 6 of 68).

### Earlier SHM experiments

**Data.** 64 train + 16 test files, each a single stress channel of 581,120 samples (values
roughly −20…+40, ~180 k rainflow cycles per file, ≤ 26 residue half-cycles). Labels 0.029–0.928.

**Model.** ASTM E1049 rainflow (`rainflow.extract_cycles`) → `S_m = Σ countᵢ · (rangeᵢ/2)^m` →
`D̂ = S_m / C`. `C` has a closed-form MAPE-optimal solution (weighted median of `S_m / D` with
weights `S_m / D`). The historical scores below used the earlier scale fit; the corrected rerun is summarised above.
A grid of 1,968 conventions — `m` ∈ 3.0…7.0 step 0.1, residue as half / full /
dropped cycles, endurance cut-off ∈ {0, 0.5, 1, 2}, Goodman mean-stress correction with
σ_u ∈ {none, 100, 200, 400} — was scored by MAPE, selecting the simplest configuration within
0.002 of the best.

| | Score (1 − MAPE) |
|---|---|
| 8-fold CV × 3 seeds, config + C chosen inside each training fold — **shipped** | **0.9736 ± 0.0000** |
| In-sample, final config on all 64 files | 0.9743 |
| Same config, C fitted by log-space least squares instead of MAPE-optimal | 0.9733 |

**Shipped configuration.** `m = 5.0`, half-cycle residue (ASTM default), no cut-off, no
mean-stress correction, `C = 7.386 × 10⁸`. All 24 CV folds independently selected this exact
configuration. The MAPE curve is sharply peaked: m = 4.9 or 5.1 already costs 1.5–1.7 points,
m = 4.5 costs 10.6. log(S₅) vs log(D) has correlation 0.9994 and slope 0.9965 (≈ 1, i.e. the
label really is `S₅ / C`). Worst single-file relative error is 14.1 %.

**Improvement phase.** Residual diagnosis: the true/predicted ratio spans 0.95–1.16; it is
uncorrelated with damage level, cycle count or max range and only weakly with series skewness
(ρ = −0.54). Seven refinements under the same 8-fold × 3-seed CV:

| # | Variant | CV 1 − MAPE |
|---|---|---|
| 0 | m = 5, ASTM half-cycle residue (baseline) | 0.9736 |
| 1 | m chosen per fold on a 0.01 grid (4.90–5.10) | 0.9744 |
| 2 | Bilinear S-N (m₁ = 5 above knee, m₂ ∈ {3, 7, 9}, knee ∈ {1, 2, 4, 8}) | 0.9747 |
| 3 | Goodman / Gerber mean-stress correction, σ_u ∈ 60–800 | 0.9735 |
| 4 | Largest residue half-cycle counted as a full cycle | 0.9680 |
| 5 | Ridge residual correction on 7 rainflow features (weights 1/D) | 0.9696 |
| 6 | m on a 0.05 grid (4.5–5.5) per fold | 0.9723 |
| 7 | Power-law calibration D = a·S^b, MAPE-fitted | 0.9747 |

The earlier best gain was +0.0011 for two extra parameters, below the practical improvement
floor, so the one-parameter m = 5 model was kept. The residual cause was not established;
this is not evidence that further improvement is impossible. See the corrected rerun above.

**Earlier test predictions.** 16 values in 0.029–0.818. The current recalibrated
`predictions/shm_predictions.csv` ranges from 0.029057 to 0.821870 (training label range 0.029–0.928).

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
