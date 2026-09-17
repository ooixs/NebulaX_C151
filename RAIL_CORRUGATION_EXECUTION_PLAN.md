# Rail Corrugation Execution Plan

Status: planning document only. Creating this document does not authorize data experiments,
model training, test-set inference, packaging, or changes to the current shipped predictions.

Repository: `NebulaX_C151`

Authoritative task material:

- `../NebulaX-Hackathon-ProblemStatement/03_References/Rail_Corrugation/Rail_Corrugation_Info_Kit.md`
- `../NebulaX-Hackathon-ProblemStatement/01_Problem_Statement_3_Specifications.md`
- `../NebulaX-Hackathon-ProblemStatement/02_Datasets/Rail_Corrugation/`
- `../NebulaX-Hackathon-ProblemStatement/04_Example_Submission/rail_predictions.csv`

## 1. Objective

Build, validate, and package a reproducible model that classifies each one-second rail recording
as exactly one of:

- `Normal`
- `Side I`
- `Side II`

The official technical metric is macro F1 across these three classes. The completed Rail
Corrugation subsystem must also work through the common non-technical application and produce an
exactly conforming `rail_predictions.csv`.

The work should improve held-out reliability, not merely the score of one convenient split. In
particular, it must address the strong association between speed and the fault labels and the
possibility that nearby file numbers came from related acquisition runs.

## 2. Fixed facts and constraints

### 2.1 Data

- Training contains 272 CSV files and `Train_Labels.csv`.
- The labels supplied with the data contain 234 Normal, 14 Side I, and 24 Side II files.
- Test contains 68 CSV files with unpublished labels.
- Each documented recording is one second at 10,000 Hz.
- Each recording has 129 columns:
  - column 1: binary pulse from a 90-tooth wheel-speed sensor;
  - 64 vibration channels and 64 shock channels, alternating by sensor position;
  - 8 cars, each with positions 1 through 8;
  - positions 1, 3, 5, and 7 are Side I;
  - positions 2, 4, 6, and 8 are Side II.
- The wheel diameter is 0.85 m.
- The source dataset is approximately 5.5 GB and must remain immutable.

The Info Kit's scoring explanation contains stale approximate counts (`~190` Normal and `~9`
Side I). The actual label file and its 234/14/24 counts are authoritative.

### 2.2 Output contract

`rail_predictions.csv` must contain exactly:

```text
file_id,prediction
Test1.csv,Normal
...
```

Requirements:

- one row for every input CSV and no extra rows;
- the source basename, including `.csv`, in `file_id`;
- only `Normal`, `Side I`, or `Side II` in `prediction`;
- deterministic natural file ordering;
- no index column;
- CSV placed at the top level of `predictions.zip` for submission.

### 2.3 Evaluation rule

All model selection uses three-class macro F1. Accuracy may be reported as context but must never
drive selection. Every report must also include:

- per-class precision, recall, and F1;
- a three-by-three confusion matrix;
- the number of true and predicted examples per class;
- out-of-fold predictions at file level;
- results under both ordinary stratified and leakage-resistant validation.

## 3. Current repository baseline

The repository already contains a substantial implementation:

- `Rail Corrugation/code/pipeline.py`: loading, pulse-based speed calculation, spectral and time
  features, Normal references, side aggregation, and shared side-row construction;
- `Rail Corrugation/code/train.py`: repeated stratified cross-validation, three-class baseline,
  shared per-side detector, threshold selection, and final fitting;
- `Rail Corrugation/code/predict.py`: single-file or directory inference;
- `Rail Corrugation/notebooks/`: exploratory, challenger, and block-validation scripts;
- `predictions/rail_predictions.csv`: existing predictions for all 68 test files;
- `write_up.md`: the existing methodology and reported results.

The reported baseline to reproduce is:

| Evaluation | Reported result |
|---|---:|
| Repeated stratified 5-fold x 3 repeats, ExtraTrees shared side detector | macro F1 0.834 +/- 0.014 |
| Contiguous file-number block validation | macro F1 0.778 |
| Test prediction distribution | 57 Normal / 5 Side I / 6 Side II |

These are existing repository claims, not newly verified results. The first experimental branch
must reproduce them before they are treated as current evidence.

Known gaps to resolve:

1. The solution repository currently has no connected Rail dataset.
2. `Rail Corrugation/model/rail_model.joblib` is absent in the current checkout.
3. The app has no implemented Rail page.
4. Prediction smoke tests skip when the data or model is missing, so a skipped test is not an
   acceptance signal.
5. Raw speed and speed bin are model inputs even though speed is strongly associated with the
   labels.
6. The current threshold is optimized on the same out-of-fold predictions used to report the
   score rather than in a fully nested or cross-fitted procedure.
7. The documented Normal reference is per channel, while the implementation pools channels by
   side and speed bin.
8. Experimental scripts share caches and output locations; stale-cache and accidental-overwrite
   protection needs to be added before further comparisons.

## 4. Mandatory Git and experiment protocol

### 4.1 Hard branch gate

No Rail Corrugation experiment may run on `main`.

For this plan, an experiment includes any command that scans the full Rail dataset, extracts or
caches features, creates validation splits, trains or evaluates a model, tunes a threshold,
generates test predictions, or changes modelling conclusions.

Before the first such command:

1. Confirm the current worktree and preserve any pre-existing user changes.
2. Start from the intended base commit on `main`.
3. Create a new branch inside `NebulaX_C151` using:

   ```text
   rail-corrugation/exp-YYYYMMDD-<short-purpose>
   ```

4. Record the branch name and base commit in the experiment registry.
5. Confirm that the active branch is not `main` before accessing the data experimentally.

One branch may contain a coherent, predeclared experiment campaign. A materially different
campaign should receive another fresh branch. Emergency fixes to the application or packaging
should use a separate fix branch rather than being mixed into a modelling experiment.

Creating or editing planning documentation is not an experiment. No experiment branch is to be
created merely for this plan unless explicitly requested.

### 4.2 Commit discipline

- Keep data, feature caches, trained weights, and large generated files out of Git.
- Commit lightweight code, configurations, frozen split definitions, compact metric reports, and
  documentation.
- Make one logical commit for each completed phase or accepted change.
- Do not commit generated test predictions until the model is frozen and their provenance is
  recorded.
- Do not merge an experiment branch solely because its best single score is higher.
- Never rewrite, reset, or discard unrelated user changes.

### 4.3 Experiment records

Each experiment receives a stable ID such as `RC-E00` and must record:

- hypothesis;
- branch and commit;
- data-manifest hash;
- code/configuration hash;
- feature-cache fingerprint;
- random seeds;
- exact train/validation file lists;
- fitting and threshold-selection procedure;
- metrics and confusion matrices;
- runtime and peak-memory notes;
- conclusion: reject, retain for follow-up, or promote.

Large outputs should live under:

```text
weights/Rail Corrugation/experiments/<experiment-id>/
```

Compact, reviewable results should eventually be tracked under:

```text
Rail Corrugation/reports/
```

An experiment must write to its own directory. It must not overwrite
`Rail Corrugation/model/rail_model.joblib` or `predictions/rail_predictions.csv`.

## 5. Execution phases

Each phase has an entry condition, work items, outputs, and an exit gate. Stop at a failed exit
gate and resolve it before proceeding.

### Phase 0 - Create the experiment branch and reproducible harness

Entry condition: approval to begin experimental work.

Work:

1. Apply the branch protocol in Section 4.
2. Connect the immutable source data without duplicating 5.5 GB. Prefer a local ignored symlink or
   a configurable `--data-dir`; never write into the problem-statement dataset.
3. Add explicit CLI parameters for data directory, output directory, seed, split file, and number
   of workers where they are currently hard-coded.
4. Create an experiment configuration and registry.
5. Make caches content-addressed using at least the source manifest, feature configuration, and
   relevant code version. Reject a cache whose fingerprint does not match.
6. Use atomic writes for caches, reports, and model artifacts so interruption cannot leave a
   plausible but incomplete file.
7. Bound parallel workers and log elapsed time and memory. The pipeline must remain usable on a
   CPU-only judging machine.

Outputs:

- active experiment branch;
- experiment registry;
- isolated output directory convention;
- reproducible command template;
- cache-validation logic.

Exit gate:

- active branch is not `main`;
- a dry run can resolve paths and configuration without training;
- no source dataset file is modified;
- no shipped artifact or prediction is overwritten.

### Phase 1 - Full data and provenance audit (`RC-E00`)

Hypothesis: the documented schema is consistent enough for one deterministic loader, and any
grouping or duplication structure can be identified before validation is frozen.

Work:

1. Inventory the 272 training and 68 test CSVs and reconcile them with `Train_Labels.csv`.
2. Record file size and a cryptographic content hash for every source file.
3. Validate every file, not a sample:
   - row and column counts;
   - header identity and ordering;
   - numeric parse success;
   - NaN and infinity counts;
   - constant or near-constant channels;
   - extreme ranges and clipping/quantization;
   - pulse values and transition counts.
4. Parse car, position, and signal type from headers and compare the parsed mapping with the
   positional mapping. Fail loudly on disagreement.
5. Confirm the speed convention:
   - inspect pulse high/low run lengths;
   - verify that a tooth normally creates two transitions;
   - calculate whole-file and sliding-window speed;
   - quantify within-file speed variation and pulse-dropout cases.
6. Detect exact duplicate files and exact duplicate sensor matrices.
7. Search for near duplicates or related acquisition runs using compact fingerprints:
   - speed and pulse-transition patterns;
   - per-channel RMS summaries;
   - low-resolution spectra;
   - adjacent file-number similarity.
8. Compare train and test covariate distributions without assigning or inferring test labels:
   speed, missingness, amplitude, spectral energy, and acquisition fingerprints.

Outputs:

- `data_manifest.csv`;
- `data_audit.json` and a readable audit report;
- channel mapping table;
- candidate acquisition-group assignments with documented confidence;
- a list of exclusions, if any, with reasons. No file may be silently excluded.

Exit gate:

- every label maps to exactly one training file;
- every test file appears exactly once;
- all schema anomalies are either fixed in the loader or explicitly rejected;
- duplicate and grouping findings are available before split definitions are created.

### Phase 2 - Freeze leakage-resistant validation (`RC-E01`)

Hypothesis: repeated stratified CV alone is optimistic, but a combination of fixed protocols can
separate real side-specific signal from speed and acquisition shortcuts.

Freeze the following protocols before testing new features:

#### Protocol A: repeated stratified file CV

- 5 folds x 3 repeats using fixed recorded seeds.
- Used for comparability with existing results.
- All reference statistics and transformations are fitted inside the training fold.

#### Protocol B: acquisition-group or blocked CV

- Prefer groups discovered in Phase 1.
- If there is no defensible group identifier, use contiguous numeric file blocks as an explicit
  stress test.
- Keep duplicates and near-duplicate families in the same fold.
- Because some blocks may contain no Side I files, calculate one pooled out-of-fold confusion
  matrix across all blocks rather than averaging undefined or highly unstable fold-level class
  scores.

#### Protocol C: speed-matched challenge evaluation

- Evaluate fault files only against Normal files from overlapping speed support.
- Report performance in the high-speed region separately.
- Do not claim this subset is the official test distribution; it is a confounding diagnostic.

#### Protocol D: speed-bin transfer stress test

- Where class coverage permits, hold out speed bins or narrow speed ranges.
- If a fold cannot contain all three classes, report the applicable binary detection and side
  localisation components rather than manufacturing a three-class macro F1.

Threshold selection must be nested or cross-fitted:

- derive the Normal/fault threshold using only inner-training predictions;
- apply that frozen threshold to the outer fold;
- never tune a threshold on the same outer predictions used for the reported score;
- for the final full-data model, derive the threshold from cross-fitted training predictions and
  aggregate fold thresholds using a predeclared rule such as the median.

Outputs:

- versioned split-definition files containing exact filenames;
- validation runner shared by all subsequent experiments;
- tests proving that fit-only objects never see validation files;
- speed-only diagnostic baseline;
- majority-class baseline.

Exit gate:

- every file has exactly one outer-fold prediction per validation repeat;
- no duplicate family crosses a grouped fold boundary;
- nested/cross-fitted threshold tests pass;
- the validation protocol is frozen before challenger work.

### Phase 3 - Reproduce the current baseline (`RC-E02`)

Hypothesis: the existing code and documented settings can reproduce the recorded metrics and
predictions from a clean run.

Work:

1. Run the current feature pipeline unchanged through the new isolated harness.
2. Reproduce the three-class RandomForest and shared per-side ExtraTrees models.
3. Run all frozen validation protocols.
4. Compare with the recorded results:
   - stratified macro F1 approximately 0.834;
   - block macro F1 approximately 0.778;
   - per-class confusion totals;
   - threshold approximately 0.28;
   - existing 68-file prediction vector.
5. Explain every discrepancy before changing features or models. Check version differences,
   ordering, seeds, cache provenance, and threshold fitting first.

Outputs:

- baseline configuration;
- out-of-fold predictions for every protocol;
- metrics, confusion matrices, and runtime;
- exact diff against the committed test prediction file;
- reproducibility verdict.

Exit gate:

- deterministic reruns match each other exactly;
- material differences from the recorded baseline are explained;
- the reproduced baseline becomes the fixed comparator for all challengers.

### Phase 4 - Quantify and control speed confounding (`RC-E03`)

Hypothesis: speed is useful for transforming frequency to wavelength but raw speed as a predictor
may inflate cross-validation performance without representing corrugation.

Run paired ablations on identical frozen folds:

1. Speed-only model.
2. Current model with raw speed and speed bin as predictive features.
3. Remove raw speed but retain speed for wavelength conversion and reference matching.
4. Remove both raw speed and speed-bin indicators from the classifier.
5. Replace hard speed bins with a training-fold-only smooth or nearest-neighbour Normal reference.
6. Apply speed-matched sample weighting to Normal files, fitted within the training fold.

For each variant, report Protocols A-D and feature importance or permutation importance. Inspect
false positives and false negatives by speed.

Decision rule:

- raw speed may remain a predictive input only if it improves grouped and speed-matched results,
  not merely ordinary stratified CV;
- speed remains available for physically required frequency-to-wavelength conversion regardless
  of this decision;
- no label-informed transformation may use validation or test distributions.

Exit gate:

- the contribution of speed alone is known;
- the selected treatment of speed is documented and defensible;
- the primary model retains measurable side-specific signal after speed matching.

### Phase 5 - Improve the signal representation (`RC-E04` onward)

Change one feature family at a time and compare it against the reproduced baseline on identical
folds. Do not launch an unrestricted feature search.

#### 5.1 Correct Normal-reference granularity

Compare:

- current pooled-by-side reference;
- per-position reference across cars;
- per-car-position/channel reference;
- hierarchical shrinkage from channel to position to side when a speed region has few Normal
  examples.

All references must be fitted from Normal files in the training fold only. Extend reference
normalization to wavelength-domain features if the audit supports it.

#### 5.2 Spectral measurement corrections

Evaluate:

- detrending and windowing choices;
- integrated linear PSD power followed by logarithm versus mean log-bin PSD;
- frequency resolution and Welch segment length;
- wavelength bands supported by the documented centimetre-to-decimetre defect scale;
- peak prominence relative to a local spectral background rather than a global median.

#### 5.3 Order/distance-domain features

If Phase 1 shows meaningful within-file speed variation:

1. derive local speed only from the pulse signal;
2. integrate distance over time;
3. resample vibration into distance coordinates;
4. compute spatial-frequency or wavelength spectra.

If speed is effectively constant within files, retain the simpler `lambda = v/f` mapping and
record why distance resampling was rejected.

#### 5.4 Cross-sensor evidence

Evaluate physically interpretable agreement features:

- same-side dominant-wavelength concentration;
- coherence or correlated narrow-band energy across cars on the same rail;
- paired Side I/Side II position contrasts within each car;
- number and spatial distribution of cars supporting the same side diagnosis;
- robust top-k rather than unconstrained maximum aggregation;
- vibration/shock agreement and disagreement indicators.

Do not swap Side I and Side II as augmentation unless Phase 1 demonstrates exchangeability. The
existing experiments report that mirroring reduced performance.

Exit gate for every feature experiment:

- cache fingerprint changes when feature semantics change;
- paired out-of-fold results are complete under the fixed protocols;
- gains are not isolated to one seed, one fold, or only low-speed Normal files;
- rejected variants and reasons remain in the registry.

### Phase 6 - Model and decision-rule challengers

Start only after the signal representation is stable.

Primary candidates:

1. Existing shared per-side ExtraTrees detector.
2. Regularized logistic model on a reduced, standardized feature set as a low-variance check.
3. Two-stage model:
   - Stage A: Normal versus corrugation;
   - Stage B: Side I versus Side II conditional on fault evidence.
4. Direct three-class model retained as a reference, not assumed to be competitive.

Rules:

- use class weighting or fold-local resampling; never duplicate validation examples;
- limit hyperparameter searches and declare ranges before running them;
- calibrate/tune decisions only inside training data;
- do not add deep sequence models unless residual analysis shows the engineered features discard
  repeatable signal and a leakage-resistant evaluation can support the added capacity;
- use ensembles only when constituent errors are demonstrably complementary on outer-fold
  predictions.

### Phase 7 - Selection and stopping rule

Compare every candidate with the reproduced baseline on paired, identical splits.

A challenger is eligible for promotion only when:

1. its pooled grouped/blocked macro F1 improves by at least 0.01, or the paired improvement is
   larger than its resampling uncertainty;
2. repeated-stratified macro F1 does not materially regress;
3. speed-matched macro F1 does not regress;
4. Side I recall does not drop by more than one out-of-fold example unless precision improves
   enough to increase Side I F1;
5. no leakage, cache-provenance, or determinism check fails;
6. inference remains practical for the app on CPU.

If candidates are statistically tied, prefer in this order:

1. lower leakage/confounding risk;
2. simpler features and model;
3. faster inference and smaller artifact;
4. better probability stability and explainability.

Stop modelling after two consecutive predeclared experiment rounds fail to produce an eligible
challenger. This avoids repeatedly tuning to 38 fault files.

### Phase 8 - Freeze and train the final Rail model

Work:

1. Freeze feature configuration, model parameters, label order, split definitions, and threshold
   derivation.
2. Fit references and the selected model on all 272 training files.
3. Derive the final threshold using the predeclared cross-fitted rule, not test predictions.
4. Serialize one application artifact containing:
   - model and fitted preprocessing/reference objects;
   - exact feature-column order;
   - class vocabulary and side map;
   - decision threshold;
   - data-manifest and code/configuration fingerprints;
   - training timestamp and library versions;
   - validation summary.
5. Load the artifact in a fresh process and verify deterministic predictions.
6. Promote it to `Rail Corrugation/model/rail_model.joblib` only after all gates pass.

Artifact acceptance tests:

- single-file inference;
- directory inference over all 68 test files;
- natural ordering (`Test2.csv` before `Test10.csv`);
- exact output columns and label vocabulary;
- clear failure for malformed headers, wrong column count, empty input, or nonnumeric data;
- no training-data dependency at inference time;
- no write outside the requested output/cache area;
- acceptable CPU runtime and peak memory.

### Phase 9 - Generate and verify held-out predictions

Run test inference only after the model is frozen.

Work:

1. Generate a candidate CSV into the final experiment directory.
2. Validate row count, unique IDs, exact test-file coverage, labels, ordering, quoting, and absence
   of an index column.
3. Re-run in a fresh process and require byte-identical output.
4. Record model hash, data-manifest hash, command, and output hash.
5. Review prediction counts, speed distribution, and confidence margins only as a sanity check.
   Never change thresholds merely to match the training class prior.
6. Replace `predictions/rail_predictions.csv` only after review and explicit promotion.

The test inputs may be used for unsupervised schema and covariate checks, but never for manual
pseudo-labelling or choosing a model based on visually preferred predictions.

### Phase 10 - Rail application page

Implement the Rail page in the shared app after the inference contract is frozen.

Minimum behavior:

- upload or drag-and-drop one CSV or a batch of CSVs;
- validate schema and display actionable errors;
- show `Normal`, `Side I`, or `Side II` prominently;
- display measured speed and both side evidence scores;
- describe scores as model evidence, not calibrated probabilities unless calibration has been
  validated;
- provide a compact, non-technical explanation of Side I versus Side II sensor positions;
- optionally plot side-by-side wavelength spectra or the strongest supporting bands;
- allow exact `rail_predictions.csv` download;
- operate without network access and without loading training data;
- cache the model, not user uploads or predictions across unrelated sessions.

Usability checks:

- a non-technical user can complete upload -> prediction -> download without instructions;
- single-file and batch behavior agree with `predict.py`;
- filenames with spaces and mixed natural ordering work;
- corrupted files do not crash the entire app;
- the interface does not expose internal paths or stack traces.

### Phase 11 - Documentation, packaging, and handoff

Update the Rail section of the write-up with:

- exact data and side mapping;
- speed decoding and confounding findings;
- validation protocols and why grouped evaluation matters;
- baseline and challenger tables;
- per-class results and confusion matrices;
- limitations caused by only 14 Side I and 24 Side II examples;
- honest expected-performance range;
- final feature/model rationale;
- inference runtime and app behavior.

Packaging checks:

- trained Rail artifact is included where the app expects it;
- `predictions.zip` contains `rail_predictions.csv` at its top level;
- raw datasets, feature caches, and CV weights are excluded;
- the packaged app runs in a clean environment;
- the packaged prediction matches the frozen output hash;
- the demo video shows Rail upload, result interpretation, and CSV download;
- the final report distinguishes validation evidence from unknown held-out performance.

## 6. Predeclared experiment sequence

| ID | Question | Single controlled change | Primary evidence |
|---|---|---|---|
| RC-E00 | Is the data/schema/provenance understood? | Audit only | Complete manifest and grouping report |
| RC-E01 | How optimistic is ordinary CV? | Freeze stratified, grouped, and speed-matched protocols | File-level OOF reports |
| RC-E02 | Is the existing result reproducible? | Current code/settings in isolated harness | Match to recorded baseline |
| RC-E03 | Is speed acting as a shortcut? | Speed-only and raw-speed ablations | Grouped and speed-matched deltas |
| RC-E04 | Does reference granularity matter? | Pooled side vs position/channel hierarchy | Paired OOF delta |
| RC-E05 | Are spectral measurements well formed? | PSD integration/background variants | Paired OOF delta |
| RC-E06 | Is distance/order processing justified? | Current wavelength map vs distance resampling | Transfer and grouped delta |
| RC-E07 | Does cross-sensor coherence add evidence? | One coherence/agreement family | Rare-class F1 and grouped delta |
| RC-E08 | Is a two-stage decision safer? | Shared side detector vs two-stage model | Macro F1 and error decomposition |
| RC-E09 | Does a small ensemble add stable value? | Only complementary accepted models | Paired gain above uncertainty |

Do not automatically run all experiments. After each result, update the registry and decide
whether the next experiment remains justified. RC-E09 is optional and should be skipped unless
outer-fold residuals show complementary errors.

## 7. Final acceptance checklist

### Scientific and computational integrity

- [ ] Raw data remained immutable.
- [ ] Every experiment ran on a new non-`main` experiment branch.
- [ ] Exact files and labels were reconciled.
- [ ] Duplicate/acquisition groups were handled before splitting.
- [ ] All preprocessing and references were fitted inside training folds.
- [ ] Threshold selection was nested or cross-fitted.
- [ ] Speed-only, speed-matched, and grouped diagnostics were reported.
- [ ] Out-of-fold predictions support every reported validation metric.
- [ ] Results are reproducible from recorded commands and fingerprints.

### Model and inference

- [ ] Final model passed the promotion rule.
- [ ] Fresh-process inference is deterministic.
- [ ] Model artifact contains preprocessing, schema, threshold, and provenance.
- [ ] Single-file and batch predictions agree.
- [ ] Malformed input produces a clear error.
- [ ] CPU runtime and memory are acceptable.

### Submission

- [ ] `rail_predictions.csv` contains exactly 68 unique test rows.
- [ ] Output columns and labels match the official contract exactly.
- [ ] The CSV is at the top level of `predictions.zip`.
- [ ] The app generates the same predictions as the CLI.
- [ ] The Rail page supports upload, explanation, and download.
- [ ] The write-up separates verified evidence, assumptions, and limitations.
- [ ] Raw data and development caches are excluded from the package.

## 8. Explicit pause points

Pause for review before:

1. creating the first experiment branch and running RC-E00;
2. freezing the validation/grouping protocol after the data audit;
3. promoting any challenger over the reproduced baseline;
4. generating or replacing final test predictions;
5. copying the final artifact into the submission package.

Until the first pause point is approved, this file is the only intended change.
