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
├── docs/references/             # Info Kits + supporting docs, per subsystem
└── tests/
```

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

Research-backed design for each subsystem. Every initial proposal was checked against the
authoritative Info Kits and the relevant literature; where the literature pointed to a better
approach the proposal was revised, otherwise it was kept. **Nothing below has been validated on the
actual data yet** — each "recommended" method is a hypothesis to be tested against its baseline on
leakage-safe splits, and the simpler method wins unless the refinement shows a convincing gain on
the official metric.

### Summary of verdicts

| Subsystem | Original proposal | Verdict | Key changes |
|---|---|---|---|
| Door | Timestamp-gap segmentation → per-segment current/position features → LightGBM/RF, 1D-CNN as comparison | **Keep, refine** | Phase-aware "excess current" features; MiniROCKET replaces the CNN as comparison model; change-point detection as segmentation fallback |
| ACV | Peer-relative robust z-scores per car → weighted sum → rank | **Keep, refine** | Peer comparison conditioned on operating mode; small healthy-response residual model; persistence-based scoring |
| Rail Corrugation | Speed → wavelength conversion, per-side aggregates, side-mirroring augmentation, per-side binary detector | **Revise** | Dual-domain (frequency + wavelength) features, normal-response reference from Normal files, cross-rail coupling features; mirroring and per-side detector demoted to challengers |
| SHM | Rainflow + Miner's rule with fitted (m, C); log-space regression fallback | **Keep, refine** | Calibrate directly on MAPE (not log least-squares); treat rainflow residue convention and channel aggregation as calibration choices; drop the "near-1.0" expectation |

### Shared principles

- **Metrics.** `common/` re-implements the four judge formulas exactly (IoU-weighted F1, linear
  rank-decay, macro F1, max(0, 1 − MAPE)) and is unit-tested against the worked examples in the
  Info Kits. All model selection uses these, end-to-end, never a proxy metric.
- **Splits.** Never random rows. Door: contiguous time blocks of `Train.csv`. ACV: leave-one-case-out.
  Rail: repeated stratified K-fold grouped by file (and by recording run if metadata allows). SHM:
  file-level K-fold, grouped by line/load condition if it can be inferred. Every fitted component
  (reference profiles, thresholds, weights, S-N constants) is refit inside each training fold.
- **Simplicity rule.** Sample sizes are small everywhere (6 ACV cases, 14 Side I files, 64 SHM
  files). Low-capacity, physically-motivated models are the default; capacity is added only when
  validation residuals justify it.

### 1. Door — segment, then classify (kept, refined)

**Task.** One continuous stream → find every open/close cycle → label `Normal` / `Abnormal
resistance`. Scored on IoU-weighted F1: timing *and* label both count, over-segmentation and loose
boundaries are penalised directly.

**Research.**
- Wei et al. (2020), *Urban Rail Transit* — resistance "sub-health" diagnosis from door motor
  signals. Splits each movement into rising-speed, uniform-speed, slow-speed and termination
  phases and compares motor current against normal operating envelopes (μ ± 3σ / 6σ); current
  reflects resistance directly because DC motor torque ∝ current.
  <https://link.springer.com/article/10.1007/s40864-020-00133-4>
- Shimizu, Perinpanayagam & Namoano (2022), *IEEE Aerospace Conf.* — real-world railway door data
  with start–stop discontinuities; the working pipeline is segmentation → feature extraction →
  dimensionality reduction → normal/abnormal separation. <https://doi.org/10.1109/aero53065.2022.9843627>
- Dempster, Schmidt & Webb (2021), *KDD* — MiniROCKET: near-deterministic random convolutional
  kernel transform + ridge classifier; state-of-the-art accuracy at very low cost, recommended for
  small datasets. <https://arxiv.org/abs/2012.08791>
- Truong, Oudre & Vayatis (2020), *Signal Processing* — offline change-point detection review;
  implemented in the `ruptures` library. <https://github.com/deepcharles/ruptures/>

**Implications.** The two-stage architecture matches how practitioners handle this data. The
literature's strongest signal is *phase-specific* excess current, not whole-cycle peak/RMS. A
trained CNN is unjustified at this data size when MiniROCKET gives a strong sequence-model
comparison for free.

**Recommended method.**
1. *Segmentation candidates, evaluated label-agnostically against `Train_Segments_Answer.csv`
   (IoU-F1 with labels ignored; target ≈ 1.0):*
   - A — timestamp gaps between rows (hypothesis: the controller only logs while the door is
     active). Verify; do not assume.
   - B — small state machine over motion evidence: position derivative, current activity,
     open/close commands, DCSR/DCSL/DLSR/DLSL transitions.
   - C — `ruptures` PELT change-point detection on [current, position] as a fallback.
   Do not over-trim boundaries: startup and locking activity may lie inside the labelled segment.
2. *Features per segment.* Build a normal-only reference current profile per operation (Open /
   Close) — and per door if `Car Number` / `Door Number` vary and are usable — by resampling
   current against normalised position (or time) into ~20 bins. Then: excess over reference
   (max, integral, location, persistence), stall / slow-travel indicators, electrical energy,
   back-EMF vs position, duration columns, plus conventional stats. Keep absolute current and raw
   duration; do not amplitude-normalise cycles (that would erase the fault signature).
3. *Classifiers to compare.* Random Forest / shallow gradient boosting on features vs MiniROCKET +
   `RidgeClassifierCV` on aligned current/position profiles. Class weights; decision threshold
   tuned on the end-to-end metric.

**Validation.** Hold out contiguous time blocks (and door groups if identifiers exist). Select the
pipeline on end-to-end IoU-F1 on the held-out stream, not on classification accuracy of pre-cut
cycles.

**Open assumptions.** Gap-based logging; usability of door identifiers; label balance in Train.

### 2. ACV — peer-relative ranking, conditioned on operating state (kept, refined)

**Task.** 6 labelled train cases + 1 test case, 8 cars each, exactly one faulty car per case.
Output: all cars ranked most → least likely faulty. Scored on linear rank-decay (rank 2 still
earns 0.875).

**Research.**
- Du, Domanski & Payne (2015), *Applied Thermal Engineering* (NIST) — fault effects, including
  refrigerant undercharge, differ by system configuration and operating condition; FDD works by
  comparing measured features with expected *fault-free* values under the same conditions.
  <https://www.nist.gov/publications/effect-common-faults-performance-different-vapor-compression-systems>
- Mattera et al. (2018), *Sensors* — virtual sensors: predict one measurement from related ones
  with a simple model, detect faults via the residual. <https://www.mdpi.com/1424-8220/18/11/3931>
- Takeuchi & Saito (2019), *arXiv* — scaling-law refrigerant-leak soft sensor, configuration-
  independent, but requires *refrigerant pipe* temperature, which our schema does not have. Only
  its operating-mode conditioning transfers. <https://arxiv.org/html/1902.09427>

**Implications.** Peer comparison is the right frame (the other 7 cars are contemporaneous
fault-free references), but it must compare like with like: cars in different running modes,
setpoints or startup phases are not comparable. Rank by *unexplained cooling underperformance*,
not by "hottest car".

**Recommended method.**
1. *Loading.* Parse `Car <NN> - <parameter>` headers dynamically, map parameters by keyword
   (temperature / mode / status), use the intersection available in the file, emit car IDs exactly
   as they appear.
2. *Condition-matched peer features.* At each timestamp compare a car only with peers in the same
   running/setting mode; robust z-score against the peer median. Features: setpoint error (indoor
   temp − cooling control temp, mean and p95 while cooling), cooling-mode duty fraction, deviation
   from peer-median indoor temp, cooldown rate after mode entry.
3. *Healthy-response residual model.* Ridge regression fitted on the healthy cars of the training
   cases predicting indoor temperature change from indoor temp, outdoor temp, target temp, running
   mode, load-halved status and time since mode entry. Residual = observed cooling weaker than
   expected; aggregate per car.
4. *Scoring.* Small weighted sum of 3–5 interpretable components (persistent excess temperature,
   weak cooldown response, sustained cooling demand without response). Weights chosen by
   leave-one-case-out with minimal search; a brief spike counts less than persistent
   underperformance. Information-valid flags mask bad data; they are not fault evidence.

**Validation.** Leave-one-case-out over the 6 cases; report the true car's rank in every case and
the mean rank-decay score. Fit reference models and weights without the held-out case.

**Open assumptions.** Single fault per file; all cars share ambient conditions; the 60+-parameter
file is handled via the parameter intersection.

### 3. Rail Corrugation — dual-domain, reference-corrected, side-relative features (revised)

**Task.** 272 one-second files at 10 kHz, 129 columns (speed pulse + 64 vibration/shock pairs);
234 Normal / 14 Side I / 24 Side II. Scored on macro F1, so the two rare classes carry two-thirds
of the score.

**Research.**
- Baasch et al. (2025), *TRA / Lecture Notes in Mobility* — STFT → log-spectral averaging to
  remove the wheel–rail system response → distance–wavenumber domain via speed (x = v·t,
  k = f/v); features extracted from wavenumber spectra.
  <https://link.springer.com/chapter/10.1007/978-3-031-85578-8_42>
- De Rosa et al. (2024), *Applied Sciences* — excitations split into frequency-constant (vehicle /
  track modes), wavelength-constant (rail / wheel defects) and broadband impulsive (welds,
  switches). Same-side axle boxes share peaks under corrugation; left and right differ, especially
  in curves. <https://www.mdpi.com/2076-3417/14/19/8920>
- Hassanieh et al. (2023), *Int. J. Rail Transportation* — Random Forest on ABA features to predict
  corrugation level; engineers features that *nullify vibration inherited from the other rail*
  through left–right dynamic coupling. <https://doi.org/10.1080/23248378.2023.2220112>
- Li et al. (2025), *Measurement* (TU Delft) — short-pitch corrugation detection under varying
  speed from time–frequency ABA features; likelihood from the number of signals detecting it and
  severity from impact energy. <https://doi.org/10.1016/j.measurement.2025.118064>

**Implications for the original proposal.** (i) Converting frequency to wavelength does not remove
speed-dependent amplitude changes or vehicle/track resonances — a reference for *normal* response
is needed. (ii) Cross-rail coupling means Side II sensors pick up Side I corrugation, so
side-relative features are essential for localisation. (iii) Side-mirroring augmentation is only
valid if sensor/operating symmetry holds; it does not create independent fault examples.

**Recommended method.**
1. *Speed decode.* Verify the pulse convention (edges per tooth) on real files before computing
   v = transitions / (edges_per_tooth × 90) × π × 0.85 m / Δt; a factor-of-two error shifts every
   inferred wavelength. Use sliding-window local speed; if speed varies within the second,
   resample to the distance domain.
2. *Per-channel features in both domains.* Welch PSD band powers in frequency *and* in wavelength
   (third-octave-like bands across the corrugation range), dominant wavelength and its prominence,
   spectral entropy/flatness, RMS, kurtosis, crest factor; shock channel: peak count and
   impulsiveness.
3. *Normal-response reference.* From Normal training files only, per channel/position and speed
   bin, take the median log-bandpower; add excess-over-reference features. Keep the raw spectral
   features alongside; never normalise a test recording against itself.
4. *Side aggregation and cross-rail features.* Side I = positions 1,3,5,7; Side II = 2,4,6,8, across
   all 8 cars → mean, max, top-quartile, std, and count of channels exceeding the reference. Add
   Side I − Side II excess per band, ratios, and same-side agreement (fraction of same-side
   channels sharing the dominant wavelength). Preserve local evidence — a one-second window does
   not guarantee every car crosses the defect.
5. *Models.* Baseline: class-weighted 3-class Random Forest / gradient boosting. Challenger: shared
   per-side detector trained on (file, side) rows with side-relative features, decoded to exactly
   one of `Normal` / `Side I` / `Side II` with thresholds tuned on out-of-fold macro F1. Mirroring
   only after symmetry is verified; augmented copies stay in the same fold as their source.

**Validation.** Repeated stratified K-fold grouped by file (and run if known); report macro F1,
per-class precision/recall and the confusion matrix.

### 4. SHM — calibrated rainflow + Miner's rule (kept, refined)

**Task.** 64 train / 16 test dynamic-stress files; predict cumulative fatigue damage. Scored on
max(0, 1 − MAPE). The Info Kit states the labels were produced by rainflow counting + Miner's rule
with an S-N curve σ^m · N = C, so this is primarily a physics-recovery problem.

**Research.**
- `rainflow` (iamlikeme) — ASTM E1049-85 implementation; `extract_cycles` yields (range, mean,
  count = 1.0 or 0.5) for every closed and residual cycle. <https://github.com/iamlikeme/rainflow/>
- Marsh et al. (2016), *Int. J. Fatigue* — rainflow residue processing materially changes damage;
  conventional half-cycle treatment can be non-conservative, and for short records the largest
  ranges often sit in the residue. <https://doi.org/10.1016/j.ijfatigue.2015.10.007>
- de Myttenaere et al. (2016), *Neurocomputing* — the MAPE-optimal model is a weighted MAE
  regression with weights 1/|y|; least squares in log space is *not* equivalent.
  <https://arxiv.org/abs/1605.02541>
- Dirlik & Benasciutti (2021), *Metals*; Slavič et al. (2023), *MSSP* (FLife package) — spectral
  fatigue methods assume stationary Gaussian loading; useful for speed, not accuracy, here.
  <https://www.mdpi.com/2075-4701/11/9/1333>, <https://doi.org/10.1016/j.ymssp.2023.110149>

**Implications.** Keep rainflow + Miner's rule as the primary model. Two things change: the
calibration objective must be MAPE itself, and the hidden conventions of the reference computation
(residue treatment, range vs amplitude, channel aggregation, mean-stress correction) must be
treated as discrete calibration choices rather than assumed.

**Recommended method.**
1. *Inspect first.* Number of channels per file ("all monitoring points"), units, sampling rate,
   length; whether the label is for one point, a sum or a maximum across points.
2. *Damage model.* Unbinned rainflow per channel keeping half cycles; for cycle range rᵢ,
   amplitude aᵢ = rᵢ / 2, count nᵢ: D̂ = (1/C) · Σ nᵢ aᵢ^m. Preserve stress magnitudes (no per-file
   standardisation); treat file numbers as identifiers, never as chronology; never concatenate
   files into an invented operating history.
3. *Calibration on MAPE.* Choose m and C — plus discrete options: residue as half / full / repeated
   cycles, range vs amplitude, per-channel sum / max / specific channel, optional Goodman
   mean-stress correction, optional endurance cut-off — by minimising mean |D − D̂| / D on the
   training folds (weighted-MAE objective). Log-space fitting is an initialiser only. Keep the
   parameter count tiny; do not add material constants just because they lower training error on
   64 files.
4. *Residual model, only if warranted.* If systematic residuals remain after step 3, first
   re-check units and conventions; then a small ridge/GBM on rainflow-derived features predicting a
   MAPE-weighted log-ratio correction.
5. *Fallback.* Gradient boosting on rainflow-derived features (Σ nᵢ aᵢ^m for several m, amplitude
   histogram moments, RMS, peak counts, duration) with a MAPE-weighted objective, if physics
   recovery fails outright.

**Validation.** File-level K-fold (grouped by line/load if inferable); report the 1 − MAPE
distribution across folds. Confirm no zero-valued labels (MAPE undefined) — if any exist, establish
the judges' convention before choosing a workaround.

**Correction.** The earlier expectation of a near-1.0 score was overconfident: the reference
conventions and material constants are not disclosed, so that performance cannot be promised.

### References

- Wei S., Xu Z., Chen J., Shi X. (2020). Research on Subhealth Diagnosis Method for Resistance of Urban Rail Transit Door System. *Urban Rail Transit* 6, 218–230. <https://link.springer.com/article/10.1007/s40864-020-00133-4>
- Shimizu M., Perinpanayagam S., Namoano B. (2022). Real-Time Techniques for Fault Detection on Railway Door Systems. *IEEE Aerospace Conference*. <https://doi.org/10.1109/aero53065.2022.9843627>
- Dempster A., Schmidt D.F., Webb G.I. (2021). MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification. *KDD 2021*. <https://arxiv.org/abs/2012.08791>
- Truong C., Oudre L., Vayatis N. (2020). Selective review of offline change point detection methods. *Signal Processing* 167. <https://github.com/deepcharles/ruptures/>
- Du Z., Domanski P.A., Payne W.V. (2015). Effect of Common Faults on the Performance of Different Vapor Compression Systems. *Applied Thermal Engineering* 98. <https://www.nist.gov/publications/effect-common-faults-performance-different-vapor-compression-systems>
- Mattera C.G., Quevedo J., Escobet T., Shaker H.R., Jradi M. (2018). A Method for Fault Detection and Diagnostics in Ventilation Units Using Virtual Sensors. *Sensors* 18(11). <https://www.mdpi.com/1424-8220/18/11/3931>
- Takeuchi S., Saito T. (2019). Fault Diagnosis Method Based on Scaling Law for On-line Refrigerant Leak Detection. *arXiv:1902.09427*. <https://arxiv.org/html/1902.09427>
- Baasch B., Heusel J., Lähns A., Roth M., Groos J. (2025). Spectral Characterization of the Rail Surface in Urban Environments Using in-Service Vehicles. *Transport Transitions (TRA 2024)*, LNMOB. <https://link.springer.com/chapter/10.1007/978-3-031-85578-8_42>
- De Rosa A., Luber B., Müller G., Fuchs J. (2024). Methodology to Detect Rail Corrugation from Vehicle On-Board Measurements by Isolating Effects from Other Sources of Excitation. *Applied Sciences* 14(19). <https://www.mdpi.com/2076-3417/14/19/8920>
- Hassanieh W., Chehade A., Facchinetti A., Carman M., Bocciolone M., Somaschini C. (2023). Leveraging machine learning to predict rail corrugation level from axle-box acceleration measurements on commercial vehicles. *Int. J. Rail Transportation* 12(4). <https://doi.org/10.1080/23248378.2023.2220112>
- Li S., Zhang P., Núñez A., Dollevoet R., Li Z. (2025). Monitoring of rail short pitch corrugation using the time-frequency features of both vertical and longitudinal axle box accelerations. *Measurement* 255. <https://doi.org/10.1016/j.measurement.2025.118064>
- iamlikeme/rainflow — ASTM E1049-85 rainflow cycle counting in Python. <https://github.com/iamlikeme/rainflow/>
- Marsh G. et al. (2016). Review and application of Rainflow residue processing techniques for accurate fatigue damage estimation. *Int. J. Fatigue* 82. <https://doi.org/10.1016/j.ijfatigue.2015.10.007>
- de Myttenaere A., Golden B., Le Grand B., Rossi F. (2016). Mean Absolute Percentage Error for regression models. *Neurocomputing* 192. <https://arxiv.org/abs/1605.02541>
- Dirlik T., Benasciutti D. (2021). Dirlik and Tovo-Benasciutti Spectral Methods in Vibration Fatigue: A Review with a Historical Perspective. *Metals* 11(9). <https://www.mdpi.com/2075-4701/11/9/1333>
- Zorman A., Slavič J., Boltežar M. (2023). Vibration fatigue by spectral methods — A review with open-source support. *MSSP* 190. <https://doi.org/10.1016/j.ymssp.2023.110149>
