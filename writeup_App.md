# Technician app and maintenance workflow

## Purpose

NebulaX gives maintenance staff one workflow for all four condition-monitoring models. It is not
only a prediction viewer: it connects a recording to an asset, explains the measurements behind
the result, preserves the analysis, and records the inspection follow-up.

The main interface uses operational language and keeps challenge submission and model setup as
secondary tools. A result supports inspection planning; it never declares a train fit for service.

## Technician workflow

1. Select Doors, Air conditioning, Rail condition, or Structural health.
2. Enter the train or asset ID, location/component, recording time, and work order.
3. Add the original sensor files and run the check.
4. Select an individual movement, car, or recording to review its evidence.
5. Inspect the asset and save the status and technician note.
6. Reopen the durable record from the Review queue or download an inspection report.

This common flow avoids four separate applications while preserving each subsystem's distinct
input and output.

## Results designed for physical follow-up

| Subsystem | Primary result | Evidence shown for a selected item | Physical follow-up |
|---|---|---|---|
| Door | Normal or Abnormal resistance for each movement | Direction, duration, model score/threshold, travel current, and resistance signal | Match the time to a movement; inspect track, seals, and motor |
| ACV | Cars in inspection order | Score gap, car score, cooling temperature, peer deviation, setpoint deviation, and data availability | Start with the leading car; inspect cooling performance and refrigerant circuit |
| Rail Corrugation | Normal, Side I, or Side II for each file | Side scores, decision threshold, speed, and comparison with the uploaded batch | Use acquisition metadata to inspect the indicated rail side |
| SHM | Fatigue-damage estimate for each file | Sample/cycle counts, stress range, damage, and comparison with compatible files | Review the highest comparable estimates and the asset's earlier checks |

The evidence panel deliberately says “model score” rather than “probability.” The underlying
classifiers have not been shown to return calibrated probabilities of physical failure.

## Relative comparisons

For each selected test case, the app extracts key statistics and compares them with relevant peers:

- Door movements are compared with other movements in the same continuous recording.
- ACV cars are compared with other cars in the same workbook.
- Rail and SHM files are compared with the other successfully processed files in the batch.

Each statistic includes the selected value, peer average, peer count, and a descriptive state:
within range, above/below average, or high/low outlier. Colour and text are both used. A two-standard-
deviation rule highlights an outlier only within the current upload; this is not a fleet normal
range or maintenance threshold. A single-item upload is explicitly labelled as having no peer
comparison.

This distinction is particularly important for SHM, where damage values should only be compared
between equivalent parts and recording periods, and for Rail, where speed affects the signal.

## Input quality and partial batches

Before inference, the browser checks file extension, empty files, duplicate names, item count, and
total upload size. The server repeats security and model-integrity checks.

Multi-file subsystems first run as one efficient batch. If that batch fails, the server checks each
file in isolation. Valid predictions are retained and malformed files are listed with a corrective
message. This avoids losing an entire inspection batch because of one bad acquisition. Door remains
a single-file operation because it represents one continuous stream.

After inference, the server validates official columns, class vocabulary, timestamps, car IDs,
finite SHM damage values, and complete filename coverage. Technician evidence is stored separately
from the official CSV.

## Durable review and handover

Live analyses are stored in a small local SQLite database. Each record contains:

- asset, location/component, acquisition time, and work order;
- input filenames, byte sizes, and SHA-256 hashes;
- model run ID, artifact hash, and processing time;
- official predictions and exact CSV bytes;
- non-submission evidence and any file-level failures; and
- review status, technician note, and update time.

The queue retains up to 500 checks and supports New, Acknowledged, Inspection scheduled, Resolved,
and False alert states. A plain-text inspection report makes the record usable outside the app.
Example predictions remain session-only and cannot enter submission exports.

The default database is `app/.nebulax/history.sqlite3`; `NEBULAX_DATA_DIR` selects another location.
Local history survives app restarts. Cloud Run storage remains instance-ephemeral unless the
deployment provides a durable mounted or external store.

## Interface simplification and accessibility

The primary navigation contains only New check, Review queue, and Help. Competition export is
inside a secondary disclosure, and internal validation scores are not shown during routine work.
The interface uses larger text and controls, explicit keyboard focus, 44-pixel minimum inputs,
responsive layouts, reduced-motion support, textual status labels, and keyboard-selectable result
rows. Files selected for one subsystem remain available when the technician temporarily switches
to another subsystem.

## Technical architecture

```text
Browser interface
    -> Python HTTP server
    -> file, context, and model-integrity checks
    -> common.inference.predict_with_evidence(...)
    -> official prediction frame + technician evidence
    -> schema validation and exact official CSV
    -> SQLite analysis/review record
```

The app uses static HTML, CSS, and JavaScript plus Python's standard library server and SQLite.
Uploads are written to an isolated temporary directory and deleted after inference. Before and
after a run, the server verifies the selected model's manifest and SHA-256 hash; a batch is rejected
if the model changes during processing.

The official `predict(...)` API and command-line outputs remain unchanged. The additional
`predict_with_evidence(...)` API returns evidence separately, so technician features cannot alter
the challenge CSV schema or full-precision prediction values.

## Submission compatibility

Submission export remains available under Review queue → Submission tools. It combines compatible
live runs into the four required CSV names at the ZIP root. Repeated filenames use the newest
prediction, Door uses the newest stream, and model hashes must agree within a subsystem. Asset
details, comparisons, review state, and notes are never added to submission CSVs.

## Validation

The app contract suite covers:

- upload and output validation;
- model manifest and checksum handling;
- asset context and review validation;
- durable history and restart-safe CSV export;
- partial success for malformed multi-file batches;
- evidence separation from official predictions;
- inspection report contents;
- duplicate/newest-wins submission behavior; and
- static technician workflow, accessibility, and comparison states.

Integration tests separately exercise the real model CLIs and packaged prediction parity when raw
datasets are available. Passing software tests establishes interface and data-contract behavior,
not model accuracy or fitness for service.

## Limitations

- The app is designed for one operator and has no accounts or concurrent-edit conflict handling.
- Relative comparison depends on the current upload; a small or mixed batch may be a poor reference.
- Asset history depends on consistent asset IDs and comparable recording conditions.
- Upload validation cannot detect every sensor calibration or acquisition fault.
- Maintenance actions remain subject to operating procedures and engineering judgement.

## Reproduction pointers

- Operator guide: `app/README.md`
- Local server and validation: `app/server.py`
- Durable store: `app/history.py`
- Browser behavior: `app/static/app.js`
- Page structure and styles: `app/static/index.html`, `app/static/style.css`
- Evidence helper: `common/evidence.py`
- Shared inference: `common/inference.py`
- App tests: `app/tests/`
- Submission packaging: `package.py`
