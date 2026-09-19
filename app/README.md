# NebulaX Control Room

A local, responsive train condition monitoring app for Team C151. All application source is in this folder. It integrates the existing models through `common.inference.predict(subsystem, input_path)`; it does not change trained models, thresholds or prediction logic.

For a private hosted installation, see the repository's
[Google Cloud Run deployment guide](../docs/cloud-run-deployment.md). Local operation remains
the default when sensor files must stay on the operator's computer.

## Start

From the repository root, using Python **3.11 or newer**:

```sh
python3 app/server.py
```

Open http://127.0.0.1:8765. On Windows use `python app/server.py`. Choose another port with `--port 8766` if needed. No Node build step, CDN, web framework or internet connection is needed. The interface and saved-result history run with Python's standard library alone.

For live inference, install the repository's inference dependencies into your environment first:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app/server.py
```

On Windows, activate with `.venv\Scripts\activate`. Prefer the same Python and library versions used to train the serialized models; the packaging script records installed versions in `requirements-runtime.txt`.

## Included models

The four supplied model binaries and their `active_model.json` manifests are included in the repository. The Git ignore rules explicitly allow these eight files; experimental checkpoints remain ignored. The app loads them from the subsystem folders at the repository root. Packaging copies them into the same subsystem folders **inside `C151/app/`**, alongside the inference code, so the packaged app is self-contained.

| Subsystem folder | Artifact | Requested research run |
| --- | --- | --- |
| `Door/model/` | `door_model.joblib` | `2026-09-18-autoresearch` |
| `ACV/model/` | `acv_model.joblib` | `2026-09-18-autoresearch` |
| `Rail Corrugation/model/` | `rail_model.joblib` | `2026-09-19-legacy_sg_mf50_leaf2_ensemble-forest` |
| `SHM/model/` | `shm_model.joblib` | `2026-09-18-sg-branch3` |

Rail uses the checkpoint from the submission with user-reported macro F1 **0.8261904762**. It blends the legacy spectral plus SG random forest (75%) with the SG random forest (25%), with threshold `tau=0.20`. All 68 Rail predictions were reproduced against that submission. Checkpoint-specific metadata remains available from the status API when its verified hash matches `app/model_details.json`.

The upload page disables unavailable checks when a model file or its manifest is missing or invalid. Restore only trusted trained artifacts; serialized Python models execute code when loaded. If restoring an original trusted artifact without its manifest, use the existing `python -m common.artifacts --activate <artifact-path> --run-id <run-id>` command. The app deliberately requires explicit selection for every model, including SHM's alternate JSON format. Refresh the page after restoration.

## Operator workflow

1. Select Doors, Air conditioning, Rail condition or Structural health. Expand the column guide when checking the data layout.
2. Drag original data files into the upload area. Doors accepts one continuous stream; the other systems accept up to 100 files, each no larger than 120 MB. Files are uploaded and checked one at a time, with progress shown for each recording.
3. Select **Check file** or **Check files**. Results appear after the entire batch completes. Select a sequence, car or recording to review measurements and source readings. The selected system and page survive refresh; switching systems retains completed results and selected files.
4. Download CSV or JSON with descriptive result headings. Door times use the recording’s clock and are displayed to whole seconds; original precision is retained internally. JSON includes measurement evidence.
5. **Previous results** provides filters by check type and elapsed time, with a separate detail view and back button. Results are saved in `app/.local/results.sqlite3` and survive browser refreshes and server restarts. This local history is excluded from Git and submission packages.
6. **Download all check results** opens one checkbox per task. Each selected task includes the latest result for every filename (and the latest Door recording). A task with multiple source files gets its own folder of individual result CSVs; a single recording gets one CSV at the ZIP root. The separate `/api/export` contract still supports automated judging exports, but is not shown in the technician interface. Contributing checks must use the same model version.

If a queued file fails, completed files remain in Previous results and only unfinished files remain selected. Input uploads are deleted after inference; history retains results, file metadata, derived metrics and selected original measurement points. By default, the server binds to 127.0.0.1 for one local operator. The separate Cloud Run mode uses `HOST` and `PORT`; its instance-local history is lost when the container is replaced. See the Cloud Run guide for hosted-mode limits.

Rail checks target **corrugation**, a repeated pattern of uneven wear, rather than all rail defects. Structural health estimates damage during equal-length recordings from healthy operation: comparisons need the same measurement point, line and AW0/AW4 load condition. Random filenames do not identify the recording order.

## Measurement evidence

`diagnostics.py` extracts evidence after inference without changing any model or prediction:

- Doors: individual movement sequences; current, voltage, back-EMF, duration, stationary samples and deviation from the trained normal-current profile. No physical door/car identity is inferred. Opening and closing sequences are compared separately.
- Air conditioning: cars in numeric order, first inspection priority in red and lower priorities in green; a separate ordered list. Details compare valid cooling temperatures, peer differences, learned response residuals, setpoint error and available pressures. This model ranks cars; it does not classify the remaining cars as confirmed healthy.
- Rail: sortable Normal / Side I / Side II columns and recording details with speed, side-level vibration/shock RMS, wavelengths, all 64 sensor-position summaries, and a representative trace from each side’s highest-RMS channel.
- Structural health: filename-labelled damage bars, stress statistics, counted cycles and original stress readings.

Metric bands are the uploaded comparison group's mean ±2 population standard deviations, including the selected item, with at least three valid items. Trace flags use that trace's mean ±3 population standard deviations. The deviation column shows 100 × (value − mean) / SD. It fades from dark green at the mean to white at 1 SD, stays neutral through 2 SD, and fades to dark red at 3 SD. Zero-variance data shows 0% at the mean; insufficient data shows a dash. These are descriptive statistical bands, not engineering limits or causal explanations of model decisions. Missing values remain unavailable. Rail/SHM batch comparison does not correct for speed, asset, line or load differences.

Traces retain up to 180 evenly spaced original readings plus extreme values; sample indices refer to the selected sequence, car time series or full recording, as labelled. The 24 largest outliers are also tabulated. ACV invalid readings excluded by its loader remain unavailable. Older saved results can be opened but must be rechecked for measurement evidence.

## Judging alignment

Based on [PS3 specifications](https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/blob/main/PS3/01_Problem_Statement_3_Specifications.md):

- **Ease of use:** one selection/upload/analyze flow, drag and drop, format guidance, actionable errors, responsive layout and direct downloads.
- **Clarity:** Door cycle sequence and exact intervals; ACV ordered car cards; rail class distribution and side localisation; SHM comparative damage bars. Tables expose individual predictions, and full-precision export is independent of UI filters.
- **Usefulness:** inspection suggestions tied to model outputs, explicit model limits, batch handling, persistent history and integrity-checked provenance.
- **Problem fit:** all four subsystems, clear maintenance questions and concise input guidance. No unsupported confidence probabilities, asset positions, remaining-life forecasts or held-out scores.
- **Technical execution:** the existing shared model interface produces the official output schema. Operator exports use descriptive headings; judging exports retain the official schema. Both preserve numeric precision.

A high placing cannot be guaranteed: held-out model performance and judges' assessments remain independent of the interface.

## Validation and packaging

```sh
python3 -m unittest discover -s app/tests -v
node --check app/static/app.js
node --check app/static/technician.js
node --test app/tests/test_frontend.cjs
```

The tests cover schema checks, upload boundaries, the shared prediction interface for all four subsystems (with injected test predictors), checksum validation, model changes during inference, exact CSV export, multi-batch merging, duplicate handling and preview exclusion. These are app contract tests. Run `python -m pytest tests -q` for real-model inference and packaged prediction parity checks; these also require the scientific dependencies and source datasets.

Run `python package.py` from the repository root to copy this folder, the shared inference modules and the selected models into a self-contained submission app. Inside `C151/app/`, run `python server.py` using the inference environment. For a new environment, install `requirements-runtime.txt` from that folder to use the recorded library versions. The server detects either repository or packaged layout automatically. Packaging refuses to overwrite an existing directory; use `python package.py --dest C151-new` for a fresh build.
