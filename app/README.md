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

Open http://127.0.0.1:8765. On Windows use `python app/server.py`. Choose another port with `--port 8766` if needed. No Node build step, CDN, web framework or internet connection is needed. The interface and saved-result previews run with Python's standard library alone.

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
| `Rail Corrugation/model/` | `rail_model.joblib` | `2026-09-18-autoresearch` |
| `SHM/model/` | `shm_model.joblib` | `2026-09-18-sg-branch3` |

Model setup reports a missing file, invalid manifest or checksum mismatch. Restore only trusted trained artifacts; serialized Python models execute code when loaded. If restoring an original trusted artifact without its manifest, use the existing `python -m common.artifacts --activate <artifact-path> --run-id <run-id>` command. The app deliberately requires explicit selection for every model, including SHM's alternate JSON format. Select Check setup again after restoration.

## Operator workflow

1. Select Doors, Air conditioning, Rail condition or Structural health.
2. Drag original data files into the upload area. Door accepts one continuous stream; the others accept up to 100 files and 120 MB per batch. Split large test sets across batches.
3. Select Check these files. Review the visual summary, What to check next, and the searchable results table. Select Download results (CSV) to save your results.
4. Select Technical record (JSON) if you need input hashes, model version, timing and output evidence.
5. Select Download submission, then Download ZIP file to save `predictions.zip`. Live batches are combined by subsystem. Repeated filenames use their newest prediction; Door uses only the newest stream. Contributing batches must use the same model hash. The ZIP has only the official prediction CSVs at its top level.

The most recent 40 runs are kept in server memory and restored after browser refresh. They disappear when the server stops or a cloud instance is replaced. Input uploads are placed in an isolated temporary directory and deleted after inference. Download records and exports before stopping. In local mode, the server binds only to 127.0.0.1 and is intended for one operator. The Cloud Run guide documents the separate private hosted mode and its limits.

“View example results” reads the repository's existing `predictions/` CSVs. It does **not** run a model or verify the historic model/input provenance. These results are visibly labelled and cannot be included in a submission archive. A packaged app without these CSVs still supports live inference.

## Judging alignment

Based on [PS3 specifications](https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/blob/main/PS3/01_Problem_Statement_3_Specifications.md):

- **Ease of use:** one selection/upload/analyze flow, drag and drop, format guidance, actionable errors, responsive layout and direct downloads.
- **Clarity:** Door cycle sequence and exact intervals; ACV ordered car cards; rail class distribution and side localisation; SHM comparative damage bars. Tables expose individual predictions, and full-precision export is independent of UI filters.
- **Usefulness:** inspection suggestions tied to model outputs, explicit model limits, batch handling, session history and integrity-checked provenance.
- **Problem fit:** all four subsystems, clear maintenance questions, model methods and research scores in Model setup. No unsupported confidence probabilities, asset positions, remaining-life forecasts or held-out scores.
- **Technical execution:** the existing shared model interface produces the official output schema. Live CSV downloads retain the predictor's CSV representation; combined exports preserve numeric strings without rounding. Preview results never enter a submission.

A high placing cannot be guaranteed: held-out model performance and judges' assessments remain independent of the interface.

## Three-minute demo

Use the included models and actual held-out inputs, not saved-result previews.

- **0:00–0:25:** explain the maintenance question and four-subsystem coverage.
- **0:25–1:15:** select a subsystem, upload data and run inference.
- **1:15–2:15:** inspect a flagged result and suggested check; show the other subsystem views using live analyses prepared earlier in the same session.
- **2:15–3:00:** show model provenance, download a CSV and export the combined archive.

Record a real screen video no longer than three minutes and add it to the final team submission folder. This implementation does not manufacture a demo recording or claim previews as real inference.

## Validation and packaging

```sh
python3 -m unittest discover -s app/tests -v
node --check app/static/app.js
```

The tests cover schema checks, upload boundaries, the shared prediction interface for all four subsystems (with injected test predictors), checksum validation, model changes during inference, exact CSV export, multi-batch merging, duplicate handling and preview exclusion. These are app contract tests. Run `python -m pytest tests -q` for real-model inference and packaged prediction parity checks; these also require the scientific dependencies and source datasets.

Run `python package.py` from the repository root to copy this folder, the shared inference modules and the selected models into a self-contained submission app. Inside `C151/app/`, run `python server.py` using the inference environment. For a new environment, install `requirements-runtime.txt` from that folder to use the recorded library versions. The server detects either repository or packaged layout automatically. Packaging refuses to overwrite an existing directory; use `python package.py --dest C151-new` for a fresh build.
