# NebulaX Control Room

A responsive train condition monitoring app for Team C151. It integrates the existing models
through `common.inference.predict_with_evidence(subsystem, input_path)` without changing trained
models, thresholds, or official prediction columns. Additional evidence is kept in the technician
record and never added to the challenge CSV.

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

## Technician workflow

1. Select Doors, Air conditioning, Rail condition, or Structural health.
2. Record the train or asset ID, component/location, acquisition time, and work order. These fields
   connect a prediction to the part that must be inspected.
3. Add original data files. Door accepts one stream; the others accept up to 100 files and 120 MB
   per local batch. The file list shows basic readiness before analysis.
4. Select **Check files**. If a multi-file batch contains a malformed file, the app isolates it and
   keeps valid results rather than discarding the whole batch.
5. Select a door movement, ACV car, Rail file, or SHM file to see its key measurements beside the
   average for the current recording or batch. Colours describe relative position only; they are
   not fleet thresholds or confirmed faults.
6. Save the inspection status and technician note. The Review queue can reopen the analysis and
   download a plain-text inspection report or the official result CSV.

Completed live checks are stored in `app/.nebulax/history.sqlite3` by default, up to the most recent
500 records. Set `NEBULAX_DATA_DIR` to place this database elsewhere. Original uploads are still
written only to an isolated temporary directory and deleted after inference; the database stores
results, hashes, evidence, context, notes, and the exact official CSV. Example previews stay in
memory and are not added to the durable queue.

Local SQLite history survives app restarts. A Cloud Run instance filesystem is ephemeral, so a
hosted deployment needs a mounted durable store or external persistence if cross-revision history
is required. The app remains a single-operator tool and has no account or multi-user editing model.

Challenge export is intentionally secondary. Open **Review queue → Submission tools** to create
`predictions.zip`. Repeated filenames use their newest prediction; Door uses the newest stream;
and contributing batches must use the same model hash. Asset context and technician notes never
enter the official CSVs.

“View example results” reads the repository's existing `predictions/` CSVs. It does **not** run a model or verify the historic model/input provenance. These results are visibly labelled and cannot be included in a submission archive. A packaged app without these CSVs still supports live inference.

## Judging alignment

Based on [PS3 specifications](https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/blob/main/PS3/01_Problem_Statement_3_Specifications.md):

- **Ease of use:** one selection/upload/analyze flow, drag and drop, format guidance, actionable errors, responsive layout and direct downloads.
- **Clarity:** selectable Door movements, ACV cars, Rail recordings, and SHM recordings expose the
  measurements that drove prioritisation and compare them with peers in the same upload.
- **Usefulness:** asset identification, partial batch recovery, durable review status, technician
  notes, inspection reports, and history connect model output to physical follow-up.
- **Problem fit:** all four subsystems retain their distinct maintenance question without showing
  model scores as calibrated probabilities or fatigue damage as remaining life.
- **Technical execution:** the existing shared model interface produces the official output schema. Live CSV downloads retain the predictor's CSV representation; combined exports preserve numeric strings without rounding. Preview results never enter a submission.

A high placing cannot be guaranteed: held-out model performance and judges' assessments remain independent of the interface.

## Three-minute demo

Use the included models and actual held-out inputs, not saved-result previews.

- **0:00–0:25:** explain the maintenance question and four-subsystem coverage.
- **0:25–1:15:** select a subsystem, upload data and run inference.
- **1:15–2:15:** select a flagged item, compare its measurements with the batch, and record the
  inspection status.
- **2:15–3:00:** open the durable Review queue and download an inspection report or official CSV.

Record a real screen video no longer than three minutes and add it to the final team submission folder. This implementation does not manufacture a demo recording or claim previews as real inference.

## Validation and packaging

```sh
python3 -m unittest discover -s app/tests -v
node --check app/static/app.js
```

The tests cover schema checks, upload boundaries, model checksums, persistent history, asset
context, review updates, inspection reports, partial batch recovery, evidence separation, exact
CSV export, duplicate handling, preview exclusion, and the static technician workflow. Run
`python -m pytest tests -q` for model inference and packaged prediction parity checks; these also
require the scientific dependencies and source datasets.

Run `python package.py` from the repository root to copy this folder, the shared inference modules and the selected models into a self-contained submission app. Inside `C151/app/`, run `python server.py` using the inference environment. For a new environment, install `requirements-runtime.txt` from that folder to use the recorded library versions. The server detects either repository or packaged layout automatically. Packaging refuses to overwrite an existing directory; use `python package.py --dest C151-new` for a fresh build.
