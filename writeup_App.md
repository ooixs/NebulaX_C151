# Operator app and maintenance workflow

## Purpose

The NebulaX Control Room gives train operators and maintenance staff one place to use all four
condition-monitoring models. Its purpose is not simply to display a predicted class. It turns an
uploaded recording into a clear inspection priority, preserves the technical evidence for that
run, and exports the official result format without requiring the operator to write code.

The interface is designed for a user who understands train operations but may not know machine
learning or Python. Technical model details remain available in the Model setup view without
interrupting the normal workflow.

## Operator workflow

The four subsystems use the same five-step interaction:

1. Select Doors, Air conditioning, Rail condition, or Structural health.
2. Drag the original files into the upload area or choose them from the device.
3. Select **Check these files**.
4. Review the summary, individual results, and **What to check next** guidance.
5. Download the result CSV, technical record, or combined submission ZIP.

The upload area changes with the selected subsystem. It explains the expected extension, number
of files, and important schema details before analysis begins. Door accepts one continuous CSV;
ACV accepts Excel workbooks; Rail Corrugation and SHM accept batches of CSV files.

This consistent workflow reduces the need to learn four separate tools while respecting the
different input and output formats.

## Designed for non-technical users

### Plain language

The main interface uses operational names such as **Doors**, **Air conditioning**, **Rail
condition**, and **Structural health**. It explains the task before asking for files. Terms such
as macro F1, model checksum, and run ID are kept in expandable technical sections.

Errors state what the operator can do next. Examples include choosing the correct file type,
removing a duplicate filename, splitting a batch above the upload limit, or restoring a missing
model file. Internal tracebacks are kept in the server log rather than shown as the primary
message.

### Guided input checks

Before inference, the app checks:

- the selected subsystem and file extension;
- that Door receives exactly one continuous stream;
- duplicate, unsafe, or empty filenames;
- the limit of 100 files and 120 MB per batch; and
- whether the selected model artifact matches its saved checksum.

After inference, it checks the required columns, class vocabulary, timestamps, file coverage,
car identifiers, and finite non-negative SHM estimates. An invalid result is not offered as a
submission file.

### One local action

After the server has been started, normal operation takes place entirely in the browser. There is
no Node build step, external web service, or internet dependency. The app runs at a local address
and is suitable for a workstation used to review collected maintenance data.

## Clear and useful results

Each subsystem has a result view suited to its maintenance question.

| Subsystem | Presentation | Maintenance use |
|---|---|---|
| Door | Cycle sequence, exact start/end times, Normal or Abnormal resistance | Review flagged intervals, then inspect the corresponding door movement, track, seals, and motor |
| ACV | Cars ordered from most to least likely faulty | Start refrigerant checks with the first car, then follow the ranking if the first inspection is inconclusive |
| Rail Corrugation | Recording-level class distribution and indicated side | Link flagged files to acquisition locations and inspect the indicated rail side |
| SHM | Comparative damage bars and numeric estimates | Prioritise engineering review of the highest values among compatible recordings |

Search and filters help an operator work through larger result sets. The full table remains
downloadable even when the on-screen view is filtered.

The app does not show unsupported probability percentages. A model score is not automatically a
calibrated chance of failure, and presenting it as one could mislead an operator. Instead, the
interface states what the result means and what physical check should follow.

## Fit with maintenance workflows

The app supports the handoff from condition-monitoring data to targeted inspection:

```text
Recorded sensor data
        |
        v
File and model checks
        |
        v
Prioritised condition result
        |
        v
Physical inspection or engineering review
        |
        v
Downloaded result and technical record
```

This fits several stages of train maintenance:

- **Screening:** process many recordings with one repeatable interface.
- **Prioritisation:** identify the door movement, car, rail side, or stress recording that should
  be reviewed first.
- **Inspection support:** connect the result to a concrete next check instead of presenting an
  isolated model label.
- **Handover:** export the official CSV and a JSON technical record containing filenames, hashes,
  model version, timing, and results.

The app supports a maintenance decision; it does not close the work order by itself. Door and
Rail results are not proof that an asset is fault-free, an ACV ranking is not a confirmed leak,
and an SHM value is not a remaining-life forecast.

## Brief technical architecture

```text
Browser interface
    -> local Python HTTP server
    -> upload and model-integrity checks
    -> common.inference.predict(...)
    -> selected subsystem model
    -> validated table, chart, CSV and JSON record
```

The browser uses static HTML, CSS, and JavaScript. `app/server.py` provides a local HTTP API with
Python's standard library. It writes each upload to an isolated temporary directory, calls the
shared inference layer, validates the returned table, and removes the temporary files when the
run finishes.

The shared inference layer loads the model selected by each subsystem's `active_model.json`.
Before a run, the server verifies the artifact's SHA-256 checksum. It checks again afterwards and
rejects the output if the model changed during inference. An inference lock also ensures that one
batch uses one consistent model snapshot.

The server binds to `127.0.0.1`. Requests that declare a different origin are rejected, and
responses use a restrictive content security policy. This is a local single-operator tool, not a
public hosted service.

## Results, history, and export

The app keeps the most recent 40 runs in server memory. Browser refreshes can recover them while
the server continues running. They disappear when the process stops, so the operator should
download any required records.

Live results retain the predictor's exact CSV representation. The combined export places the
official subsystem CSVs at the top level of `predictions.zip`. For multi-file subsystems, later
runs can add files or replace a repeated filename. Batches are combined only when they used the
same model hash. Door uses the newest continuous-stream run because combining segments from two
streams would be ambiguous.

Saved example results are clearly labelled as previews. They do not run a model, do not establish
the provenance of the old input files, and cannot be included in a live submission archive.

## Validation

The app contract tests cover:

- subsystem-specific upload limits and error messages;
- model manifest and checksum checks;
- all four calls through the shared prediction interface using injected test predictors;
- output schema and value validation;
- model changes during inference;
- exact CSV downloads;
- multi-batch merging and duplicate handling; and
- exclusion of preview data from submissions.

Integration tests separately exercise the real selected models, the four prediction CLIs, and
packaged inference when source datasets are available. These tests show that the software passes
data between components and preserves the official output schemas. They do not establish model
accuracy; model validation is documented in the four subsystem write-ups.

## Limitations

- The server is designed for one local operator. It has no accounts, persistent database, or
  multi-user access controls.
- Run history is held in memory and is lost when the server stops.
- The app cannot map a recording to an asset or track location unless that context is preserved
  outside the supplied file.
- Upload validation can detect structural problems, but not every possible sensor or acquisition
  fault.
- Maintenance actions remain subject to operator procedures and engineering judgement.

## Reproduction pointers

- Operator guide: `app/README.md`
- Local server and validation: `app/server.py`
- Browser behaviour and subsystem explanations: `app/static/app.js`
- Page structure: `app/static/index.html`
- App contract tests: `app/tests/test_server.py`
- Shared inference: `common/inference.py`
- Model selection and integrity: `common/artifacts.py`
- Submission packaging: `package.py`
