# NebulaX technician app and maintenance workflow

## Purpose

The NebulaX Control Room translates sensor analysis across Doors, Air conditioning, Rail condition
and Structural health into information maintenance technicians can inspect and act on. It brings
four prediction pipelines into one browser interface, connecting each result with identifying
information, relevant measurements and comparisons with the uploaded dataset.

The interface is designed for technicians who already understand the maintenance task but should
not need to understand machine learning or write code to use the models. Routine views therefore
emphasise results and evidence. Longer explanations, interpretation guidance and model limitations
sit in **How to use this app**. The former Model setup page and demonstration walkthrough have
been removed from the technician workflow.

## Common workflow

1. Select a train system and add its original recordings.
2. Select **Check file** or **Check files**.
3. Review the completed result overview, then select a sequence, car or recording for details.
4. Reopen saved checks in **Previous results**, or download results for the selected tasks.

The upload panel includes a collapsed guide to required columns and optional measurements.
Doors accepts one continuous CSV recording; Air conditioning accepts Excel workbooks; Rail
condition and Structural health accept CSV recordings. The latter three tasks accept up to
100 files, each no larger than 120 MB in local operation.

Files in a batch are uploaded and processed sequentially. The main result panel shows progress
until the entire batch completes, rather than displaying a succession of partial results. If a
file fails, completed checks remain available in Previous results and only unfinished files
remain selected for retry. Server errors are translated into readable messages instead of exposing
JSON parsing errors or internal tracebacks.

For Doors, selecting a file replaces the upload box with its filename and size, preventing an
accidental second upload. Removing that file restores the chooser. The check action and
**View previously saved predictions** share an aligned action row; on narrow screens they stack
as full-width buttons. The latter action opens Previous results rather than example predictions.

Completed results and selected files remain available when switching systems. The selected system
and application page also survive a browser refresh; file selections themselves are not restored
after refresh. A system without completed results shows a clear empty state.

## Views designed around each task

### Doors: individual movement sequences

The Door dataset has no physical train, car or door identifiers. Accordingly, the overview shows
one numbered tile per detected opening or closing sequence, without assigning sequences to
physical doors. Green tiles indicate Normal; red tiles indicate Abnormal resistance. Text labels
accompany the colours.

Selecting a tile opens a detail panel with a large sequence identifier, status, and prominent
start and end times. Displayed times use the recording's clock and omit milliseconds; original
timestamp precision remains available internally and in the judging-format export.

Measurements include mid-travel and travel current, peak current, motor voltage, back-EMF,
movement duration, stationary samples under load, and deviation from the trained normal-current
profile. Opening and closing sequences have separate dataset comparison groups. Expandable traces
show motor current and door position, with selected original readings and unusual values.

These measurements support investigation of resistance faults. They are supporting evidence,
not a claim that one displayed measurement caused the classifier's decision.

### Air conditioning: car overview and inspection order

Cars appear in numerical car order in a clickable train overview. The first inspection priority
is red; the remaining cars are green and labelled **Lower priority**. The model produces a ranking,
so green does not mean that a car has been confirmed healthy. Side I/Side II labels are omitted
because they are irrelevant to this task.

Inspection order appears separately as a numbered list, rather than rearranging the train
illustration. A recording selector allows technicians to switch between uploaded cases, with
its label vertically aligned to the dropdown.

Selecting a car shows its identifier, source recording, recording period and inspection rank.
Measurements include valid cabin temperatures while cooling, temperature differences from peers,
cooling-setpoint error, temperature change, deviation from the learned temperature response,
invalid-reading fraction and pressure measurements where available. Comparisons use cars within
the same recording. Missing or invalid measurements remain unavailable rather than becoming zero.

### Rail condition: sortable condition columns

The Rail view uses a recording table without a train illustration. Each row has separate
**Normal**, **Side I** and **Side II** columns. The active result is highlighted; the other two
states are greyed out. Technicians can sort by condition, filter a specific condition or search
for a source filename.

Selecting a recording opens its source identifier and measurements: estimated speed, vibration
and shock RMS for each rail side, and dominant wavelength. Additional details show all 64 sensor
positions and a vibration trace from the highest-RMS channel on each side.

The target is rail corrugation, not every form of rail wear. Source filenames must still be
linked to acquisition locations using the team's recording records. Differences in speed and
sensor position can affect the displayed measurements.

### Structural health: damage by source recording

Horizontal bars rank fatigue-damage estimates from highest to lowest. Each bar is labelled with
the source filename without its extension, rather than a rank number, and shows its estimated
value. The full filename remains available in the detail panel. A searchable table provides
another route to individual recordings.

Selecting a recording shows estimated fatigue damage, mean stress, stress RMS, peak-to-peak stress,
largest stress-cycle range, counted cycles and a stress trace. Comparisons should use recordings
with matching measurement point, duration, line and load condition. The supplied recordings
represent healthy operation; a higher damage estimate is not a confirmed structural fault or
remaining-life forecast. Random file numbers do not establish recording chronology.

## Measurement comparisons and colour coding

The detail table shows the selected measurement, dataset mean, deviation from that mean and a
reference band. Comparison groups include the selected item:

- Doors: sequences of the same movement type in the uploaded recording.
- Air conditioning: cars in the same case file.
- Rail and Structural health: recordings in the completed check.

The reference band is the mean ±2 population standard deviations and requires at least three
valid items. The deviation column is calculated as:

```text
Deviation (% of SD) = 100 × (selected value − dataset mean) / standard deviation
```

For example, +150% means 1.5 standard deviations above the mean; −150% means the same distance
below it. Colour depends on the absolute distance, while the sign preserves the direction:

| Absolute deviation | Appearance |
|---|---|
| Below 50% SD | Dark green at the mean, fading towards white at 50% |
| 50–100% SD | Neutral white |
| Above 100% SD | Increasingly red, reaching dark red at 200% and above |

Missing data or insufficient comparison data shows a dash. When a valid comparison group has zero
variance, a value equal to its mean shows 0%. This colour scale is a visual measure of dataset
deviation, not the model's fault classification, a probability of failure or an approved operating
limit. A predicted abnormal result can still lie within the uploaded dataset's reference band.
Interpretation guidance is in the user guide rather than repeated above every detail table.

Trace outliers use a separate criterion: values outside that trace's mean ±3 population standard
deviations. Plots retain up to 180 evenly spaced original readings plus selected extreme points;
up to 24 of the largest outliers are tabulated. Sample indices refer to the selected sequence,
car time series or full recording, as labelled. These plots are sampled views, not a display of
every original row. Dataset bands do not adjust Rail or Structural health comparisons for
operating-condition differences.

## History and downloads

Results, file metadata, derived measurements and selected original measurement points are saved
in SQLite at `app/.local/results.sqlite3`. Local history survives browser refreshes and server
restarts and is excluded from Git, submission packages and container build inputs. Original
uploaded files are removed from temporary storage after processing.

Previous results supports filters by task and elapsed time: one day, one week or a chosen number
of hours. The filters have no surrounding borders or focus boxes; keyboard focus uses a subtle
background and text change. Entire history rows are clickable and open a separate saved-result
view with a back button. Rows show task, source filenames and check date without result counts or
execution times. Older checks remain readable but must be rerun to gain measurement evidence
that was not previously stored.

Individual checks offer CSV and JSON downloads. CSV headings describe the output, such as
**Estimated fatigue damage**, rather than a generic prediction label. JSON also includes the
measurement evidence, source-file metadata and model version.

**Download all check results** opens one checkbox per task. Each selected task includes the latest
result for each filename; Doors uses the latest continuous recording. If a task has multiple
source files, the ZIP contains a task folder with one result CSV per source file. Tasks with a
single recording have one CSV at the ZIP root. For example:

```text
selected_check_results.zip
├── door_check_results.csv
├── acv_check_results.csv
├── rail_condition/
│   ├── Test1_results.csv
│   └── Test9_results.csv
└── structural_health/
    ├── test01_results.csv
    └── test02_results.csv
```

Screen filters do not silently remove downloaded rows. Selected results for a task must use the
same model version. Numeric prediction precision is preserved in exports.

The organiser-submission section is removed from the interface. The separate `/api/export`
contract remains available for automated judging exports with the original required column names
and timestamps; the technician downloads use descriptive headings and the folder layout above.

## Technical architecture

```text
HTML / CSS / JavaScript interface
    -> Python HTTP server
    -> input and model-integrity checks
    -> common.inference.predict(...)
    -> selected subsystem model and output validation
    -> diagnostic measurement extraction
    -> SQLite history, interactive results and downloads
```

The application uses Python's standard-library HTTP server and SQLite, with vanilla JavaScript,
HTML, CSS and SVG in the browser. It requires no frontend build step. Inference and measurement
processing use the repository's scientific Python dependencies. `app/diagnostics.py` extracts
supporting measurements without changing model predictions. If extraction fails, the prediction
remains available and the interface explicitly reports that measurements are unavailable.

Model selection uses each subsystem's `active_model.json`. The server verifies its SHA-256
fingerprint before and after inference and rejects a result if the model changed. A lock
serialises inference, and combining checks requires consistent model hashes. The detailed model
methods and their validation are documented separately in the four subsystem writeups.

Local operation binds to `127.0.0.1` by default. Configurable host/port settings and Docker support
a separate private Cloud Run deployment, documented in `docs/cloud-run-deployment.md`. The server
accepts matching HTTP/HTTPS origins and HTTP loopback proxy origins and supplies a restrictive
content security policy. Cloud instance-local storage is not durable across container replacement;
a hosted deployment therefore does not inherit the local workstation's persistence guarantees.

## Validation

The final interface changes were checked using 25 Python tests and four JavaScript tests. Coverage
includes upload and output contracts, model integrity, persistent history, exact export precision,
selected-file and task-folder ZIP contents, comparison groups, missing values, extreme-sample
retention, and deviation calculations and colour ranges. Browser-logic tests also verify that
partial batch results remain hidden and that completed files survive a later upload failure.

Browser verification used official recordings for all four tasks, including a three-file Rail
batch, 38 Door sequences, eight Air-conditioning cars and three Structural health recordings.
Checks covered clickable details, sorting, tab persistence, task selection, downloads, control
alignment, focused history filters and responsive layout. These checks establish software
behaviour; they do not independently establish model accuracy or maintenance outcomes.

## Maintenance value and limits

The interface helps technicians move from a large recording set to a specific sequence, car or
source file, then inspect the evidence behind the inspection priority. Persistent records and
consistent downloads support handover and repeat review. Together, these features support
prioritisation of maintenance work and allocation of limited inspection resources.

The app remains a review tool rather than an autonomous maintenance controller. It has no
application-level accounts, work-order integration or multi-user access controls. It cannot infer
missing asset identities or track locations. Input checks cannot detect every acquisition fault,
and sampled traces are not a complete raw-data archive. Inspection, maintenance and service
release decisions remain subject to the team's procedures and engineering judgement.

## Reproduction pointers

- Usage and packaging: `app/README.md`
- HTTP API, validation, history and exports: `app/server.py`
- Diagnostic measurements: `app/diagnostics.py`
- Upload flow, navigation and history: `app/static/app.js`
- Technician result views, deviation colours and download selection: `app/static/technician.js`
- Page structure and styling: `app/static/index.html`, `app/static/style.css`
- Python tests: `app/tests/test_server.py`, `app/tests/test_diagnostics.py`
- Browser-logic tests: `app/tests/test_frontend.cjs`
- Shared inference and artifact selection: `common/inference.py`, `common/artifacts.py`
- Submission packaging: `scripts/package.py`

```sh
.venv/bin/python -m unittest discover -s app/tests -v
node --test app/tests/test_frontend.cjs
node --check app/static/app.js
node --check app/static/technician.js
```
