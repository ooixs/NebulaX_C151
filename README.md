# NebulaX Train Condition Monitoring

Team C151's solution for Problem Statement 3 of the NebulaX Hackathon. The repository contains
four condition-monitoring models, a local web app for operators, the selected model artifacts,
and the code used to validate and package the submission.

The four subsystems address different maintenance questions:

| Subsystem | Operator question | Output |
|---|---|---|
| Door | Which door movements show abnormal resistance? | One result for each detected open or close cycle |
| ACV | Which train car should be checked first for a refrigerant leak? | Cars ranked from most to least likely faulty |
| Rail Corrugation | Does the recording indicate corrugation on either rail? | `Normal`, `Side I`, or `Side II` |
| SHM | How much fatigue damage is represented by this stress recording? | One cumulative-damage estimate per file |

## Quick start

Use Python 3.11 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app/server.py
```

On Windows, activate the environment with `.venv\Scripts\activate` and use `python` in place of
`python3`. Open <http://127.0.0.1:8765> after the server starts.

The app runs locally and does not require an internet connection. During normal use, an operator
does not need to use the command line after the server has started.

## Using the app

1. Select Doors, Air conditioning, Rail condition, or Structural health.
2. Drag in the original sensor files. The app shows the required file type and format.
3. Select **Check these files**.
4. Review the visual summary, detailed results, and suggested inspection action.
5. Download the result CSV or technical record. Results from compatible live runs can also be
   combined into `predictions.zip`.

Door accepts one continuous CSV stream. ACV accepts Excel files. Rail Corrugation and SHM accept
one or more CSV files. The app validates uploads before inference and reports format problems in
plain language.

See [the app guide](app/README.md) for the full operator workflow, upload limits, and packaging
instructions.

## Technical write-ups

- [Air-conditioning refrigerant-leak ranking](writeup_ACV.md)
- [Door-cycle segmentation and resistance classification](writeup_Door.md)
- [Rail Corrugation classification](writeup_Rail_Corrugation.md)
- [Structural Health Monitoring fatigue estimation](writeup_SHM.md)
- [Operator app and maintenance workflow](writeup_App.md)

### Rail research updates

- [Submission feedback and decision-layer experiment](docs/rail_decision_experiment.md)
- [Feature, MiniROCKET, and ensemble experiments](docs/rail_model_experiments.md)

The reported first-submission scores were Door 1.0000, ACV 1.0000, Rail 0.7448, and
SHM 0.9725, giving an overall score of 0.9293. These organiser-reported results are
separate from the internal validation estimates below. Later Rail candidates remain
separate from the active app model; their local archives and reproduction steps are
documented in the research updates.

## Approach and results

The four tasks share one principle: validation follows the unit that will be predicted. Door data
is split in contiguous cycle blocks, ACV is evaluated by leaving out complete fault cases, and
Rail Corrugation and SHM are split by complete files. References, thresholds, calibration
constants, and learned corrections are fitted using training folds only. This avoids the common
failure mode of obtaining a strong score from rows or derived features that leak information
between training and validation.

The models are deliberately matched to the data and maintenance problem. Door classification
uses phase-specific motor-current features. ACV compares each car with its peers under the same
conditions. Rail Corrugation uses speed-aware spectral features and compares the two rail sides.
SHM starts from rainflow counting and Miner's rule, then applies a small learned correction only
after the physical model's residuals were evaluated.

| Subsystem | Method used by the app | Internal validation | Result |
|---|---|---|---:|
| Door | Timestamp-gap segmentation and RF/logistic-regression ensemble | 5 contiguous blocks with inner blocked threshold selection | IoU-weighted F1 **1.000** |
| ACV | Peer-relative temperature ranking with a pressure-deficit term when available | Leave-one-case-out over 6 cases | Rank-decay **1.000** |
| Rail Corrugation | Shared per-side ExtraTrees detector on spectral and side-relative features | 5-fold x 3 repeats with fold-local references and nested threshold selection | Macro F1 **0.8232 +/- 0.0167** |
| SHM | Rainflow and Miner's rule with a Huber log-residual correction | 8-fold x 3 seeds over 64 files | 1 - MAPE **0.9797 +/- 0.0011** |

These are estimates from the labelled training data, not scores on the organisers' private test
labels. They also have different levels of certainty. Door has a clear class separation in the
available data, while ACV has only six labelled cases. Rail Corrugation has 14 Side I examples
and is sensitive to speed and acquisition grouping. SHM's learned correction is fitted on 64
files. The detailed write-ups report these limitations and the alternative validation checks.

Together, the system supports maintenance triage: it locates suspicious door movements, orders
cars for ACV inspection, indicates the rail side associated with corrugation, and compares fatigue
damage across compatible recordings. The app presents these as inspection priorities, not as
confirmed diagnoses or remaining-life forecasts.

## Repository layout

```text
.
├── Door/                         # Door model, prediction and training code
├── ACV/                          # Air-conditioning model and code
├── Rail Corrugation/             # Rail Corrugation model and code
├── SHM/                          # Fatigue-damage model and code
├── app/                          # Local operator web app
├── common/                       # Shared inference, metrics and artifact handling
├── docs/references/              # Organisers' subsystem information kits
├── predictions/                  # Current prediction CSVs
├── tests/                        # Metric, inference and packaging tests
├── package.py                    # Builds the C151 submission directory
└── writeup_*.md                  # Detailed technical write-ups
```

The selected model in each subsystem is named by `model/active_model.json`. The manifest records
the artifact filename, run ID, and SHA-256 checksum. The app refuses to use a missing or
checksum-mismatched selected model.

Raw datasets are not committed. For research or full inference tests, copy the organisers'
`02_Datasets/<Subsystem>` directories into `data/<Subsystem>` with the same structure.

## Prediction interfaces

The app uses the shared interface:

```python
from common.inference import predict

result = predict("Rail Corrugation", input_path)
```

Each subsystem also provides a command-line wrapper:

```sh
python "Rail Corrugation/code/predict.py" --input data/Rail_Corrugation/Test --output rail_predictions.csv
```

Use only trusted model files. Joblib and pickle artifacts can execute Python code when loaded.

## Testing

Run the app contract tests and JavaScript syntax check without the raw datasets:

```sh
python -m unittest discover -s app/tests -v
node --check app/static/app.js
```

When the datasets are available, run the complete model and packaging checks:

```sh
python -m pytest tests -q
```

The test suites cover upload validation, official output schemas, metric implementations, model
integrity checks, shared inference, CSV export, and packaged prediction parity. App tests verify
software behaviour; model accuracy is assessed separately by the validation described in the
technical write-ups.

## Packaging

```sh
python package.py --dest C151-new
```

The destination must not already exist. The command creates the self-contained app,
`predictions.zip`, and `Optional_Items/` with the five write-ups and the selected model/code for
each subsystem. Add the required demo video to the resulting directory separately.

## Important limitations

- Private held-out labels are unavailable. Reported submission scores come from organiser
  feedback and cannot be independently recomputed from the repository.
- The app supports an operator's inspection decision; it does not replace physical inspection or
  engineering judgement.
- Results from different files are comparable only when the recordings represent compatible
  assets and operating periods.
- The local server binds to `127.0.0.1` and is intended for one operator. It is not configured as
  a public or multi-user service.
