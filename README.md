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
