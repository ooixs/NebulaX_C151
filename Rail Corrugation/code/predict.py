"""Rail Corrugation inference: CSV file or directory -> rail_predictions.csv rows.

CLI: python "Rail Corrugation/code/predict.py" --input data/Rail_Corrugation/Test --output rail_predictions.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rail_corrugation.pipeline import CLASSES, aggregate, file_channel_table, side_relative_rows  # noqa: E402
from rail_corrugation.decision import decode, legacy_decision  # noqa: E402
from common.artifacts import load_active_model, load_rail_model

MODEL_PATH = ROOT / "Rail Corrugation/model/rail_model.joblib"


def _artefact():
    return load_active_model(MODEL_PATH.parent, (MODEL_PATH.name,), load_rail_model)


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def predict_one(path: Path, with_proba: bool = False, *, artifact=None):
    art = _artefact() if artifact is None else artifact
    ct, v = file_channel_table(path)
    engineered = art.get("engineered_features", False)
    feats = aggregate(art["ref"].excess(ct, v), v, paired=engineered)
    if art["choice"] == "side_detector":
        r1, r2 = side_relative_rows(feats, engineered=engineered)
        R = pd.DataFrame([r1, r2]).reindex(columns=art["det_cols"]).fillna(-9)
        p1, p2 = art["det"].predict_proba(R)[:, 1]
        label = str(decode([p1], [p2], art.get("decision") or legacy_decision(art["tau"]))[0])
        proba = {"Normal": 1 - max(p1, p2), "Side I": p1, "Side II": p2}
    else:
        X = pd.DataFrame([feats]).reindex(columns=art["clf_cols"]).fillna(-9)
        pr = art["clf"].predict_proba(X)[0]
        proba = {c: float(pr[list(art["clf"].classes_).index(c)]) for c in CLASSES}
        label = max(proba, key=proba.get)
    row = {"file_id": path.name, "prediction": label}
    if with_proba:
        row.update({f"p_{k}": v for k, v in proba.items()}); row["speed_mps"] = v
    return row


def predict(input_path: str | Path, with_proba: bool = False) -> pd.DataFrame:
    p = Path(input_path)
    files = sorted(p.glob("*.csv"), key=_natural_key) if p.is_dir() else [p]
    artifact = _artefact()
    return pd.DataFrame([predict_one(f, with_proba, artifact=artifact) for f in files])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    predict(a.input).to_csv(a.output, index=False)
    print("wrote", a.output)
