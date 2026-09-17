"""Door inference: continuous stream CSV -> door_predictions.csv rows.

CLI: python Door/code/predict.py --input data/Door/Test.csv --output door_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Door.code.pipeline import features_table, load_stream, segment, segments_to_frame  # noqa: E402

MODEL_PATH = ROOT / "Door/model/door_model.joblib"
_ART = None


def _artefact():
    global _ART
    if _ART is None:
        _ART = joblib.load(MODEL_PATH)
    return _ART


def predict(input_path: str | Path, with_confidence: bool = False) -> pd.DataFrame:
    art = _artefact()
    df = load_stream(input_path)
    segs = segment(df)
    X = features_table(segs, art["ref"])[art["feature_names"]]
    p = art["clf"].predict_proba(X)[:, 1]
    labels = ["Abnormal resistance" if v >= art["threshold"] else "Normal" for v in p]
    out = segments_to_frame(segs, labels)
    if with_confidence:
        out["confidence"] = p
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    predict(a.input).to_csv(a.output, index=False)
    print("wrote", a.output)
