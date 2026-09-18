"""ACV inference: case .xlsx (or a directory of them) -> acv_predictions.csv rows.

CLI: python ACV/code/predict.py --input data/ACV/Test/acv_test_case.xlsx --output acv_predictions.csv
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
from ACV.code.pipeline import car_features, load_case, score_cars  # noqa: E402

from common.artifacts import load_active_model

MODEL_PATH = ROOT / "ACV/model/acv_model.joblib"


def _artefact():
    return load_active_model(MODEL_PATH.parent, (MODEL_PATH.name,), joblib.load)


def predict_one(path: Path, with_scores: bool = False, *, artifact=None):
    art = _artefact() if artifact is None else artifact
    long, cars, _ = load_case(path)
    F = car_features(long, cars, art["response"])
    s = score_cars(F, art["weights"])
    row = {"file_id": path.name, "ranked_cars": "|".join(s.index.tolist())}
    return (row, s) if with_scores else row


def predict(input_path: str | Path) -> pd.DataFrame:
    p = Path(input_path)
    files = sorted(p.glob("*.xlsx")) if p.is_dir() else [p]
    artifact = _artefact()
    return pd.DataFrame([predict_one(f, artifact=artifact) for f in files])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    predict(a.input).to_csv(a.output, index=False)
    print("wrote", a.output)
