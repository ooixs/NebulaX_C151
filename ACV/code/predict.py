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
from common.evidence import comparison

MODEL_PATH = ROOT / "ACV/model/acv_model.joblib"


def _artefact():
    return load_active_model(MODEL_PATH.parent, (MODEL_PATH.name,), joblib.load)


def predict_one(path: Path, with_scores: bool = False, *, artifact=None, with_evidence: bool = False):
    art = _artefact() if artifact is None else artifact
    long, cars, _ = load_case(path)
    F = car_features(long, cars, art["response"])
    s = score_cars(F, art["weights"])
    row = {"file_id": path.name, "ranked_cars": "|".join(s.index.tolist())}
    if with_evidence:
        finite_scores = s[s.notna() & (s != float("-inf"))]
        leaders = finite_scores.iloc[:2]
        top_car = str(leaders.index[0]) if len(leaders) else None
        top_features = F.loc[top_car] if top_car in F.index else None
        car_items = []
        finite_score_values = finite_scores.tolist()
        for car, score in finite_scores.items():
            features = F.loc[car]
            metrics = [comparison("Model ranking score", score, finite_score_values, "", 3)]
            for column, label_text, unit in (
                ("t_in_mean", "Cooling temperature", "°C"),
                ("peer_dev_mean", "Temperature above peers", "°C"),
                ("set_err_mean", "Temperature above setpoint", "°C"),
            ):
                if column in F and pd.notna(features[column]):
                    metrics.append(comparison(label_text, features[column], F[column].dropna(), unit, 2))
            car_items.append({"car": str(car), "comparisons": metrics})
        evidence = {
            "top_car": top_car,
            "runner_up": str(leaders.index[1]) if len(leaders) > 1 else None,
            "score_gap": round(float(leaders.iloc[0] - leaders.iloc[1]), 6) if len(leaders) > 1 else None,
            "valid_readings": int(top_features["n_valid"]) if top_features is not None else 0,
            "cooling_fraction": round(float(top_features["cool_frac"]), 4) if top_features is not None else None,
            "pressure_evidence_available": bool(
                top_features is not None
                and "p_high_def" in top_features.index
                and pd.notna(top_features["p_high_def"])
            ),
            "model_scores": {str(car): round(float(score), 6) for car, score in finite_scores.items()},
            "items": car_items,
        }
        return row, evidence
    return (row, s) if with_scores else row


def predict(input_path: str | Path) -> pd.DataFrame:
    p = Path(input_path)
    files = sorted(p.glob("*.xlsx")) if p.is_dir() else [p]
    artifact = _artefact()
    return pd.DataFrame([predict_one(f, artifact=artifact) for f in files])


def predict_with_evidence(input_path: str | Path) -> tuple[pd.DataFrame, dict]:
    p = Path(input_path)
    files = sorted(p.glob("*.xlsx")) if p.is_dir() else [p]
    artifact = _artefact()
    rows, evidence = [], {}
    for file in files:
        row, details = predict_one(file, artifact=artifact, with_evidence=True)
        rows.append(row)
        evidence[file.name] = details
    return pd.DataFrame(rows), {"files": evidence}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    predict(a.input).to_csv(a.output, index=False)
    print("wrote", a.output)
