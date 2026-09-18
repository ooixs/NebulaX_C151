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
from Door.code.pipeline import features_table, load_stream, operation_of, segment, segments_to_frame  # noqa: E402

from common.artifacts import load_active_model
from common.evidence import comparison

MODEL_PATH = ROOT / "Door/model/door_model.joblib"


def _artefact():
    return load_active_model(MODEL_PATH.parent, (MODEL_PATH.name,), joblib.load)


def predict_with_evidence(input_path: str | Path) -> tuple[pd.DataFrame, dict]:
    art = _artefact()
    df = load_stream(input_path)
    segs = segment(df)
    X = features_table(segs, art["ref"])[art["feature_names"]]
    p = art["clf"].predict_proba(X)[:, 1]
    labels = ["Abnormal resistance" if v >= art["threshold"] else "Normal" for v in p]
    out = segments_to_frame(segs, labels)
    comparison_features = {
        "duration": ("Movement duration", "s", 3),
        "trav_mean": ("Average travel current", "", 2),
        "exc_mid_mean": ("Mid-travel resistance signal", "", 2),
    }
    populations = {
        feature: X[feature].astype(float).tolist()
        for feature in comparison_features
        if feature in X
    }
    populations["duration"] = [
        (movement["t"].iloc[-1] - movement["t"].iloc[0]).total_seconds()
        for movement in segs
    ]
    items = []
    for index, (movement, score, label) in enumerate(zip(segs, p, labels), start=1):
        duration = (movement["t"].iloc[-1] - movement["t"].iloc[0]).total_seconds()
        comparisons = []
        for feature, (label_text, unit, digits) in comparison_features.items():
            if feature not in populations:
                continue
            value = duration if feature == "duration" else X.iloc[index - 1][feature]
            population = populations[feature]
            comparisons.append(comparison(label_text, value, population, unit, digits))
        items.append({
            "movement": index,
            "operation": operation_of(movement),
            "duration_seconds": round(float(duration), 3),
            "model_score": round(float(score), 6),
            "decision_threshold": round(float(art["threshold"]), 6),
            "threshold_margin": round(float(abs(score - art["threshold"])), 6),
            "result": label,
            "comparisons": comparisons,
        })
    evidence = {
        "summary": {"samples": int(len(df)), "movements": len(segs)},
        "items": items,
    }
    return out, evidence


def predict(input_path: str | Path, with_confidence: bool = False) -> pd.DataFrame:
    out, evidence = predict_with_evidence(input_path)
    if with_confidence:
        out["confidence"] = [item["model_score"] for item in evidence["items"]]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    predict(a.input).to_csv(a.output, index=False)
    print("wrote", a.output)
