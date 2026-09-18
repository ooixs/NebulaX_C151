"""SHM inference: stress CSV file or directory -> shm_predictions.csv rows.

CLI: python SHM/code/predict.py --input data/SHM/Test --output shm_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from SHM.code.pipeline import DamageModel, ResidualDamageModel, cycles, load_series  # noqa: E402

from common.artifacts import load_active_model

MODEL_PATH = ROOT / "SHM/model/shm_model.json"
RESIDUAL_MODEL_PATH = ROOT / "SHM/model/shm_model.joblib"


def _read_model(path):
    if path.suffix == ".joblib":
        art = joblib.load(path)
        if art.get("kind") != "physics_plus_residual":
            raise ValueError("unsupported SHM joblib model format")
        return ResidualDamageModel(art["base"], art["feature_names"], art["scaler"], art["model"])
    return DamageModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _model():
    return load_active_model(MODEL_PATH.parent, (MODEL_PATH.name, RESIDUAL_MODEL_PATH.name), _read_model)


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def predict(input_path: str | Path) -> pd.DataFrame:
    p = Path(input_path)
    files = sorted(p.glob("*.csv"), key=_natural_key) if p.is_dir() else [p]
    m = _model()
    return pd.DataFrame([{"file_id": f.name, "prediction": m.predict(load_series(f))} for f in files])


def predict_with_evidence(input_path: str | Path) -> tuple[pd.DataFrame, dict]:
    p = Path(input_path)
    files = sorted(p.glob("*.csv"), key=_natural_key) if p.is_dir() else [p]
    model = _model()
    rows, evidence = [], {}
    for file in files:
        series = load_series(file)
        extracted_cycles = cycles(series)
        prediction = model.predict_from_cycles(extracted_cycles) if isinstance(model, DamageModel) else model.predict(series)
        rows.append({"file_id": file.name, "prediction": prediction})
        evidence[file.name] = {
            "estimated_damage": round(float(prediction), 12),
            "samples": int(len(series)),
            "counted_cycles": round(float(extracted_cycles[:, 2].sum()), 1),
            "stress_range": round(float(series.max() - series.min()), 6),
        }
    return pd.DataFrame(rows), {"files": evidence}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    predict(a.input).to_csv(a.output, index=False)
    print("wrote", a.output)
