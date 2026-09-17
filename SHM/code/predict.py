"""SHM inference: stress CSV file or directory -> shm_predictions.csv rows.

CLI: python SHM/code/predict.py --input data/SHM/Test --output shm_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from SHM.code.pipeline import DamageModel, load_series  # noqa: E402

MODEL_PATH = ROOT / "SHM/model/shm_model.json"
_M = None


def _model() -> DamageModel:
    global _M
    if _M is None:
        _M = DamageModel.from_dict(json.loads(MODEL_PATH.read_text()))
    return _M


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def predict(input_path: str | Path) -> pd.DataFrame:
    p = Path(input_path)
    files = sorted(p.glob("*.csv"), key=_natural_key) if p.is_dir() else [p]
    m = _model()
    return pd.DataFrame([{"file_id": f.name, "prediction": m.predict(load_series(f))} for f in files])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    predict(a.input).to_csv(a.output, index=False)
    print("wrote", a.output)
