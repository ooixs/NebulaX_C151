"""Rail Corrugation inference: CSV file or directory -> rail_predictions.csv rows.

CLI: python "Rail Corrugation/code/predict.py" --input data/Rail_Corrugation/Test --output rail_predictions.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "Rail Corrugation/code")):
    if p not in sys.path:
        sys.path.insert(0, p)
from pipeline import CLASSES, aggregate, file_channel_table, side_relative_rows  # noqa: E402

MODEL_PATH = Path(os.environ.get("RAIL_MODEL_PATH", ROOT / "Rail Corrugation/model/rail_model.joblib"))
_ART = None


def _artefact():
    global _ART
    if _ART is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Rail model artifact not found: {MODEL_PATH}")
        _ART = joblib.load(MODEL_PATH)
    return _ART


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else re.sub(r"\s+", "", t).casefold()
            for t in re.split(r"(\d+)", p.name)]


def predict_one(path: Path, with_proba: bool = False):
    art = _artefact()
    ct, v = file_channel_table(path)
    feats = aggregate(art["ref"].excess(ct, v), v)
    if art["choice"] == "side_detector":
        r1, r2 = side_relative_rows(feats)
        R = pd.DataFrame([r1, r2]).reindex(columns=art["det_cols"]).fillna(-9)
        p1, p2 = art["det"].predict_proba(R)[:, 1]
        label = "Normal" if max(p1, p2) < art["tau"] else ("Side I" if p1 >= p2 else "Side II")
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
    if not files:
        raise ValueError(f"no CSV inputs found: {p}")
    return pd.DataFrame([predict_one(f, with_proba) for f in files])


def atomic_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    try:
        frame.to_csv(name, index=False)
        os.replace(name, output)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--with-evidence", action="store_true")
    a = ap.parse_args()
    atomic_csv(predict(a.input, with_proba=a.with_evidence), a.output)
    print("wrote", a.output)
