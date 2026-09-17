"""End-to-end smoke tests: run each subsystem's predict.py CLI on a real test file and check the schema.

Skipped automatically if the data or the shipped model artefact is missing.
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CASES = {
    "door": dict(script=ROOT / "Door/code/predict.py", model=ROOT / "Door/model/door_model.joblib",
                 inp=ROOT / "data/Door/Test.csv"),
    "acv": dict(script=ROOT / "ACV/code/predict.py", model=ROOT / "ACV/model/acv_model.joblib",
                inp=ROOT / "data/ACV/Test/acv_test_case.xlsx"),
    "rail": dict(script=ROOT / "Rail Corrugation/code/predict.py", model=ROOT / "Rail Corrugation/model/rail_model.joblib",
                 inp=ROOT / "data/Rail_Corrugation/Test/Test1.csv"),
    "shm": dict(script=ROOT / "SHM/code/predict.py", model=ROOT / "SHM/model/shm_model.json",
                inp=ROOT / "data/SHM/Test/test01.csv"),
}


def run_cli(name: str, tmp_path: Path) -> pd.DataFrame:
    c = CASES[name]
    if not c["inp"].exists():
        pytest.skip(f"{name}: data not present")
    if not c["model"].exists():
        pytest.skip(f"{name}: model artefact not present")
    out = tmp_path / f"{name}_predictions.csv"
    r = subprocess.run([PY, str(c["script"]), "--input", str(c["inp"]), "--output", str(out)],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, f"{name} CLI failed:\n{r.stdout}\n{r.stderr}"
    assert out.exists()
    return pd.read_csv(out, dtype=str)


def test_door_cli(tmp_path):
    df = run_cli("door", tmp_path)
    assert list(df.columns) == ["start_time", "end_time", "prediction"]
    assert len(df) > 0
    assert set(df.prediction) <= {"Normal", "Abnormal resistance"}
    ts = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,3}$")
    assert df.start_time.str.match(ts).all() and df.end_time.str.match(ts).all()


def test_acv_cli(tmp_path):
    df = run_cli("acv", tmp_path)
    assert list(df.columns) == ["file_id", "ranked_cars"]
    assert len(df) == 1 and df.file_id.iloc[0] == "acv_test_case.xlsx"
    cars = df.ranked_cars.iloc[0].split("|")
    assert len(cars) == len(set(cars)) == 8
    assert all(re.fullmatch(r"\d{2}", c) for c in cars)


def test_rail_cli(tmp_path):
    df = run_cli("rail", tmp_path)
    assert list(df.columns) == ["file_id", "prediction"]
    assert df.file_id.iloc[0] == "Test1.csv"
    assert set(df.prediction) <= {"Normal", "Side I", "Side II"}


def test_shm_cli(tmp_path):
    df = run_cli("shm", tmp_path)
    assert list(df.columns) == ["file_id", "prediction"]
    assert df.file_id.iloc[0] == "test01.csv"
    v = float(df.prediction.iloc[0])
    assert 0 < v < 10
