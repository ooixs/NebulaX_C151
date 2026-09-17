from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Rail Corrugation/code"))


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_module("rail_pipeline_test", "Rail Corrugation/code/pipeline.py")
predict_module = load_module("rail_predict_test", "Rail Corrugation/code/predict.py")


def valid_frame(rows: int = 4) -> pd.DataFrame:
    columns = ["Rotating speed"] + [f"{kind} of bearing in position {position} of car {car}"
              for car in range(1, 9) for position in range(1, 9) for kind in ("Vibration", "Shock")]
    return pd.DataFrame(np.zeros((rows, 129)), columns=columns)


@pytest.mark.parametrize("mutation,match", [
    (lambda frame: frame.iloc[:, :-1], "expected 129 columns"),
    (lambda frame: frame.rename(columns={frame.columns[1]: "bad"}), "unexpected Rail header"),
])
def test_load_file_rejects_malformed_schema(tmp_path, mutation, match):
    path = tmp_path / "bad.csv"
    mutation(valid_frame()).to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        pipeline.load_file(path)


def test_load_file_rejects_nonnumeric(tmp_path):
    frame = valid_frame().astype(object); frame.iloc[0, 2] = "oops"
    path = tmp_path / "bad.csv"; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonnumeric"):
        pipeline.load_file(path)


def test_natural_order_and_empty_directory(tmp_path):
    for name in ("Test10.csv", "Test2.csv", "Test 1.csv"):
        (tmp_path / name).touch()
    assert [p.name for p in sorted(tmp_path.glob("*.csv"), key=predict_module._natural_key)] == [
        "Test 1.csv", "Test2.csv", "Test10.csv"]
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(ValueError, match="no CSV inputs"):
        predict_module.predict(empty)


def test_side_reference_never_uses_non_normal_rows():
    base = pd.DataFrame({c: np.ones(64) for c in pipeline.REF_FEATURES})
    base["car"] = np.arange(64) // 8; base["position"] = np.arange(64) % 8 + 1
    base["side"] = [pipeline.channel_side(k) for k in range(64)]
    fault = base.copy(); fault[pipeline.REF_FEATURES] = 1_000
    ref = pipeline.NormalReference().fit([base, fault], [10.0, 10.0], ["Normal", "Side I"])
    transformed = ref.excess(base, 10.0)
    assert np.allclose(transformed["ex_log_rms"], 0)
