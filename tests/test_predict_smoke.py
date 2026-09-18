"""End-to-end smoke tests: run each subsystem's predict.py CLI on a real test file and check the schema.

Skipped automatically if the data or the shipped model artefact is missing.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def model_path(name: str) -> Path:
    from common.artifacts import resolve_model

    path = CASES[name]["model"]
    candidates = (path.name, path.with_suffix(".joblib").name) if name == "shm" else (path.name,)
    return resolve_model(path.parent, candidates)


def run_cli(name: str, tmp_path: Path) -> pd.DataFrame:
    c = CASES[name]
    if not c["inp"].exists():
        pytest.skip(f"{name}: data not present")
    try:
        model_path(name)
    except FileNotFoundError:
        if (c["model"].parent / "active_model.json").exists():
            raise
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


def latest_research_run(name):
    subsystem = CASES[name]["model"].parents[1].name
    reports = []
    for path in (ROOT / "weights" / subsystem / "runs").glob("*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("published"):
            reports.append((path, summary))
    if not reports:
        pytest.skip(f"{name}: no published research run")
    path, summary = max(reports, key=lambda pair: pair[0].stat().st_mtime_ns)
    return path.parent, summary


@pytest.mark.parametrize("name", CASES)
def test_research_model_and_stopping_rule(name):
    from common.research import Plateau, fingerprint

    folder, summary = latest_research_run(name)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    trials = json.loads((folder / "trials.json").read_text(encoding="utf-8"))
    search = Plateau(manifest["min_delta"], manifest["patience"])
    for trial in trials:
        result = search.update(trial["scores"], trial.get("eligible", True))
        assert result["accepted"] == trial["accepted"]
        assert result["mean"] == pytest.approx(trial["mean"])
        assert result["stale_trials"] == trial["stale_trials"]
    assert search.stop_reason == summary["stop_reason"]
    assert search.best_mean == pytest.approx(summary["best_score"])
    assert summary["trials"] == len(trials)
    assert not manifest["test_data_used_for_selection"]
    assert fingerprint(folder / summary["model"]) == fingerprint(model_path(name))
    assert (folder / "before").is_dir()


@pytest.mark.parametrize("name", CASES)
def test_archived_full_predictions_cover_input(name):
    from common.metrics import parse_door_time
    from Door.code.pipeline import load_stream, segment

    folder, _ = latest_research_run(name)
    path = folder / f"{name}_predictions.csv"
    if not path.exists() or not CASES[name]["inp"].exists():
        pytest.skip(f"{name}: full prediction export or input not present")
    frame = pd.read_csv(path, dtype=str)
    published = ROOT / "predictions" / path.name
    if published.exists():
        pd.testing.assert_frame_equal(frame, pd.read_csv(published, dtype=str))
    assert frame.notna().all().all()
    if name == "door":
        assert list(frame.columns) == ["start_time", "end_time", "prediction"]
        assert len(frame) == len(segment(load_stream(CASES[name]["inp"])))
        assert set(frame.prediction) <= {"Normal", "Abnormal resistance"}
        starts, ends = frame.start_time.map(parse_door_time), frame.end_time.map(parse_door_time)
        assert (starts <= ends).all()
        assert (starts.iloc[1:].to_numpy() > ends.iloc[:-1].to_numpy()).all()
    elif name == "acv":
        assert list(frame.columns) == ["file_id", "ranked_cars"]
        assert frame.file_id.tolist() == [CASES[name]["inp"].name]
        assert set(frame.ranked_cars.iloc[0].split("|")) == {f"{i:02d}" for i in range(1, 9)}
        assert len(frame.ranked_cars.iloc[0].split("|")) == 8
    else:
        assert list(frame.columns) == ["file_id", "prediction"]
        expected = {p.name for p in CASES[name]["inp"].parent.glob("*.csv")}
        assert set(frame.file_id) == expected and len(frame) == len(expected)
        if name == "rail":
            assert set(frame.prediction) <= {"Normal", "Side I", "Side II"}
        else:
            prediction = frame.prediction.astype(float)
            assert np.isfinite(prediction).all() and (prediction >= 0).all()


def test_rail_research_scores_and_nested_splits_replay():
    from common.metrics import rail_macro_f1

    folder, _ = latest_research_run("rail")
    reports = list(folder.glob("trial_*.json")) + list(folder.glob("confirmation_*.json")) + list(folder.glob("blocked_*.json"))
    checked = 0
    for path in reports:
        arrays = path.with_name(f"{path.stem}_oof.npz")
        if not arrays.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        with np.load(arrays, allow_pickle=False) as archive:
            y, p, prediction = archive["labels"], archive["probabilities"], archive["prediction"]
            scores = [rail_macro_f1(y, labels)["macro_f1"] for labels in prediction]
            assert scores == pytest.approx(report["scores"])
            for fold in report["folds"]:
                te = np.array(fold["validation_indices"])
                assert set(te).isdisjoint(fold["threshold_training_indices"])
                a, b = p[fold["repeat"], te].T
                decoded = np.where(np.maximum(a, b) < fold["threshold"], "Normal", np.where(a >= b, "Side I", "Side II"))
                assert np.array_equal(decoded, prediction[fold["repeat"], te])
        checked += 1
    assert checked > 0
    for protocol in ("folds.json", "confirmation_folds.json", "blocked_folds.json"):
        for fold in json.loads((folder / protocol).read_text(encoding="utf-8")):
            tr, te = set(fold["train"]), set(fold["validation"])
            for inner_tr, inner_te in fold["inner"]:
                assert set(inner_tr) | set(inner_te) == tr
                assert set(inner_tr).isdisjoint(inner_te)
                assert te.isdisjoint(inner_tr) and te.isdisjoint(inner_te)


def test_shm_research_calibration_replays_inside_training_folds():
    from common.metrics import shm_score
    from SHM.code.train import select_config

    folder, _ = latest_research_run("shm")
    inputs = json.loads((folder / "inputs.json").read_text(encoding="utf-8"))
    files = {name: i for i, name in enumerate(inputs["files"])}
    trials = json.loads((folder / "trials.json").read_text(encoding="utf-8"))
    replayed = 0
    for i, trial in enumerate(trials):
        with np.load(folder / f"trial_{i:02d}.npz", allow_pickle=False) as archive:
            D, oof = archive["target"], archive["oof"]
            assert [shm_score(D, p)["score"] for p in oof] == pytest.approx(trial["scores"])
            configs_path = folder / f"configs_{i:02d}.json"
            if not configs_path.exists():
                assert trial.get("kind") == "custom"
                continue
            configs = json.loads(configs_path.read_text(encoding="utf-8"))
            S = archive["S"]
            for fold in trial["folds"]:
                te = np.array([files[name] for name in fold["validation_files"]])
                tr = np.setdiff1d(np.arange(len(D)), te)
                j, C = select_config(S, D, tr, configs, trial["fit_tolerance"])
                assert configs[j] == fold["config"]
                assert C == pytest.approx(fold["C"])
                assert np.allclose(S[te, j] / C, oof[fold["seed"], te])
            replayed += 1
    assert replayed > 0


def test_door_research_end_to_end_score_replays():
    from common.metrics import door_iou_f1
    from Door.code.pipeline import load_stream, segment, segments_to_frame

    folder, summary = latest_research_run("door")
    data = ROOT / "data" / "Door"
    if not (data / "Train.csv").exists():
        pytest.skip("Door training data not present")
    report = json.loads((folder / f"cv_{summary['best_name']}.json").read_text(encoding="utf-8"))
    probabilities = np.array(report["oof"])
    predictions = np.array(report["oof_pred"])
    for fold in report["folds"]:
        te = np.array(fold["validation_indices"])
        assert set(te).isdisjoint(fold["threshold_training_indices"])
        assert np.array_equal(probabilities[te] >= fold["threshold"], predictions[te])
    segs = segment(load_stream(data / "Train.csv"))
    frame = segments_to_frame(segs, np.where(predictions, "Abnormal resistance", "Normal"))
    answer = pd.read_csv(data / "Train_Segments_Answer.csv")
    assert door_iou_f1(answer, frame)["score"] == pytest.approx(report["oof_iou_f1"])


def test_acv_research_rank_scores_replay():
    from common.metrics import acv_rank_score

    folder, _ = latest_research_run("acv")
    path = ROOT / "data" / "ACV" / "Train_Labels.csv"
    if not path.exists():
        pytest.skip("ACV training labels not present")
    labels = pd.read_csv(path, dtype=str).set_index("filename")["faulty_car"]
    for fold in json.loads((folder / "folds.json").read_text(encoding="utf-8")):
        assert fold["validation_case"] not in fold["training_cases"]
    for trial in json.loads((folder / "trials.json").read_text(encoding="utf-8")):
        actual = {name: acv_rank_score(ranking, labels[name]) for name, ranking in trial["rankings"].items()}
        assert actual == pytest.approx(trial["case_scores"])
