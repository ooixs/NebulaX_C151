import json
import runpy
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_rail_import_preserves_standard_library_code(monkeypatch):
    import code

    monkeypatch.setattr(sys, "path", sys.path.copy())
    runpy.run_path(str(ROOT / "Rail Corrugation/code/predict.py"), run_name="rail_import_probe")
    assert sys.modules["code"] is code
    assert hasattr(code, "InteractiveConsole")
    assert str(ROOT / "Rail Corrugation") not in sys.path
    from rail_corrugation import train
    assert train.NormalReference.__module__ == "rail_corrugation.pipeline"


@pytest.mark.parametrize("compression", [0, 3, ("gzip", 3)])
def test_legacy_rail_pickle_without_namespace_mutation(tmp_path, monkeypatch, compression):
    import code
    from common.artifacts import load_rail_model
    from rail_corrugation import pipeline

    path = tmp_path / "legacy.joblib"
    legacy = types.ModuleType("code")
    legacy.__path__ = []
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "code", legacy)
        patch.setitem(sys.modules, "code.pipeline", pipeline)
        patch.setattr(pipeline.NormalReference, "__module__", "code.pipeline")
        joblib.dump(dict(ref=pipeline.NormalReference(), values=np.arange(24).reshape(6, 4)), path, compress=compression)
    before = path.read_bytes()
    result = load_rail_model(path)
    assert isinstance(result["ref"], pipeline.NormalReference)
    assert np.array_equal(result["values"], np.arange(24).reshape(6, 4))
    assert sys.modules["code"] is code and "code.pipeline" not in sys.modules
    assert path.read_bytes() == before
    joblib.dump(result, tmp_path / "canonical.joblib", compress=compression)
    assert isinstance(joblib.load(tmp_path / "canonical.joblib")["ref"], pipeline.NormalReference)


def test_active_model_is_explicit_and_integrity_checked(tmp_path):
    from common.artifacts import activate_model, resolve_model

    physics, residual = tmp_path / "shm_model.json", tmp_path / "shm_model.joblib"
    candidates = (physics.name, residual.name)
    physics.write_text('{"m": 5, "C": 2}')
    assert resolve_model(tmp_path, candidates) == physics
    residual.write_bytes(b"inactive artifact")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_model(tmp_path, candidates)
    activate_model(physics, run_id="physics-v1")
    assert resolve_model(tmp_path, candidates) == physics
    physics.write_text('{"m": 5, "C": 4}')
    with pytest.raises(ValueError, match="checksum"):
        resolve_model(tmp_path, candidates)
    activate_model(physics, run_id="physics-v2")
    assert resolve_model(tmp_path, candidates) == physics
    manifest = tmp_path / "active_model.json"
    record = json.loads(manifest.read_text())
    assert record["run_id"] == "physics-v2"
    record["file"] = "../outside.json"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        resolve_model(tmp_path, candidates)
    record["file"] = residual.name
    record["sha256"] = "0" * 64
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        resolve_model(tmp_path, candidates)


def test_shm_switches_models_without_deleting_alternatives(tmp_path, monkeypatch):
    from common.artifacts import activate_model
    from SHM.code import predict

    physics = tmp_path / "shm_model.json"
    residual = tmp_path / "shm_model.joblib"
    physics.write_text('{"m": 5, "C": 2}')
    residual.write_bytes(b"inactive and deliberately not deserializable")
    monkeypatch.setattr(predict, "MODEL_PATH", physics)
    monkeypatch.setattr(predict, "RESIDUAL_MODEL_PATH", residual)
    activate_model(physics)
    first = predict._model()
    assert first.C == 2
    import os
    previous_stat = physics.stat()
    physics.write_text('{"m": 5, "C": 4}')
    os.utime(physics, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    activate_model(physics)
    assert predict._model().C == 4 and predict._model() is not first
    assert residual.read_bytes() == b"inactive and deliberately not deserializable"


def test_model_switch_during_loading_is_rejected(tmp_path):
    from common.artifacts import activate_model, load_active_model

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text("1")
    second.write_text("2")
    activate_model(first)
    def reader(path):
        activate_model(second)
        return path.read_text()
    with pytest.raises(RuntimeError, match="changed during loading"):
        load_active_model(tmp_path, (first.name, second.name), reader)


def test_missing_active_model_does_not_fall_back(tmp_path):
    from common.artifacts import MANIFEST, resolve_model

    (tmp_path / "shm_model.json").write_text('{"m": 5, "C": 2}')
    manifest = tmp_path / MANIFEST
    manifest.write_text(json.dumps(dict(schema_version=1, file="shm_model.joblib", sha256="0" * 64)))
    with pytest.raises(FileNotFoundError, match="active model missing"):
        resolve_model(tmp_path, ("shm_model.json", "shm_model.joblib"))
    manifest.write_text("[]")
    with pytest.raises(ValueError, match="manifest"):
        resolve_model(tmp_path, ("shm_model.json",))


def test_research_publish_activates_new_artifact(tmp_path):
    from common.artifacts import activate_model, resolve_model
    from common.research import ResearchRun

    models = tmp_path / "SHM/model"
    models.mkdir(parents=True)
    previous = models / "shm_model.joblib"
    previous.write_bytes(b"old model")
    activate_model(previous)
    run = ResearchRun("SHM", "new-run", root=tmp_path)
    candidate = run.path / "shm_model.json"
    candidate.write_text('{"m": 5, "C": 2}')
    run.publish(candidate, models / candidate.name)
    assert resolve_model(models, (candidate.name, previous.name)).name == candidate.name
    assert previous.read_bytes() == b"old model"
    assert (run.path / "before/SHM/model/active_model.json").exists()


def test_packaging_refuses_to_replace_existing_directory(tmp_path):
    from package import assemble

    dest = tmp_path / "existing"
    dest.mkdir()
    sentinel = dest / "keep.txt"
    sentinel.write_text("do not remove")
    with pytest.raises(FileExistsError):
        assemble(dest)
    assert sentinel.read_text() == "do not remove"


@pytest.fixture(scope="module")
def shared_inputs(tmp_path_factory):
    paths = {
        "Door": [ROOT / "data/Door/Test.csv"],
        "ACV": [ROOT / "data/ACV/Test/acv_test_case.xlsx"],
        "Rail Corrugation": [ROOT / "data/Rail_Corrugation/Test/Test1.csv", ROOT / "data/Rail_Corrugation/Test/Test13.csv"],
        "SHM": [ROOT / "data/SHM/Test/test01.csv", ROOT / "data/SHM/Test/test02.csv"],
    }
    if any(not p.exists() for files in paths.values() for p in files):
        pytest.skip("integration data not present")
    from common.artifacts import MANIFEST, resolve_model
    from package import MODEL_FILES
    for subsystem, candidates in MODEL_FILES.items():
        directory = ROOT / subsystem / "model"
        try:
            resolve_model(directory, candidates)
        except FileNotFoundError:
            if (directory / MANIFEST).exists():
                raise
            pytest.skip(f"{subsystem}: integration model not present")
    folder = tmp_path_factory.mktemp("integration-inputs")
    inputs = {}
    for subsystem, files in paths.items():
        target = folder / subsystem
        target.mkdir()
        for source in files:
            shutil.copy2(source, target / source.name)
        inputs[subsystem] = str(target / files[0].name if subsystem == "Door" else target)
    return inputs


SHARED_PROCESS = '''
import code, importlib, json, sys
from pathlib import Path
root, inputs, output = Path(sys.argv[1]), json.loads(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(root))
from common.inference import predict
import pandas as pd
for subsystem, path in inputs.items():
    first = predict(subsystem, path)
    if sys.argv[4] == "1":
        pd.testing.assert_frame_equal(first, predict(subsystem, path))
    first.to_csv(output / (subsystem + ".csv"), index=False)
for name in ("common.inference", "common.artifacts", "Door.code.predict", "ACV.code.predict", "SHM.code.predict", "rail_corrugation.predict", "rail_corrugation.pipeline"):
    assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(root.resolve()), name
assert sys.modules["code"] is code and hasattr(code, "InteractiveConsole")
assert "code.pipeline" not in sys.modules
'''


def run_shared(root, inputs, output, repeat=True):
    output.mkdir()
    result = subprocess.run([sys.executable, "-I", "-c", SHARED_PROCESS, str(root), json.dumps(inputs), str(output), str(int(repeat))],
                            cwd=output.parent, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return {name: pd.read_csv(output / f"{name}.csv", dtype=str) for name in inputs}


def compare_golden(frames):
    names = {"Door": "door", "ACV": "acv", "Rail Corrugation": "rail", "SHM": "shm"}
    for subsystem, frame in frames.items():
        expected = pd.read_csv(ROOT / "predictions" / f"{names[subsystem]}_predictions.csv", dtype=str)
        if subsystem != "Door":
            expected = expected[expected.file_id.isin(frame.file_id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame, expected)


def test_all_predictors_share_one_process(shared_inputs, tmp_path):
    compare_golden(run_shared(ROOT, shared_inputs, tmp_path / "shared"))


def test_full_batch_matches_submission_exports(shared_inputs, tmp_path):
    inputs = {"Door": str(ROOT / "data/Door/Test.csv"), "ACV": str(ROOT / "data/ACV/Test"),
              "Rail Corrugation": str(ROOT / "data/Rail_Corrugation/Test"), "SHM": str(ROOT / "data/SHM/Test")}
    frames = run_shared(ROOT, inputs, tmp_path / "full-batch", repeat=False)
    compare_golden(frames)
    assert {name: len(frame) for name, frame in frames.items()} == {"Door": 38, "ACV": 1, "Rail Corrugation": 68, "SHM": 16}


def test_packaged_predictors_and_zip_match_exports(shared_inputs, tmp_path):
    from package import assemble

    dest = tmp_path / "C151"
    assemble(dest)
    assert not (dest / "app/data").exists()
    assert not (dest / "app/weights").exists()
    with zipfile.ZipFile(dest / "predictions.zip") as archive:
        assert set(archive.namelist()) == {f"{name}_predictions.csv" for name in ("door", "acv", "rail", "shm")}
        for name in archive.namelist():
            assert archive.read(name) == (ROOT / "predictions" / name).read_bytes()
    frames = run_shared(dest / "app", shared_inputs, tmp_path / "packaged")
    compare_golden(frames)
    optional = run_shared(dest / "Optional_Items", shared_inputs, tmp_path / "optional")
    compare_golden(optional)
    for subsystem, input_path in shared_inputs.items():
        output = tmp_path / f"cli-{subsystem}.csv"
        command = [sys.executable, "-I", str(dest / "app" / subsystem / "code/predict.py"),
                   "--input", input_path, "--output", str(output)]
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        pd.testing.assert_frame_equal(pd.read_csv(output, dtype=str), frames[subsystem])
