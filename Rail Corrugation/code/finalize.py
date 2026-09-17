"""Freeze, fit, verify, and promote the selected Rail Corrugation model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Rail Corrugation/code"))
from experiment_io import atomic_joblib, atomic_json, sha256_file, stable_hash  # noqa: E402
from pipeline import CLASSES, NormalReference, aggregate, file_channel_table, side_relative_rows  # noqa: E402
from validate import make_detector, side_table  # noqa: E402


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, name); os.replace(name, destination)
    except BaseException:
        Path(name).unlink(missing_ok=True); raise


def run_prediction(model: Path, input_dir: Path, output: Path) -> None:
    env = os.environ.copy(); env["RAIL_MODEL_PATH"] = str(model.resolve())
    subprocess.run([sys.executable, str(ROOT / "Rail Corrugation/code/predict.py"),
                    "--input", str(input_dir), "--output", str(output)],
                   cwd=ROOT, env=env, check=True, timeout=1800)


def validate_submission(path: Path, test_dir: Path) -> dict:
    frame = pd.read_csv(path, dtype=str)
    expected = sorted((p.name for p in test_dir.glob("*.csv")), key=lambda s: int(''.join(filter(str.isdigit, s))))
    if list(frame.columns) != ["file_id", "prediction"]:
        raise ValueError(f"wrong output columns: {frame.columns.tolist()}")
    if frame.file_id.tolist() != expected or frame.file_id.duplicated().any():
        raise ValueError("prediction IDs do not exactly match naturally ordered test inputs")
    if not set(frame.prediction) <= set(CLASSES):
        raise ValueError("prediction contains an invalid label")
    return {"rows": len(frame), "counts": frame.prediction.value_counts().to_dict(), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch == "main" or not branch:
        raise RuntimeError("final fitting is forbidden on main or detached HEAD")
    exp_dir = args.experiment_root / args.experiment_id
    config = json.loads((exp_dir / "config.json").read_text())
    provenance = json.loads((exp_dir / "provenance.json").read_text())
    metrics = json.loads((exp_dir / "metrics.json").read_text())
    labels_frame = pd.read_csv(args.data_dir / "Train_Labels.csv")
    labels = labels_frame.label.to_numpy(); files = [args.data_dir / "Train" / f for f in labels_frame.filename]
    cache = args.experiment_root / "cache" / f"channel_tables_{provenance['feature_cache_fingerprint'][:16]}.joblib"
    payload = joblib.load(cache)
    if payload.get("fingerprint") != provenance["feature_cache_fingerprint"]:
        raise RuntimeError("feature-cache fingerprint mismatch")
    values = payload["value"]; chan_tables = [v[0] for v in values]; speeds = np.asarray([v[1] for v in values])
    all_idx = np.arange(len(labels))
    ref = NormalReference(config["reference"], config.get("smooth_speed", False)).fit(chan_tables, speeds, labels)
    frame = pd.DataFrame([aggregate(ref.excess(ct, speeds[i]), speeds[i]) for i, ct in enumerate(chan_tables)])
    rows, targets = side_table(frame, all_idx, labels, config)
    detector = make_detector(0, args.workers).fit(rows, targets)
    thresholds = [item["threshold"] for item in metrics["stratified_5x3"]["thresholds"]]
    threshold = float(np.median(thresholds))
    artifact = {
        "artifact_version": 2, "choice": "side_detector", "ref": ref, "det": detector,
        "det_cols": rows.columns.tolist(), "tau": threshold, "classes": CLASSES,
        "feature_config": config, "training": {
            "experiment_id": args.experiment_id, "branch": branch,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_manifest_hash": provenance["data_manifest_hash"], "code_hash": provenance["code_hash"],
            "feature_cache_fingerprint": provenance["feature_cache_fingerprint"], "threshold_rule": "median outer-fold inner-OOF thresholds",
            "validation": metrics, "libraries": {"python": sys.version, "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__},
        },
    }
    final_dir = args.experiment_root / "final"; final_dir.mkdir(parents=True, exist_ok=True)
    candidate_model = final_dir / "rail_model.joblib"; atomic_joblib(candidate_model, artifact)
    # Fresh-process determinism and exact submission-contract checks.
    first = final_dir / "rail_predictions.first.csv"; second = final_dir / "rail_predictions.second.csv"
    run_prediction(candidate_model, args.data_dir / "Test", first)
    run_prediction(candidate_model, args.data_dir / "Test", second)
    first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
    if first_bytes != second_bytes:
        raise RuntimeError("fresh-process predictions are not byte-identical")
    submission = validate_submission(first, args.data_dir / "Test")
    artifact_hash = sha256_file(candidate_model)
    frozen = {"selected_experiment": args.experiment_id, "selection_metrics": metrics,
              "threshold": threshold, "artifact_sha256": artifact_hash, "prediction": submission,
              "command": " ".join(sys.argv), "config_hash": stable_hash(config)}
    atomic_json(final_dir / "frozen_model.json", frozen)
    atomic_copy(candidate_model, ROOT / "Rail Corrugation/model/rail_model.joblib")
    atomic_copy(first, ROOT / "predictions/rail_predictions.csv")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
