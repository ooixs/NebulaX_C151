from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def nested_folds(y, folds=5, repeats=3, inner_folds=3, seed=0):
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

    y = np.asarray(y)
    outer = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    result = []
    for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y)):
        count = int(np.unique(y[tr], return_counts=True)[1].min())
        if count < 2:
            raise ValueError("each outer training class needs at least two files for inner calibration")
        inner = StratifiedKFold(n_splits=min(inner_folds, count), shuffle=True, random_state=seed + 100 + k)
        pairs = [(tr[a], tr[b]) for a, b in inner.split(np.zeros(len(tr)), y[tr])]
        result.append(dict(fold=k, repeat=k // folds, train=tr, validation=te, inner=pairs))
    return result


class Plateau:
    def __init__(self, min_delta=0.002, patience=5):
        if min_delta < 0 or patience < 1:
            raise ValueError("min_delta must be nonnegative and patience must be positive")
        self.min_delta, self.patience = min_delta, patience
        self.best_mean, self.best_std = None, 0.0
        self.stale = 0

    def update(self, scores, eligible=True):
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("scores must be a nonempty vector of finite values in [0, 1]")
        mean, std = float(scores.mean()), float(scores.std())
        required = max(self.min_delta, self.best_std)
        gain = None if self.best_mean is None else mean - self.best_mean
        accepted = eligible and (gain is None or gain > required + 1e-12)
        if accepted:
            self.best_mean, self.best_std, self.stale = mean, std, 0
        else:
            self.stale += 1
        return dict(mean=mean, std=std, scores=scores.tolist(), gain=gain, required_gain=required,
                    accepted=bool(accepted), eligible=bool(eligible), stale_trials=self.stale)

    @property
    def stop_reason(self):
        if self.best_mean is not None and self.best_mean >= 1.0 - 1e-12:
            return "score_ceiling"
        if self.stale >= self.patience:
            return "negligible_improvement"
        return None


class ResearchRun:
    def __init__(self, subsystem, run_id=None, min_delta=0.002, patience=5, root=ROOT):
        self.root, self.subsystem = Path(root), subsystem
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if Path(self.run_id).name != self.run_id or self.run_id in (".", ".."):
            raise ValueError("run_id must be a directory name, not a path")
        self.path = self.root / "weights" / subsystem / "runs" / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.tracker = Plateau(min_delta, patience)
        self.trials, self.best_name = [], None
        self.started = time.monotonic()
        weights = self.root / "weights" / subsystem
        paths = list((self.root / subsystem / "model").glob("*"))
        paths += list(weights.glob("cv*.json")) + list(weights.glob("loo_report.json"))
        for path in paths:
            if path.is_file():
                self.snapshot(path)
        versions = {name: importlib.metadata.version(name) for name in
                    ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}
        sources = list((self.root / subsystem / "code").glob("*.py"))
        sources += list((self.root / "common").glob("*.py"))
        for source in sources:
            self.snapshot(source)
        self.save_json("manifest.json", dict(subsystem=subsystem, run_id=self.run_id, python=sys.version,
                       versions=versions, min_delta=min_delta, patience=patience,
                       acceptance="gain > max(min_delta, incumbent repeat-score standard deviation)",
                       test_data_used_for_selection=False,
                       sources={str(p.relative_to(self.root)): fingerprint(p) for p in sources}))

    def snapshot(self, path):
        path = Path(path)
        backup = self.path / "before" / path.relative_to(self.root)
        if path.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        return backup

    def save_json(self, name, value):
        (self.path / name).write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False), encoding="utf-8")

    def record(self, name, scores, eligible=True, **details):
        result = dict(name=name, **self.tracker.update(scores, eligible), **details,
                      elapsed_seconds=time.monotonic() - self.started)
        if result["accepted"]:
            self.best_name = name
        self.trials.append(result)
        self.save_json("trials.json", self.trials)
        print(f"{name}: {result['mean']:.6f} +/- {result['std']:.6f} "
              f"{'ACCEPT' if result['accepted'] else 'retain incumbent'} "
              f"(stale {self.tracker.stale}/{self.tracker.patience})", flush=True)
        return result

    def publish(self, source, destination):
        source, destination = Path(source), Path(destination)
        if not source.is_file():
            raise FileNotFoundError(source)
        self.snapshot(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def finish(self, **details):
        summary = dict(subsystem=self.subsystem, run_id=self.run_id, best_name=self.best_name,
                       best_score=self.tracker.best_mean, best_std=self.tracker.best_std,
                       stop_reason=self.tracker.stop_reason or "candidate_budget_exhausted",
                       trials=len(self.trials), elapsed_seconds=time.monotonic() - self.started, **details)
        self.save_json("summary.json", summary)
        print(json.dumps(summary, default=json_value, allow_nan=False), flush=True)
        return summary
