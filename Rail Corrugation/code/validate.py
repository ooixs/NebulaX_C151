"""RC-E01+ validation and controlled Rail Corrugation experiments.

All reported thresholds are selected from inner-training OOF predictions and then
frozen for the corresponding outer fold.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Rail Corrugation/code"))
from experiment_io import (atomic_joblib, atomic_json, load_fingerprinted_cache,  # noqa: E402
                                save_fingerprinted_cache, sha256_file, stable_hash)
from pipeline import (CLASSES, NormalReference, aggregate, file_channel_table,  # noqa: E402
                           side_relative_rows)


def natural_number(name: str) -> int:
    return int(re.search(r"(\d+)", name).group(1))


def decode(probs: np.ndarray, threshold: float) -> np.ndarray:
    best = probs.argmax(axis=1)
    return np.where(probs.max(axis=1) < threshold, "Normal", np.where(best == 0, "Side I", "Side II"))


def full_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=CLASSES, zero_division=0)
    return {
        "macro_f1": float(f1.mean()), "accuracy": float(np.mean(y == pred)),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                           "f1": float(f1[i]), "true_count": int(support[i]),
                           "predicted_count": int(np.sum(pred == c))} for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
    }


def tune_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    candidates = np.linspace(0.10, 0.90, 81)
    scored = [(full_metrics(y, decode(probs, t))["macro_f1"], -abs(t - 0.5), float(t)) for t in candidates]
    return max(scored)[2]


def make_detector(seed: int, workers: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=800, class_weight="balanced_subsample", random_state=seed,
                                n_jobs=workers, max_features="sqrt")


def make_file_features(chan_tables, speeds, labels, fit_idx, config):
    ref = NormalReference(config["reference"], config.get("smooth_speed", False)).fit(
        [chan_tables[i] for i in fit_idx], [speeds[i] for i in fit_idx], [labels[i] for i in fit_idx])
    frame = pd.DataFrame([aggregate(ref.excess(ct, speeds[i]), speeds[i]) for i, ct in enumerate(chan_tables)])
    return ref, frame


def side_table(frame: pd.DataFrame, indices: np.ndarray, labels: np.ndarray, config: dict):
    rows, targets = [], []
    for i in indices:
        left, right = side_relative_rows(frame.iloc[i].to_dict())
        rows.extend([left, right]); targets.extend([labels[i] == "Side I", labels[i] == "Side II"])
    table = pd.DataFrame(rows)
    if not config.get("raw_speed", True):
        table = table.drop(columns=["speed"], errors="ignore")
    if not config.get("speed_bin", True):
        table = table.drop(columns=["speed_bin"], errors="ignore")
    return table.replace([np.inf, -np.inf], np.nan).fillna(-9), np.asarray(targets, dtype=int)


def fit_predict(train_idx, test_idx, chan_tables, speeds, labels, config, seed, workers):
    if config.get("speed_only"):
        model = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                       random_state=seed, n_jobs=workers).fit(
            np.asarray(speeds)[train_idx, None], labels[train_idx])
        raw = model.predict_proba(np.asarray(speeds)[test_idx, None])
        class_probs = np.zeros((len(test_idx), 3))
        for j, c in enumerate(model.classes_): class_probs[:, CLASSES.index(c)] = raw[:, j]
        return class_probs, None
    _, frame = make_file_features(chan_tables, speeds, labels, train_idx, config)
    train_rows, targets = side_table(frame, train_idx, labels, config)
    test_rows, _ = side_table(frame, test_idx, labels, config)
    test_rows = test_rows.reindex(columns=train_rows.columns, fill_value=-9)
    sample_weight = None
    if config.get("speed_weight"):
        bins = np.clip(np.digitize(np.asarray(speeds)[train_idx], [0, 3, 6, 9, 12, 15, 30]) - 1, 0, 5)
        fault = labels[train_idx] != "Normal"
        weights = np.ones(len(train_idx), dtype=float)
        for speed_bin in set(bins):
            in_bin = bins == speed_bin
            normal_count = np.sum(in_bin & ~fault); fault_count = np.sum(in_bin & fault)
            if normal_count:
                weights[in_bin & ~fault] = (fault_count + 1) / (normal_count + 1)
        weights /= np.mean(weights)
        sample_weight = np.repeat(weights, 2)
    model = make_detector(seed, workers).fit(train_rows, targets, sample_weight=sample_weight)
    side_probs = model.predict_proba(test_rows)[:, 1].reshape(-1, 2)
    return side_probs, (model, frame, train_rows.columns.tolist())


def inner_threshold(outer_train, chan_tables, speeds, labels, config, seed, workers):
    if config.get("speed_only"):
        return None
    y_outer = labels[outer_train]
    inner_splits = min(3, int(pd.Series(y_outer).value_counts().min()))
    splitter = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    inner_probs = np.zeros((len(outer_train), 2))
    for inner_fold, (itr_local, iva_local) in enumerate(splitter.split(np.zeros(len(outer_train)), y_outer)):
        itr, iva = outer_train[itr_local], outer_train[iva_local]
        probs, _ = fit_predict(itr, iva, chan_tables, speeds, labels, config,
                               seed * 100 + inner_fold, workers)
        inner_probs[iva_local] = probs
    return tune_threshold(y_outer, inner_probs)


def evaluate_protocol(name, splits, chan_tables, speeds, labels, filenames, config, workers):
    n_repeats = max(repeat for repeat, _, _, _ in splits) + 1
    records, thresholds = [], []
    for repeat, fold, train_idx, test_idx in splits:
        started = time.monotonic()
        threshold = inner_threshold(train_idx, chan_tables, speeds, labels, config,
                                    10_000 * repeat + fold + 1, workers)
        probs, _ = fit_predict(train_idx, test_idx, chan_tables, speeds, labels, config,
                               20_000 * repeat + fold + 1, workers)
        if config.get("speed_only"):
            pred = np.asarray(CLASSES)[probs.argmax(axis=1)]
        else:
            pred = decode(probs, threshold)
        thresholds.append({"repeat": repeat, "fold": fold, "threshold": threshold})
        for local, i in enumerate(test_idx):
            rec = {"protocol": name, "repeat": repeat, "fold": fold, "file_id": filenames[i],
                   "true_label": labels[i], "prediction": pred[local], "speed_mps": speeds[i],
                   "elapsed_fold_seconds": time.monotonic() - started}
            if config.get("speed_only"):
                rec.update({f"p_{c}": float(probs[local, j]) for j, c in enumerate(CLASSES)})
            else:
                rec.update({"side_i_evidence": float(probs[local, 0]),
                            "side_ii_evidence": float(probs[local, 1]), "threshold": threshold})
            records.append(rec)
        print(f"{config['id']} {name} repeat={repeat} fold={fold} done", flush=True)
    oof = pd.DataFrame(records)
    reports = []
    for repeat in range(n_repeats):
        part = oof[oof.repeat == repeat]
        if len(part) == len(labels):
            reports.append(full_metrics(part.true_label.to_numpy(), part.prediction.to_numpy()))
    pooled = full_metrics(oof.true_label.to_numpy(), oof.prediction.to_numpy())
    summary = {"pooled": pooled, "per_repeat": reports, "thresholds": thresholds,
               "mean_repeat_macro_f1": float(np.mean([r["macro_f1"] for r in reports])) if reports else None,
               "std_repeat_macro_f1": float(np.std([r["macro_f1"] for r in reports])) if reports else None}
    return oof, summary


def build_splits(labels, filenames, speeds, duplicate_families=()):
    labels = np.asarray(labels); n = len(labels)
    repeated = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    for k, (train, test) in enumerate(rskf.split(np.zeros(n), labels)):
        repeated.append((k // 5, k % 5, train, test))
    order = np.argsort([natural_number(name) for name in filenames])
    block_tests = [set(part.tolist()) for part in np.array_split(order, 5)]
    index_by_name = {name: i for i, name in enumerate(filenames)}
    for family in duplicate_families:
        indices = {index_by_name[name] for name in family}
        target_fold = next(f for f, test in enumerate(block_tests) if min(indices) in test)
        for test in block_tests: test.difference_update(indices)
        block_tests[target_fold].update(indices)
    blocked = []
    for fold, test_set in enumerate(block_tests):
        test = np.asarray(sorted(test_set))
        blocked.append((0, fold, np.setdiff1d(np.arange(n), test), test))
    # Transfer folds are defined by occupied speed bins. They are diagnostics and may lack all classes.
    edges = np.asarray([0, 3, 6, 9, 12, 15, 30])
    bins = np.clip(np.digitize(speeds, edges) - 1, 0, len(edges) - 2)
    transfer = []
    for fold, speed_bin in enumerate(sorted(set(bins))):
        test = np.flatnonzero(bins == speed_bin); train = np.flatnonzero(bins != speed_bin)
        if len(test) and len(set(labels[train])) == 3:
            transfer.append((0, fold, train, test))
    return {"stratified_5x3": repeated, "contiguous_block_5": blocked, "speed_bin_transfer": transfer}


def serialise_splits(splits, filenames):
    return {name: [{"repeat": r, "fold": f,
                    "train": [filenames[i] for i in train], "validation": [filenames[i] for i in test]}
                   for r, f, train, test in definitions] for name, definitions in splits.items()}


def write_experiment(exp_dir, config, oofs, summaries, provenance):
    exp_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in oofs.items():
        temp = exp_dir / f".{name}_oof.csv.tmp"; frame.to_csv(temp, index=False); temp.replace(exp_dir / f"{name}_oof.csv")
    atomic_json(exp_dir / "metrics.json", summaries)
    atomic_json(exp_dir / "config.json", config)
    atomic_json(exp_dir / "provenance.json", provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--experiments", default="RC-E01,RC-E02,RC-E03A,RC-E03B,RC-E03C,RC-E03D,RC-E03E,RC-E04A,RC-E04B")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch == "main" or not branch:
        raise RuntimeError("Rail experiments are forbidden on main or detached HEAD")
    audit = json.loads((args.audit_dir / "data_audit.json").read_text())
    manifest = pd.read_csv(args.audit_dir / "data_manifest.csv")
    labels_frame = pd.read_csv(args.data_dir / "Train_Labels.csv")
    filenames = labels_frame.filename.tolist(); labels = labels_frame.label.to_numpy()
    files = [args.data_dir / "Train" / name for name in filenames]
    pipeline_path = ROOT / "Rail Corrugation/code/pipeline.py"
    feature_code_hash = sha256_file(pipeline_path)
    code_hash = stable_hash({p.name: sha256_file(p) for p in [Path(__file__), pipeline_path]})
    cache_fingerprint = stable_hash({"manifest": audit["source_manifest_fingerprint"], "feature_code": feature_code_hash,
                                     "feature_config": "baseline-channel-v2-position-metadata"})
    cache = args.output_root / "cache" / f"channel_tables_{cache_fingerprint[:16]}.joblib"
    print(json.dumps({"branch": branch, "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "data_dir": str(args.data_dir.resolve()), "output_root": str(args.output_root),
                      "cache_fingerprint": cache_fingerprint, "workers": args.workers}, indent=2))
    if args.dry_run: return
    cached = load_fingerprinted_cache(cache, cache_fingerprint)
    if cached is None:
        values = Parallel(n_jobs=args.workers)(delayed(file_channel_table)(path) for path in files)
        save_fingerprinted_cache(cache, cache_fingerprint, values)
    else:
        values = cached
    chan_tables, speeds = [v[0] for v in values], np.asarray([v[1] for v in values])
    splits = build_splits(labels, filenames, speeds, audit["duplicates"]["exact_file_groups"])
    index_by_name = {name: i for i, name in enumerate(filenames)}
    for family in audit["duplicates"]["exact_file_groups"]:
        family_idx = {index_by_name[name] for name in family}
        for _, fold, _, validation in splits["contiguous_block_5"]:
            overlap = family_idx.intersection(validation)
            if overlap and overlap != family_idx:
                raise RuntimeError(f"duplicate family crosses grouped fold {fold}: {family}")
    split_path = args.output_root / "RC-E01" / "splits.json"
    atomic_json(split_path, serialise_splits(splits, filenames))
    configs = {
        "RC-E01": {"id": "RC-E01", "speed_only": True},
        "RC-E02": {"id": "RC-E02", "reference": "side", "raw_speed": True, "speed_bin": True},
        "RC-E03A": {"id": "RC-E03A", "speed_only": True},
        "RC-E03B": {"id": "RC-E03B", "reference": "side", "raw_speed": True, "speed_bin": True},
        "RC-E03C": {"id": "RC-E03C", "reference": "side", "raw_speed": False, "speed_bin": True},
        "RC-E03D": {"id": "RC-E03D", "reference": "side", "raw_speed": False, "speed_bin": False},
        "RC-E03E": {"id": "RC-E03E", "reference": "side", "smooth_speed": True, "raw_speed": False, "speed_bin": False},
        "RC-E03F": {"id": "RC-E03F", "reference": "side", "raw_speed": False, "speed_bin": False, "speed_weight": True},
        "RC-E04A": {"id": "RC-E04A", "reference": "position", "raw_speed": False, "speed_bin": False},
        "RC-E04B": {"id": "RC-E04B", "reference": "channel", "raw_speed": False, "speed_bin": False},
    }
    requested = [item.strip() for item in args.experiments.split(",") if item.strip()]
    provenance_base = {"branch": branch, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "data_manifest_hash": audit["manifest_sha256"], "code_hash": code_hash,
                       "feature_cache_fingerprint": cache_fingerprint, "seeds": [0],
                       "python": sys.version, "platform": platform.platform(), "libraries": {"numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}}
    for experiment_id in requested:
        config = configs[experiment_id]; started = time.monotonic(); oofs = {}; summaries = {}
        protocol_names = ["stratified_5x3", "contiguous_block_5"]
        if experiment_id in {"RC-E01", "RC-E02", "RC-E03D"}: protocol_names.append("speed_bin_transfer")
        for name in protocol_names:
            oof, summary = evaluate_protocol(name, splits[name], chan_tables, speeds, labels, filenames, config, args.workers)
            oofs[name] = oof; summaries[name] = summary
            if name == "stratified_5x3":
                high = oof[oof.speed_mps >= 9.0]
                summaries["speed_matched_high_speed"] = full_metrics(high.true_label.to_numpy(), high.prediction.to_numpy())
        provenance = dict(provenance_base, elapsed_seconds=time.monotonic() - started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          conclusion="retain for comparison")
        write_experiment(args.output_root / experiment_id, config, oofs, summaries, provenance)
        print(experiment_id, {k: v.get("pooled", v).get("macro_f1") for k, v in summaries.items()}, flush=True)


if __name__ == "__main__":
    main()
