"""Rail Corrugation training: repeated stratified K-fold, baseline 3-class RF vs per-side detector.

Usage: python "Rail Corrugation/code/train.py" [--repeats 3] [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Rail Corrugation"))
from code.pipeline import (CLASSES, NormalReference, aggregate, file_channel_table,  # noqa: E402
                           side_relative_rows)
from common.metrics import rail_macro_f1  # noqa: E402

DATA = ROOT / "data/Rail_Corrugation"
WEIGHTS = ROOT / "weights/Rail Corrugation"
MODEL = ROOT / "Rail Corrugation/model"


def load_all(files: list[Path], cache: Path):
    if cache.exists():
        return joblib.load(cache)
    res = Parallel(n_jobs=-1)(delayed(file_channel_table)(f) for f in files)
    joblib.dump(res, cache)
    return res


def make_rf(seed=0):
    return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=1,
                                  max_features="sqrt", random_state=seed, n_jobs=-1)


def build_tables(chan_tables, speeds, labels, tr_idx, te_idx):
    ref = NormalReference().fit([chan_tables[i] for i in tr_idx], [speeds[i] for i in tr_idx], [labels[i] for i in tr_idx])
    feats = [aggregate(ref.excess(chan_tables[i], speeds[i]), speeds[i]) for i in range(len(chan_tables))]
    F = pd.DataFrame(feats)
    return ref, F


def decode_side(p1: np.ndarray, p2: np.ndarray, tau: float) -> list[str]:
    out = []
    for a, b in zip(p1, p2):
        if max(a, b) < tau:
            out.append("Normal")
        else:
            out.append("Side I" if a >= b else "Side II")
    return out


def tune_3class(proba: np.ndarray, y: np.ndarray):
    """Scale minority-class probabilities by a factor to maximise macro-F1 (OOF)."""
    best = (rail_macro_f1(y, [CLASSES[i] for i in proba.argmax(1)])["macro_f1"], 1.0, 1.0)
    for w1 in np.linspace(0.5, 4, 15):
        for w2 in np.linspace(0.5, 4, 15):
            pr = proba * np.array([1.0, w1, w2])
            sc = rail_macro_f1(y, [CLASSES[i] for i in pr.argmax(1)])["macro_f1"]
            if sc > best[0]:
                best = (sc, float(w1), float(w2))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    WEIGHTS.mkdir(parents=True, exist_ok=True); MODEL.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(DATA / "Train_Labels.csv")
    files = [DATA / "Train" / f for f in lab.filename]
    y = lab.label.to_numpy()
    print("extracting channel features ...")
    ct_v = load_all(files, WEIGHTS / "train_channel_tables.joblib")
    chan_tables, speeds = [c for c, _ in ct_v], [v for _, v in ct_v]
    print("speed (m/s) by class:", pd.Series(speeds).groupby(y).describe()[["min", "50%", "max"]].round(1).to_dict())

    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=0)
    n = len(files)
    oof_base = np.zeros((args.repeats, n, 3)); oof_side = np.zeros((args.repeats, n, 2))
    for k, (tr, te) in enumerate(rskf.split(np.zeros(n), y)):
        rep = k // args.folds
        ref, F = build_tables(chan_tables, speeds, y, tr, te)
        cols = [c for c in F.columns]
        Xtr, Xte = F.iloc[tr][cols].fillna(-9), F.iloc[te][cols].fillna(-9)
        # baseline 3-class
        clf = make_rf(k).fit(Xtr, y[tr])
        pr = clf.predict_proba(Xte)
        oof_base[rep][te] = pr[:, [list(clf.classes_).index(c) for c in CLASSES]]
        # challenger: per-side detector on (file, side) rows
        rows_tr, ys_tr = [], []
        for i in tr:
            r1, r2 = side_relative_rows(F.iloc[i].to_dict())
            rows_tr += [r1, r2]; ys_tr += [int(y[i] == "Side I"), int(y[i] == "Side II")]
        Rtr = pd.DataFrame(rows_tr).fillna(-9)
        det = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=k, n_jobs=-1).fit(Rtr, ys_tr)
        for i in te:
            r1, r2 = side_relative_rows(F.iloc[i].to_dict())
            Rte = pd.DataFrame([r1, r2]).fillna(-9)[Rtr.columns]
            oof_side[rep][i] = det.predict_proba(Rte)[:, 1]
        print(f"  fold {k+1}/{args.folds*args.repeats} done", flush=True)

    # evaluate per repeat
    res = dict(baseline_argmax=[], baseline_tuned=[], side_detector=[], tuned_w=[], taus=[])
    for rep in range(args.repeats):
        pb = oof_base[rep]
        res["baseline_argmax"].append(rail_macro_f1(y, [CLASSES[i] for i in pb.argmax(1)])["macro_f1"])
        sc, w1, w2 = tune_3class(pb, y)
        res["baseline_tuned"].append(sc); res["tuned_w"].append((w1, w2))
        ps = oof_side[rep]
        best = max(((rail_macro_f1(y, decode_side(ps[:, 0], ps[:, 1], t))["macro_f1"], t) for t in np.linspace(0.1, 0.9, 33)))
        res["side_detector"].append(best[0]); res["taus"].append(best[1])
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items() if k in ("baseline_argmax", "baseline_tuned", "side_detector")}
    print("macro-F1 mean±std over repeats:", {k: f"{m:.4f}±{s:.4f}" for k, (m, s) in summary.items()})
    # per-class report for the mean-OOF baseline (argmax) and side detector
    pb = oof_base.mean(0); ps = oof_side.mean(0)
    base_pred = [CLASSES[i] for i in pb.argmax(1)]
    tau = float(np.mean(res["taus"])); side_pred = decode_side(ps[:, 0], ps[:, 1], tau)
    rep_base = rail_macro_f1(y, base_pred); rep_side = rail_macro_f1(y, side_pred)
    print("baseline per-class F1", {k: round(v, 3) for k, v in rep_base["per_class"].items()})
    print("side-det per-class F1", {k: round(v, 3) for k, v in rep_side["per_class"].items()}, "tau", round(tau, 3))
    print("baseline confusion:\n", pd.crosstab(pd.Series(y, name="true"), pd.Series(base_pred, name="pred")))
    print("side-det confusion:\n", pd.crosstab(pd.Series(y, name="true"), pd.Series(side_pred, name="pred")))

    # acceptance: challenger must beat baseline by > 1 std of baseline
    m_b, s_b = summary["baseline_argmax"]; m_s, _ = summary["side_detector"]
    choice = "side_detector" if m_s > m_b + s_b else "baseline_3class"
    print("SHIPPED:", choice)
    report = dict(summary=summary, per_repeat=res, choice=choice, tau=tau,
                  baseline_per_class=rep_base["per_class"], side_per_class=rep_side["per_class"],
                  speeds=speeds, labels=y.tolist())
    (WEIGHTS / "cv_report.json").write_text(json.dumps(report, indent=1, default=float))

    # final fit on all data
    ref, F = build_tables(chan_tables, speeds, y, np.arange(n), np.arange(n))
    cols = list(F.columns)
    clf = make_rf(0).fit(F[cols].fillna(-9), y)
    rows, ys = [], []
    for i in range(n):
        r1, r2 = side_relative_rows(F.iloc[i].to_dict()); rows += [r1, r2]; ys += [int(y[i] == "Side I"), int(y[i] == "Side II")]
    R = pd.DataFrame(rows).fillna(-9)
    det = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(R, ys)
    art = dict(choice=choice, ref=ref, clf=clf, clf_cols=cols, det=det, det_cols=list(R.columns), tau=tau,
               cv_macro_f1=summary[choice if choice in summary else "baseline_argmax"][0])
    joblib.dump(art, WEIGHTS / "rail_artefact.joblib"); joblib.dump(art, MODEL / "rail_model.joblib")
    print("saved", MODEL / "rail_model.joblib")


if __name__ == "__main__":
    main()
