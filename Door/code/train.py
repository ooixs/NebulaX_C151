"""Train the Door classifier with contiguous time-block CV; write weights + final model.

Usage: python Door/code/train.py [--blocks 5] [--model rf|lgbm]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.metrics import door_iou_f1, parse_door_time  # noqa: E402
from Door.code.pipeline import (ReferenceProfiles, RfLrEnsemble, features_table, load_stream,  # noqa: E402
                                match_labels, segment, segments_to_frame)

DATA = ROOT / "data/Door"
WEIGHTS = ROOT / "weights/Door"
MODEL = ROOT / "Door/model"
POS = "Abnormal resistance"


def make_model(kind: str, seed: int = 0):
    if kind == "ens":
        return RfLrEnsemble(seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=2,
                                      random_state=seed, n_jobs=-1)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=5,
                                  subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                                  random_state=seed, verbose=-1)
    raise ValueError(kind)


def run_cv(segs, labels, blocks: int, kind: str, thresholds=np.linspace(0.2, 0.8, 25),
           nested=False, answer=None):
    n = len(segs)
    if not 2 <= blocks <= n:
        raise ValueError("blocks must be between 2 and the number of cycles")
    fold_id = (np.arange(n) * blocks) // n  # contiguous blocks
    truth = segments_to_frame(segs).assign(status=labels) if answer is None else answer
    boundaries = np.array([segs[np.flatnonzero(fold_id == k)[0]]["t"].iloc[0]
                           for k in range(1, blocks)], dtype="datetime64[ns]")
    truth_fold_id = np.searchsorted(boundaries, truth.start_time.map(parse_door_time).to_numpy(), side="right")
    oof = np.zeros(n)
    pred_b = np.zeros(n, dtype=bool)
    fold_reports = []
    for k in range(blocks):
        tr, te = fold_id != k, fold_id == k
        train_segs = [s for s, m in zip(segs, tr) if m]
        train_labels = [y for y, m in zip(labels, tr) if m]
        ref = ReferenceProfiles().fit(train_segs, train_labels)
        Xtr = features_table(train_segs, ref)
        Xte = features_table([s for s, m in zip(segs, te) if m], ref)
        ytr = np.array([y == POS for y in train_labels])
        clf = make_model(kind).fit(Xtr, ytr)
        oof[te] = clf.predict_proba(Xte)[:, 1]
        report = dict(fold=k, n_test=int(te.sum()), n_pos_test=int(sum(y == POS for y, m in zip(labels, te) if m)))
        if nested:
            inner = run_cv(train_segs, train_labels, max(2, blocks - 1), kind, thresholds,
                           answer=truth.loc[truth_fold_id != k])
            report["threshold"] = inner["threshold"]
            report["threshold_training_indices"] = np.flatnonzero(tr).tolist()
            report["validation_indices"] = np.flatnonzero(te).tolist()
            pred_b[te] = oof[te] >= inner["threshold"]
        fold_reports.append(report)
    # pick threshold on OOF end-to-end IoU-F1; among equally good thresholds take the centre of the interval
    scores = []
    for th in thresholds:
        pred = segments_to_frame(segs, [POS if p >= th else "Normal" for p in oof])
        scores.append(door_iou_f1(truth, pred)["score"])
    scores = np.array(scores)
    top = thresholds[scores >= scores.max() - 1e-9]
    best = (float(np.median(top)), float(scores.max()))
    if not nested:
        pred_b = oof >= best[0]
    # per-fold end-to-end score at the chosen threshold
    for r in fold_reports:
        m = fold_id == r["fold"]
        t_f = truth.loc[truth_fold_id == r["fold"]]
        p_f = segments_to_frame([s for s, mm in zip(segs, m) if mm], [POS if p else "Normal" for p in pred_b[m]])
        r["iou_f1"] = door_iou_f1(t_f, p_f)["score"]
    prediction = segments_to_frame(segs, [POS if p else "Normal" for p in pred_b])
    score = door_iou_f1(truth, prediction)["score"]
    yb = np.array([y == POS for y in labels])
    tp, fp, fn = int((pred_b & yb).sum()), int((pred_b & ~yb).sum()), int((~pred_b & yb).sum())
    return dict(model=kind, blocks=blocks, threshold=best[0], oof_iou_f1=score, folds=fold_reports,
                threshold_calibration_iou_f1=best[1], nested_thresholds=nested,
                oof_tp=tp, oof_fp=fp, oof_fn=fn, oof=oof.tolist(), oof_pred=pred_b.tolist())


def research(args):
    from common.research import ResearchRun

    run = ResearchRun("Door", args.run_id, args.min_delta, args.patience)
    df = load_stream(DATA / "Train.csv")
    answer = pd.read_csv(DATA / "Train_Segments_Answer.csv")
    segs = segment(df)
    labels = match_labels(segs, answer)
    if any(label is None for label in labels):
        raise ValueError("segmentation does not reproduce the labelled cycle starts")
    segmentation = door_iou_f1(answer, segments_to_frame(segs), ignore_labels=True)["score"]
    print(f"Door: {len(segs)} training cycles; segmentation IoU-F1={segmentation:.6f}", flush=True)
    best = None
    for kind in ("ens", "rf", "lgbm"):
        report = run_cv(segs, labels, args.blocks, kind, np.linspace(0.05, 0.95, 91), nested=True, answer=answer)
        run.save_json(f"cv_{kind}.json", report)
        result = run.record(kind, [report["oof_iou_f1"]],
                            per_fold_iou_f1=[fold["iou_f1"] for fold in report["folds"]],
                            end_to_end_iou_f1=report["oof_iou_f1"], segmentation_iou_f1=segmentation)
        if result["accepted"]:
            best = report
        if run.tracker.stop_reason:
            break
    if run.tracker.stop_reason is None:
        run.finish(published=False)
        raise RuntimeError("Door candidate list exhausted before convergence; extend the search")
    ref = ReferenceProfiles().fit(segs, labels)
    X = features_table(segs, ref)
    clf = make_model(best["model"]).fit(X, np.array([y == POS for y in labels]))
    artifact = dict(kind=best["model"], clf=clf, ref=ref, threshold=best["threshold"],
                    feature_names=list(X.columns), cv_iou_f1=best["oof_iou_f1"],
                    validation="nested contiguous-block CV", research_run=run.run_id)
    output = run.path / "door_model.joblib"
    joblib.dump(artifact, output)
    if args.ship:
        run.publish(output, MODEL / output.name)
    run.finish(published=args.ship, model=output.name, threshold=best["threshold"],
               segmentation_iou_f1=segmentation, training_cycles=len(segs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--model", default="ens", choices=["rf", "lgbm", "ens"])
    ap.add_argument("--ship", action="store_true", help="also copy this model to Door/model/door_model.joblib")
    ap.add_argument("--research", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    if args.research:
        research(args)
        return
    WEIGHTS.mkdir(parents=True, exist_ok=True); MODEL.mkdir(parents=True, exist_ok=True)

    df = load_stream(DATA / "Train.csv")
    answer = pd.read_csv(DATA / "Train_Segments_Answer.csv")
    segs = segment(df)
    labels = match_labels(segs, answer)
    assert all(l is not None for l in labels), "segmentation does not reproduce answer starts"
    seg_score = door_iou_f1(answer.rename(columns={}), segments_to_frame(segs, ["x"] * len(segs)), ignore_labels=True)
    print(f"segmentation-only IoU-F1 = {seg_score['score']:.4f}  ({len(segs)} segments)")

    rep = run_cv(segs, labels, args.blocks, args.model, thresholds=np.linspace(0.05, 0.95, 91), nested=True, answer=answer)
    oof = np.array(rep["oof"]); yb = np.array([l == POS for l in labels])
    print(f"[{args.model}] OOF end-to-end IoU-F1 = {rep['oof_iou_f1']:.4f} @ thr={rep['threshold']:.3f}  "
          f"TP={rep['oof_tp']} FP={rep['oof_fp']} FN={rep['oof_fn']}  "
          f"OOF margin: normal max={oof[~yb].max():.3f} abnormal min={oof[yb].min():.3f}")
    for f in rep["folds"]:
        print(f"  fold {f['fold']}: n={f['n_test']} pos={f['n_pos_test']} iou_f1={f['iou_f1']:.4f}")
    rep["segmentation_iou_f1"] = seg_score["score"]
    (WEIGHTS / f"cv_{args.model}.json").write_text(json.dumps(rep, indent=1))

    # final fit on all data
    ref = ReferenceProfiles().fit(segs, labels)
    X = features_table(segs, ref)
    clf = make_model(args.model).fit(X, np.array([y == POS for y in labels]))
    artefact = dict(kind=args.model, clf=clf, ref=ref, threshold=rep["threshold"], feature_names=list(X.columns),
                    cv_iou_f1=rep["oof_iou_f1"])
    joblib.dump(artefact, WEIGHTS / f"door_{args.model}.joblib")
    print("saved", WEIGHTS / f"door_{args.model}.joblib")
    if args.ship:
        from common.artifacts import publish_model
        publish_model(WEIGHTS / f"door_{args.model}.joblib", MODEL / "door_model.joblib")
        print("SHIPPED ->", MODEL / "door_model.joblib")


if __name__ == "__main__":
    main()
