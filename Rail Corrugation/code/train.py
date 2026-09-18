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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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


def load_all(files: list[Path], cache: Path, jobs=-1):
    if cache.exists():
        return joblib.load(cache)
    res = Parallel(n_jobs=jobs)(delayed(file_channel_table)(f) for f in files)
    joblib.dump(res, cache)
    return res


def make_rf(seed=0):
    return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=1,
                                  max_features="sqrt", random_state=seed, n_jobs=-1)


def make_detector(seed=0):
    """Per-side detector. ExtraTrees chosen in the improvement phase (0.834 vs RF 0.819 macro-F1)."""
    return ExtraTreesClassifier(n_estimators=800, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)


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


def research_table(chan_tables, speeds, labels, train_indices):
    ref = NormalReference().fit([chan_tables[i] for i in train_indices],
                                [speeds[i] for i in train_indices], [labels[i] for i in train_indices])
    rows = []
    for ct, v in zip(chan_tables, speeds):
        features = aggregate(ref.excess(ct, v), v, paired=True)
        rows.extend(side_relative_rows(features, engineered=True))
    return ref, pd.DataFrame(rows).fillna(-9).astype(np.float32)


def research_columns(columns, mode):
    legacy = [c for c in columns if not c.startswith(("own_pair_", "rel_pair_", "ratio_")) and c != "side_id"]
    if mode == "legacy":
        return legacy
    if mode == "ratios":
        return legacy + [c for c in columns if c.startswith("ratio_") or c == "side_id"]
    if mode == "compact":
        physics = ("log_rms", "shock_log_rms", "spec_entropy", "dom_prom", "lb_", "fb_200_400", "fb_400_800", "fb_800_1600")
        return [c for c in columns if c in ("speed", "speed_bin", "side_id") or c.startswith("ratio_") or
                (any(part in c for part in physics) and c.endswith(("_mean", "_median", "_p90", "_positive_frac")))]
    if mode == "cardom":
        return legacy + [c for c in columns if "pair_car_" in c or "pair_speednorm" in c]
    if mode == "paired":
        return [c for c in columns if c.startswith(("own_pair_", "rel_pair_", "ratio_")) or
                c in ("speed", "speed_bin", "side_id", "own_log_rms_mean", "rel_log_rms_mean")]
    return list(columns)


def research_model(config, seed, jobs):
    options = dict(n_estimators=config.get("trees", 800), class_weight=config.get("class_weight", "balanced_subsample"),
                   min_samples_leaf=config.get("leaf", 1), max_features=config.get("max_features", "sqrt"),
                   random_state=seed, n_jobs=jobs)
    kind = config.get("kind", "et")
    if kind == "et":
        return ExtraTreesClassifier(**options)
    if kind == "rf":
        return RandomForestClassifier(**options)
    if kind == "svc":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(C=config.get("C", 1.0), class_weight="balanced",
                                                   probability=True, random_state=seed))
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                              reg_lambda=5.0, colsample_bytree=0.8, class_weight="balanced",
                              random_state=seed, n_jobs=jobs, verbosity=-1)
    raise ValueError(kind)


def tune_side_threshold(labels, probabilities):
    grid = np.linspace(0.1, 0.9, 33)
    scores = np.array([rail_macro_f1(labels, decode_side(probabilities[:, 0], probabilities[:, 1], tau))["macro_f1"]
                       for tau in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(grid[best[(len(best) - 1) // 2]]), float(scores.max())


def research(args):
    from sklearn.model_selection import StratifiedKFold
    from common.research import ResearchRun, fingerprint, nested_folds

    run = ResearchRun("Rail Corrugation", args.run_id, args.min_delta, args.patience)
    lab = pd.read_csv(DATA / "Train_Labels.csv")
    y = lab.label.to_numpy()
    n = len(y)
    cache_path = WEIGHTS / "train_channel_tables.joblib"
    cached = load_all([DATA / "Train" / filename for filename in lab.filename], cache_path, jobs=args.jobs)
    if len(cached) != n:
        raise ValueError("channel cache does not match the labelled files")
    CT, V = [c for c, _ in cached], [v for _, v in cached]
    checked = sorted({0, n // 2, n - 1})
    for i in checked:
        ct, v = file_channel_table(DATA / "Train" / lab.filename.iloc[i])
        if not np.isclose(v, V[i]) or list(ct.columns) != list(CT[i].columns) or not np.allclose(ct, CT[i], equal_nan=True):
            raise ValueError("stale channel cache; rebuild before research")
    run.save_json("inputs.json", dict(labels_sha256=fingerprint(DATA / "Train_Labels.csv"),
                  channel_cache_sha256=fingerprint(cache_path), checked_files=lab.filename.iloc[checked].tolist(),
                  files=lab.filename.tolist(), training_only=True))
    folds = nested_folds(y, args.folds, args.repeats, args.inner_folds)
    run.save_json("folds.json", folds)
    tables = {}
    def prime(protocol):
        pending = {}
        for fold in protocol:
            for indices in [fold["train"]] + [pair[0] for pair in fold["inner"]]:
                key = tuple(indices)
                if key not in tables:
                    pending[key] = indices
        if not pending:
            return
        print(f"Preparing {len(pending)} fold-local reference/feature tables ...", flush=True)
        def table(indices):
            return research_table(CT, V, y, indices)[1]
        values = Parallel(n_jobs=args.jobs)(delayed(table)(indices) for indices in pending.values())
        tables.update(zip(pending, values))
        print(f"Feature tables ready ({len(tables)} cached)", flush=True)
    prime(folds)
    row_indices = lambda indices: (2 * np.asarray(indices)[:, None] + np.arange(2)).ravel()
    side_y = (y[:, None] == np.array(["Side I", "Side II"])).astype(int).ravel()
    def evaluate(config, protocol, tag):
        repeats = max(f["repeat"] for f in protocol) + 1
        oof = np.zeros((repeats, n, 2))
        prediction = np.full((repeats, n), "", dtype="<U7")
        reports = []
        for fold in protocol:
            k, repeat, tr, te = fold["fold"], fold["repeat"], fold["train"], fold["validation"]
            R = tables[tuple(tr)]
            cols = research_columns(R.columns, config["features"])
            clf = research_model(config, k, args.jobs).fit(R.iloc[row_indices(tr)][cols], side_y[row_indices(tr)])
            probabilities = clf.predict_proba(R.iloc[row_indices(te)][cols])[:, 1].reshape(-1, 2)
            calibration = np.zeros((n, 2))
            for j, (inner_tr, inner_te) in enumerate(fold["inner"]):
                Ri = tables[tuple(inner_tr)]
                model = research_model(config, 10000 + 10 * k + j, args.jobs)
                model.fit(Ri.iloc[row_indices(inner_tr)][cols], side_y[row_indices(inner_tr)])
                calibration[inner_te] = model.predict_proba(Ri.iloc[row_indices(inner_te)][cols])[:, 1].reshape(-1, 2)
            tau, inner_score = tune_side_threshold(y[tr], calibration[tr])
            oof[repeat, te] = probabilities
            prediction[repeat, te] = decode_side(probabilities[:, 0], probabilities[:, 1], tau)
            reports.append(dict(fold=k, repeat=repeat, threshold=tau, inner_calibration_macro_f1=inner_score,
                                validation_indices=te.tolist(), threshold_training_indices=tr.tolist()))
            if (k + 1) % args.folds == 0:
                print(f"  {tag}: {k + 1}/{len(protocol)} outer folds", flush=True)
        scores = [rail_macro_f1(y, p)["macro_f1"] for p in prediction]
        per_class = {c: float(np.mean([rail_macro_f1(y, p)["per_class"][c] for p in prediction])) for c in CLASSES}
        calibrated = [tune_side_threshold(y, p) for p in oof]
        result = dict(scores=scores, mean=float(np.mean(scores)), std=float(np.std(scores)), per_class=per_class,
                      final_tau=float(np.median([t for t, _ in calibrated])),
                      optimistic_same_oof_tuned_scores=[s for _, s in calibrated], folds=reports)
        run.save_json(f"{tag}.json", result)
        np.savez_compressed(run.path / f"{tag}_oof.npz", probabilities=oof, prediction=prediction, labels=y.astype("U7"))
        return result
    if args.candidate_set == "sg":
        candidates = [
            dict(name="et_legacy", features="legacy"),
            dict(name="et_cardom", features="cardom"),
            dict(name="rf_cardom", kind="rf", features="cardom"),
            dict(name="lgbm_cardom", kind="lgbm", features="cardom"),
            dict(name="et_cardom_mf03", features="cardom", max_features=0.3),
            dict(name="et_cardom_leaf2", features="cardom", leaf=2),
        ]
    else:
        candidates = [
        dict(name="et_legacy", features="legacy"),
        dict(name="et_ratios_identity", features="ratios"),
        dict(name="et_paired_channels", features="all"),
        dict(name="et_compact_physics", features="compact"),
        dict(name="rf_compact_physics", kind="rf", features="compact"),
        dict(name="svc_compact_C1", kind="svc", features="compact", C=1.0),
        dict(name="et_paired_unweighted", features="all", class_weight=None),
        dict(name="lgbm_compact", kind="lgbm", features="compact"),
        dict(name="et_paired_leaf2", features="all", leaf=2),
        dict(name="svc_compact_C10", kind="svc", features="compact", C=10.0),
        dict(name="et_paired_mf03", features="all", max_features=0.3),
        dict(name="et_spatial_only", features="paired"),
        dict(name="et_ratios_leaf3", features="ratios", leaf=3),
        dict(name="svc_ratios_C1", kind="svc", features="ratios", C=1.0),
        dict(name="et_compact_unweighted", features="compact", class_weight=None),
        dict(name="et_ratios_1600trees", features="ratios", trees=1600),
        dict(name="svc_compact_C01", kind="svc", features="compact", C=0.1),
        dict(name="rf_paired_leaf2", kind="rf", features="all", leaf=2, max_features=0.3),
    ]
    results, accepted = {}, []
    for trial, config in enumerate(candidates):
        name = config["name"]
        result = evaluate(config, folds, f"trial_{trial:02d}_{name}")
        results[name] = result
        record = run.record(name, result["scores"], parameters=config, per_class=result["per_class"],
                            optimistic_same_oof_tuned_scores=result["optimistic_same_oof_tuned_scores"])
        if record["accepted"]:
            accepted.append(config)
        if run.tracker.stop_reason:
            break
    if run.tracker.stop_reason is None:
        run.finish(published=False)
        raise RuntimeError("Rail candidate list exhausted before convergence; extend the search")
    confirmation_folds = nested_folds(y, args.folds, args.confirmation_repeats, args.inner_folds, seed=2026)
    run.save_json("confirmation_folds.json", confirmation_folds)
    prime(confirmation_folds)
    baseline = candidates[0]
    baseline_confirmation = evaluate(baseline, confirmation_folds, "confirmation_baseline")
    confirmation = {baseline["name"]: baseline_confirmation}
    selected = baseline
    for config in reversed(accepted[1:]):
        result = evaluate(config, confirmation_folds, f"confirmation_{config['name']}")
        gain = result["mean"] - baseline_confirmation["mean"]
        required = max(args.min_delta, baseline_confirmation["std"])
        consistent = bool(np.all(np.array(result["scores"]) > np.array(baseline_confirmation["scores"])))
        confirmation[config["name"]] = dict(**result, gain=gain, required_gain=required, accepted=consistent and gain > required)
        if consistent and gain > required:
            selected = config
            break
    run.save_json("confirmation.json", confirmation)
    ordered = np.argsort(lab.filename.str.extract(r"(\d+)", expand=False).astype(int).to_numpy())
    blocked = []
    for k, te in enumerate(np.array_split(ordered, args.folds)):
        tr = np.setdiff1d(np.arange(n), te)
        inner = StratifiedKFold(args.inner_folds, shuffle=True, random_state=500 + k)
        pairs = [(tr[a], tr[b]) for a, b in inner.split(np.zeros(len(tr)), y[tr])]
        blocked.append(dict(fold=k, repeat=0, train=tr, validation=te, inner=pairs))
    run.save_json("blocked_folds.json", blocked)
    prime(blocked)
    block_base = evaluate(baseline, blocked, "blocked_baseline")
    block_selected = block_base if selected == baseline else evaluate(selected, blocked, "blocked_selected")
    block_rejected = block_selected["mean"] < block_base["mean"] - 0.02
    if block_rejected:
        selected, block_selected = baseline, block_base
    final_result = results[selected["name"]]
    ref, R = research_table(CT, V, y, np.arange(n))
    cols = research_columns(R.columns, selected["features"])
    detector = research_model(selected, 0, args.jobs).fit(R[cols], side_y)
    artifact = dict(choice="side_detector", ref=ref, det=detector, det_cols=cols, tau=final_result["final_tau"],
                    engineered_features=selected["features"] != "legacy", cv_macro_f1=final_result["mean"],
                    research_run=run.run_id, parameters=selected, validation="nested reference and threshold CV")
    output = run.path / "rail_model.joblib"
    joblib.dump(artifact, output)
    joblib.dump(tables, run.path / "fold_feature_tables.joblib", compress=3)
    if args.ship:
        run.publish(output, MODEL / output.name)
    run.finish(published=args.ship, model=output.name, retained_name=selected["name"],
               retained_cv_mean=final_result["mean"], retained_cv_std=final_result["std"],
               retained_confirmation_mean=confirmation[selected["name"]]["mean"],
               blocked_cv=block_selected["mean"], blocked_baseline_cv=block_base["mean"],
               rejected_for_blocked_regression=block_rejected, tau=artifact["tau"],
               per_class=final_result["per_class"], training_files=n,
               limitation="Repeated CV uses the same 272 files, including only 14 Side I examples; no independent labelled test set is available.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--research", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--confirmation-repeats", type=int, default=2)
    ap.add_argument("--candidate-set", default="default", choices=["default", "sg"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--ship", action="store_true")
    args = ap.parse_args()
    if args.research:
        research(args)
        return
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
        det = make_detector(k).fit(Rtr, ys_tr)
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
    det = make_detector(0).fit(R, ys)
    art = dict(choice=choice, ref=ref, clf=clf, clf_cols=cols, det=det, det_cols=list(R.columns), tau=tau,
               cv_macro_f1=summary[choice if choice in summary else "baseline_argmax"][0])
    joblib.dump(art, WEIGHTS / "rail_artefact.joblib"); joblib.dump(art, MODEL / "rail_model.joblib")
    print("saved", MODEL / "rail_model.joblib")


if __name__ == "__main__":
    main()
