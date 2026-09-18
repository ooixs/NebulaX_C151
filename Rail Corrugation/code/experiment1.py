"""Experiment 1: decision-layer re-selection for the FIXED Rail ExtraTrees detector.

The model, features and normal reference are unchanged. Only the mapping from side probabilities
to labels is a candidate. Selection happens on inner contiguous file-number blocks; scoring on
outer contiguous blocks (the protocol that tracked the hidden score best).

Usage: python "Rail Corrugation/code/experiment1.py" --run-id <new-id> [--blocks 5] [--inner-blocks 3]
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.metrics import rail_macro_f1  # noqa: E402
from common.research import ResearchRun, fingerprint  # noqa: E402
from rail_corrugation.decision import RULES, block_protocol, decode, legacy_decision, select  # noqa: E402
from rail_corrugation.pipeline import CLASSES, file_channel_table  # noqa: E402
from rail_corrugation.train import load_all, research_columns, research_model, research_table  # noqa: E402

DATA = ROOT / "data/Rail_Corrugation"
WEIGHTS = ROOT / "weights/Rail Corrugation"
MODEL = ROOT / "Rail Corrugation/model"
INCUMBENT = legacy_decision(0.25)


def confusion(labels, prediction):
    return {t: {p: int(np.sum((labels == t) & (prediction == p))) for p in CLASSES} for t in CLASSES}


def probabilities_for(protocol, tables, y, side_y, config, jobs):
    """Outer validation probabilities and inner out-of-fold calibration probabilities, per fold."""
    row_indices = lambda idx: (2 * np.asarray(idx)[:, None] + np.arange(2)).ravel()
    out = []
    for fold in protocol:
        k, tr, te = fold["fold"], fold["train"], fold["validation"]
        R = tables[tuple(tr)]
        cols = research_columns(R.columns, config["features"])
        clf = research_model(config, k, jobs).fit(R.iloc[row_indices(tr)][cols], side_y[row_indices(tr)])
        outer = clf.predict_proba(R.iloc[row_indices(te)][cols])[:, 1].reshape(-1, 2)
        inner_oof = np.full((len(y), 2), np.nan)
        for j, (inner_tr, inner_te) in enumerate(fold["inner"]):
            Ri = tables[tuple(inner_tr)]
            model = research_model(config, 10000 + 10 * k + j, jobs)
            model.fit(Ri.iloc[row_indices(inner_tr)][cols], side_y[row_indices(inner_tr)])
            inner_oof[inner_te] = model.predict_proba(Ri.iloc[row_indices(inner_te)][cols])[:, 1].reshape(-1, 2)
        assert not np.isnan(inner_oof[tr]).any()
        out.append(dict(outer=outer, inner=inner_oof[tr]))
        print(f"  fold {k + 1}/{len(protocol)} probabilities ready", flush=True)
    return out


def evaluate_rule(name, protocol, probs, y, selector):
    """selector(fold, inner_labels, inner_probs) -> decision dict. Returns pooled scores per repeat."""
    repeats = max(f["repeat"] for f in protocol) + 1
    prediction = np.full((repeats, len(y)), "", dtype="<U7")
    decisions = []
    for fold, p in zip(protocol, probs):
        decision = selector(fold, y[fold["train"]], p["inner"])
        prediction[fold["repeat"], fold["validation"]] = decode(p["outer"][:, 0], p["outer"][:, 1], decision)
        decisions.append(dict(fold=fold["fold"], repeat=fold["repeat"], decision=decision,
                              validation_indices=fold["validation"].tolist(),
                              selection_indices=fold["train"].tolist()))
    assert (prediction != "").all()
    scores = [rail_macro_f1(y, p)["macro_f1"] for p in prediction]
    per_class = {c: float(np.mean([rail_macro_f1(y, p)["per_class"][c] for p in prediction])) for c in CLASSES}
    pooled = confusion(y, prediction[0])
    wrong_side = int(sum(((y == "Side I") & (prediction[0] == "Side II")) | ((y == "Side II") & (prediction[0] == "Side I"))))
    return dict(name=name, scores=scores, mean=float(np.mean(scores)), std=float(np.std(scores)), per_class=per_class,
                confusion_first_partition=pooled, wrong_side_first_partition=wrong_side, folds=decisions,
                prediction=prediction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--inner-blocks", type=int, default=3)
    ap.add_argument("--offsets", default="0,27,54", help="rotation offsets (files) for repeated block partitions")
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    offsets = tuple(int(o) for o in args.offsets.split(","))

    run = ResearchRun("Rail Corrugation", args.run_id, args.min_delta, patience=len(RULES))
    lab = pd.read_csv(DATA / "Train_Labels.csv")
    y = lab.label.to_numpy()
    n = len(y)
    numbers = lab.filename.str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    cache_path = WEIGHTS / "train_channel_tables.joblib"
    cached = load_all([DATA / "Train" / f for f in lab.filename], cache_path, jobs=args.jobs)
    CT, V = [c for c, _ in cached], [v for _, v in cached]
    for i in sorted({0, n // 2, n - 1}):
        ct, v = file_channel_table(DATA / "Train" / lab.filename.iloc[i])
        if not np.isclose(v, V[i]) or not np.allclose(ct, CT[i], equal_nan=True):
            raise ValueError("stale channel cache; rebuild before research")
    run.save_json("inputs.json", dict(labels_sha256=fingerprint(DATA / "Train_Labels.csv"),
                  channel_cache_sha256=fingerprint(cache_path), files=lab.filename.tolist(),
                  file_numbers=numbers.tolist(), training_only=True, model_fixed=True))

    protocol = block_protocol(numbers, args.blocks, args.inner_blocks, offsets)
    run.save_json("block_protocol.json", protocol)
    for fold in protocol:
        te = fold["validation"]
        print(f"fold {fold['fold']:2d} (offset {fold['offset']:2d}): files {numbers[te].min()}-{numbers[te].max()} "
              f"n={len(te)} Normal={int((y[te]=='Normal').sum())} SideI={int((y[te]=='Side I').sum())} "
              f"SideII={int((y[te]=='Side II').sum())}", flush=True)

    tables, pending = {}, {}
    for fold in protocol:
        for idx in [fold["train"]] + [pair[0] for pair in fold["inner"]]:
            pending.setdefault(tuple(idx), idx)
    print(f"Preparing {len(pending)} fold-local reference/feature tables ...", flush=True)
    values = Parallel(n_jobs=args.jobs)(delayed(lambda idx: research_table(CT, V, y, idx)[1])(idx) for idx in pending.values())
    tables.update(zip(pending, values))
    side_y = (y[:, None] == np.array(["Side I", "Side II"])).astype(int).ravel()
    config = dict(name="et_legacy", features="legacy")
    probs = probabilities_for(protocol, tables, y, side_y, config, args.jobs)
    outer_oof = np.zeros((len(offsets), n, 2))
    for fold, p in zip(protocol, probs):
        outer_oof[fold["repeat"], fold["validation"]] = p["outer"]
    np.savez_compressed(run.path / "block_probabilities.npz", outer=outer_oof, labels=y.astype("U7"))

    # ---- diagnosis of the incumbent under block validation
    incumbent = evaluate_rule("incumbent_fixed_tau_0.25", protocol, probs, y, lambda fold, yl, pi: INCUMBENT)
    print("\nIncumbent (tau=0.25) block-CV macro-F1: "
          f"{incumbent['mean']:.4f} ± {incumbent['std']:.4f}  per-class {incumbent['per_class']}")
    print("confusion (first partition, rows=true, cols=pred):")
    print(pd.DataFrame(incumbent["confusion_first_partition"]).T.to_string())
    print(f"wrong-side calls: {incumbent['wrong_side_first_partition']}")
    mx = outer_oof[0].max(axis=1)
    for cls in CLASSES:
        q = np.percentile(mx[y == cls], [10, 50, 90])
        print(f"max side-probability for true {cls:7s}: p10={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f}")
    run.record(incumbent["name"], incumbent["scores"], per_class=incumbent["per_class"],
               confusion=incumbent["confusion_first_partition"], wrong_side=incumbent["wrong_side_first_partition"],
               decision=INCUMBENT)
    run.save_json("incumbent_folds.json", incumbent["folds"])
    results = {incumbent["name"]: incumbent}

    # ---- challengers: parameters selected on inner blocks only
    for rule in RULES:
        name = f"{rule}_selected_on_inner_blocks"
        res = evaluate_rule(name, protocol, probs, y, lambda fold, yl, pi, rule=rule: select(rule, yl, pi)[0])
        results[name] = res
        run.record(name, res["scores"], per_class=res["per_class"], confusion=res["confusion_first_partition"],
                   wrong_side=res["wrong_side_first_partition"],
                   selected=[f["decision"] for f in res["folds"]])
        run.save_json(f"{name}_folds.json", res["folds"])
    np.savez_compressed(run.path / "block_predictions.npz", labels=y.astype("U7"),
                        **{name: r["prediction"] for name, r in results.items()})

    best_name = run.best_name
    print(f"\nblock-CV winner: {best_name}  ({run.tracker.best_mean:.4f})")
    summary_extra = dict(incumbent_block_cv=incumbent["mean"], winner=best_name, protocol=dict(
        blocks=args.blocks, inner_blocks=args.inner_blocks, offsets=list(offsets)))
    if best_name != incumbent["name"]:
        # Final decision parameters: select on out-of-fold probabilities pooled over ALL outer blocks
        # (every file predicted by a model that never saw it), for the unchanged shipped detector.
        rule = best_name.split("_selected")[0]
        decision, pooled_score = select(rule, y, outer_oof[0])
        from common.artifacts import load_rail_model, resolve_model
        art = load_rail_model(resolve_model(MODEL, ("rail_model.joblib",)))
        candidate = dict(art)
        candidate["decision"] = decision
        candidate["decision_source"] = dict(experiment=run.run_id, rule=rule, selected_on="pooled outer-block OOF",
                                            pooled_block_oof_macro_f1=pooled_score)
        joblib.dump(candidate, run.path / "rail_model.joblib")
        summary_extra.update(final_decision=decision, final_decision_pooled_oof_macro_f1=pooled_score,
                             candidate_artifact="rail_model.joblib (same detector, new decision)")
        print(f"final decision for the unchanged detector: {decision} (pooled block-OOF macro-F1 {pooled_score:.4f})")
    run.finish(published=False, **summary_extra)


if __name__ == "__main__":
    main()
