"""ACV: leave-one-case-out evaluation of scoring configurations; saves the chosen config + response model."""
from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ACV.code.pipeline import ResponseModel, car_features, load_case, prepare, score_cars  # noqa: E402
from common.metrics import acv_rank_score  # noqa: E402

DATA = ROOT / "data/ACV"
WEIGHTS = ROOT / "weights/ACV"
MODEL = ROOT / "ACV/model"

COMPONENTS = ["peer_dev_mean", "set_err_p95", "peer_dev_persist", "peer_dev_pos_frac", "resid_p90", "p_high_def"]
GRID = [0.0, 0.5, 1.0]


def main():
    WEIGHTS.mkdir(parents=True, exist_ok=True); MODEL.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(DATA / "Train_Labels.csv", dtype=str).set_index("filename")["faulty_car"]
    cases = {}
    for f in sorted(glob.glob(str(DATA / "Train/*.xlsx"))):
        long, cars, roles = load_case(f)
        cases[Path(f).name] = dict(long=long, cars=cars, roles=roles)
    names = list(cases)

    # ---- LOO: response model fitted on healthy cars of the 5 other cases, features on the held-out case
    feats_loo = {}
    for held in names:
        healthy = [prepare(c["long"])[lambda d: d["car"] != labels[n]] for n, c in cases.items() if n != held]
        rm = ResponseModel().fit(healthy)
        feats_loo[held] = car_features(cases[held]["long"], cases[held]["cars"], rm)

    # ---- baseline: single component peer_dev_mean
    def evaluate(weights, feats):
        return {n: acv_rank_score(score_cars(feats[n], weights).index.tolist(), labels[n]) for n in names}

    base = evaluate({"peer_dev_mean": 1.0}, feats_loo)
    print("baseline peer_dev_mean:", np.mean(list(base.values())).round(4), base)

    # ---- weight grid, chosen by nested LOO (weights picked on 5 cases, scored on the 6th)
    combos = [dict(zip(COMPONENTS, w)) for w in itertools.product(GRID, repeat=len(COMPONENTS)) if any(w)]
    table = {tuple(c.values()): evaluate(c, feats_loo) for c in combos}
    nested = {}
    for held in names:
        best = max(table, key=lambda k: (np.mean([table[k][n] for n in names if n != held]), -sum(1 for v in k if v)))
        nested[held] = table[best][held]
    print("nested-LOO weighted score:", np.mean(list(nested.values())).round(4), nested)
    # in-sample best (reported for transparency only; NOT shipped unless nested-LOO beats the baseline)
    insample_key = max(table, key=lambda k: (np.mean(list(table[k].values())), -sum(1 for v in k if v)))
    insample_w = dict(zip(COMPONENTS, insample_key))
    print("in-sample best weights", insample_w, "score", np.mean(list(table[insample_key].values())).round(4),
          "(optimistic: weights chosen on the same 6 cases)")
    # acceptance rule: ship the tuned combination only if its nested-LOO mean beats the untuned baseline
    if np.mean(list(nested.values())) > np.mean(list(base.values())) + 1e-9:
        final_w, final, choice = insample_w, table[insample_key], "weighted (nested-LOO beat baseline)"
    else:
        final_w, final, choice = {"peer_dev_mean": 1.0}, base, "baseline peer_dev_mean (tuned weights did not beat it under nested LOO)"
    print("SHIPPED:", choice, final_w, "LOO score", np.mean(list(final.values())).round(4))

    # response model on all healthy cars
    rm = ResponseModel().fit([prepare(c["long"])[lambda d: d["car"] != labels[n]] for n, c in cases.items()])
    report = dict(baseline=base, baseline_mean=float(np.mean(list(base.values()))),
                  nested_loo=nested, nested_loo_mean=float(np.mean(list(nested.values()))),
                  insample_best_weights=insample_w, insample_best_mean=float(np.mean(list(table[insample_key].values()))),
                  shipped=choice, final_weights=final_w, final_scores=final, final_mean=float(np.mean(list(final.values()))),
                  ranked={n: score_cars(feats_loo[n], final_w).index.tolist() for n in names})
    (WEIGHTS / "loo_report.json").write_text(json.dumps(report, indent=1))
    joblib.dump(dict(weights=final_w, response=rm, components=COMPONENTS), WEIGHTS / "acv_scorer.joblib")
    joblib.dump(dict(weights=final_w, response=rm, components=COMPONENTS), MODEL / "acv_model.joblib")
    for n in names:
        print(f"  {n}: true={labels[n]} ranked={'|'.join(report['ranked'][n])}")
    print("saved", MODEL / "acv_model.joblib")


if __name__ == "__main__":
    main()
