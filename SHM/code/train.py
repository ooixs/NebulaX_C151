"""SHM calibration: grid over (m, residue, magnitude, cutoff, Goodman) minimising MAPE, 8-fold CV × 3 seeds."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SHM.code.pipeline import DamageModel, cycles, damage_sum, fit_C, load_series, mape  # noqa: E402

DATA = ROOT / "data/SHM"
WEIGHTS = ROOT / "weights/SHM"
MODEL = ROOT / "SHM/model"

M_GRID = np.round(np.arange(3.0, 7.01, 0.1), 2)
RESIDUE = ["half", "full", "drop"]
MAGNITUDE = ["amplitude"]  # range differs from amplitude only by a constant factor absorbed in C
CUTOFF = [0.0, 0.5, 1.0, 2.0]
GOODMAN = [None, 100.0, 200.0, 400.0]


def config_grid():
    for m, r, mag, cut, su in itertools.product(M_GRID, RESIDUE, MAGNITUDE, CUTOFF, GOODMAN):
        yield dict(m=float(m), residue=r, magnitude=mag, cutoff=cut, goodman_su=su)


def S_matrix(cyc_list, cfgs):
    """(n_files, n_cfgs) matrix of S_m values."""
    def col(cfg):
        return np.array([damage_sum(c, cfg["m"], cfg["residue"], cfg["magnitude"], cfg["goodman_su"], cfg["cutoff"]) for c in cyc_list])
    cols = Parallel(n_jobs=-1)(delayed(col)(cfg) for cfg in cfgs)
    return np.stack(cols, axis=1)


def complexity(cfg) -> int:
    return int(cfg["residue"] != "half") + int(cfg["cutoff"] > 0) + int(cfg["goodman_su"] is not None)


def select_config(S, D, indices, configs, tolerance=0.0002):
    scales = np.array([fit_C(S[indices, j], D[indices]) for j in range(S.shape[1])])
    errors = np.mean(np.abs(1 - S[indices] / (D[indices, None] * scales)), axis=0)
    eligible = np.flatnonzero(errors <= errors.min() + tolerance)
    def key(j):
        cfg = configs[j]
        extra = int(cfg.get("range_bins") is not None or cfg.get("range_bin_width") is not None)
        extra += int(cfg.get("bin_mode", "ceil") != "ceil")
        return complexity(cfg) + extra, abs(cfg["m"] - 5.0), errors[j], j
    j = min(eligible, key=key)
    return int(j), float(scales[j])


def research(args):
    from common.metrics import shm_score
    from common.research import ResearchRun, fingerprint

    run = ResearchRun("SHM", args.run_id, args.min_delta, args.patience)
    lab = pd.read_csv(DATA / "Train_Labels.csv")
    D = lab.damage.to_numpy()
    cache = WEIGHTS / "train_cycles.joblib"
    cyc = joblib.load(cache)
    if len(cyc) != len(D):
        raise ValueError("cycle cache does not match the labelled files")
    checked = sorted({0, len(D) // 2, len(D) - 1})
    for i in checked:
        actual = cycles(load_series(DATA / "Train" / lab.filename.iloc[i]))
        if actual.shape != cyc[i].shape or not np.allclose(actual, cyc[i], rtol=1e-12, atol=1e-12):
            raise ValueError("stale cycle cache; rebuild before research")
    run.save_json("inputs.json", dict(labels_sha256=fingerprint(DATA / "Train_Labels.csv"),
                  cycles_sha256=fingerprint(cache), checked_files=lab.filename.iloc[checked].tolist(),
                  files=lab.filename.tolist(), training_only=True))
    base = DamageModel(m=5.0, C=1.0).to_dict()
    base.pop("C")
    key = lambda cfg: json.dumps(cfg, sort_keys=True)
    columns = {}
    old_configs = list(config_grid())
    old_S = np.load(WEIGHTS / "S_matrix.npy", allow_pickle=False)
    if old_S.shape != (len(D), len(old_configs)):
        raise ValueError("S-matrix cache does not match the configuration grid")
    for j, cfg in enumerate(old_configs):
        columns[key({**base, **cfg})] = old_S[:, j]
    s5 = np.array([damage_sum(c, 5.0) for c in cyc])
    if not np.allclose(s5, columns[key(base)], rtol=1e-12):
        raise ValueError("stale S-matrix cache; rebuild before research")
    splits = [(seed, tr, te) for seed in range(args.repeats)
              for tr, te in KFold(args.folds, shuffle=True, random_state=seed).split(D)]
    def legacy_C(S, y):
        ratio = S / y
        order = np.argsort(ratio)
        cumulative = np.cumsum((1 / y)[order])
        return float(ratio[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    legacy = np.zeros((args.repeats, len(D)))
    for seed, tr, te in splits:
        legacy[seed, te] = s5[te] / legacy_C(s5[tr], D[tr])
    legacy_scores = [shm_score(D, p)["score"] for p in legacy]
    run.save_json("legacy_baseline.json", dict(scores=legacy_scores, mean=float(np.mean(legacy_scores))))
    print(f"SHM legacy calibration: {np.mean(legacy_scores):.6f}; verified {len(checked)} raw-series caches", flush=True)
    bins = [None, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    families = [
        ("corrected_m5", [base]),
        ("upper_edge_range_bins", [{**base, "range_bins": b} for b in bins]),
        ("fine_range_bin_count", [{**base, "range_bins": b} for b in range(4, 257, 4)]),
        ("alternative_bin_centres", [{**base, "range_bins": b, "bin_mode": mode}
                                      for b in bins[1:] for mode in ("ceil", "midpoint", "nearest")]),
        ("fine_SN_exponent", [{**base, "m": float(m), "range_bins": b}
                              for b in (None, 16, 32, 64, 128) for m in np.round(np.arange(4.90, 5.101, 0.01), 2)]),
        ("residue_conventions", [{**base, "residue": residue, "range_bins": b}
                                 for b in bins for residue in ("half", "full", "drop")]),
        ("mean_stress_correction", [{**base, "goodman_su": su, "range_bins": b}
                                   for b in (None, 16, 32, 64, 128) for su in (None, 60.0, 100.0, 200.0, 400.0, 800.0)]),
        ("endurance_cutoff", [{**base, "cutoff": cutoff, "range_bins": b}
                              for b in bins for cutoff in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)]),
        ("fixed_range_bin_width", [{**base, "range_bin_width": width, "bin_mode": mode}
                                   for width in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0) for mode in ("ceil", "midpoint", "nearest")]),
        ("broad_physics_grid", [{**base, **cfg} for cfg in old_configs]),
    ]
    all_configs = {key(cfg): cfg for _, configs in families for cfg in configs}
    families += [(f"joint_grid_tolerance_{tolerance:g}", list(all_configs.values()))
                 for tolerance in (0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005)]
    best = None
    for trial, (name, configs) in enumerate(families):
        tolerance = float(name.rsplit("_", 1)[1]) if name.startswith("joint_grid_tolerance_") else 0.0002
        missing = [cfg for cfg in configs if key(cfg) not in columns]
        def column(cfg):
            return key(cfg), np.array([damage_sum(c, **cfg) for c in cyc])
        columns.update(Parallel(n_jobs=args.jobs, prefer="threads")(delayed(column)(cfg) for cfg in missing))
        S = np.column_stack([columns[key(cfg)] for cfg in configs])
        oof = np.zeros((args.repeats, len(D)))
        fold_reports = []
        for seed, tr, te in splits:
            j, C = select_config(S, D, tr, configs, tolerance)
            oof[seed, te] = S[te, j] / C
            fold_reports.append(dict(seed=seed, validation_files=lab.filename.iloc[te].tolist(),
                                     config=configs[j], C=C, score=shm_score(D[te], oof[seed, te])["score"]))
        scores = [shm_score(D, p)["score"] for p in oof]
        result = run.record(name, scores, configurations=len(configs), folds=fold_reports,
                            fit_tolerance=tolerance, worst_file_ape=float(np.max(np.abs(1 - oof / D))))
        np.savez_compressed(run.path / f"trial_{trial:02d}.npz", oof=oof, S=S, target=D)
        run.save_json(f"configs_{trial:02d}.json", configs)
        if result["accepted"]:
            j, C = select_config(S, D, np.arange(len(D)), configs, tolerance)
            best = dict(config=configs[j], C=C, oof=oof, scores=scores, folds=fold_reports)
        if run.tracker.stop_reason:
            break
    if run.tracker.stop_reason is None:
        run.finish(published=False)
        raise RuntimeError("SHM candidate list exhausted before convergence; extend the search")
    model = DamageModel(C=best["C"], **best["config"])
    run.save_json("shm_model.json", model.to_dict())
    errors = np.abs(1 - best["oof"] / D)
    run.save_json("per_file_errors.json", dict(zip(lab.filename, errors.mean(axis=0).tolist())))
    if args.ship:
        run.publish(run.path / "shm_model.json", MODEL / "shm_model.json")
    run.finish(published=args.ship, model="shm_model.json", final=model.to_dict(),
               legacy_cv_mean=float(np.mean(legacy_scores)), validation="file-level CV; all convention and scale selection inside training folds",
               p50_ape=float(np.median(errors)), p90_ape=float(np.quantile(errors, 0.9)), worst_ape=float(errors.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--ship", action="store_true")
    args = ap.parse_args()
    if args.research:
        research(args)
        return
    WEIGHTS.mkdir(parents=True, exist_ok=True); MODEL.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(DATA / "Train_Labels.csv")
    D = lab.damage.to_numpy()
    cache = WEIGHTS / "train_cycles.joblib"
    if cache.exists():
        cyc = joblib.load(cache)
    else:
        cyc = Parallel(n_jobs=-1)(delayed(lambda f: cycles(load_series(DATA / "Train" / f)))(f) for f in lab.filename)
        joblib.dump(cyc, cache)
    cfgs = list(config_grid())
    print(f"{len(cfgs)} configurations; computing S matrix ...")
    S = S_matrix(cyc, cfgs)
    np.save(WEIGHTS / "S_matrix.npy", S)

    # ---- CV: for each fold, choose (cfg, C) on train files by MAPE (simplest within 0.002 of best), score held-out
    cv_scores, chosen = [], []
    for seed in range(3):
        oof = np.zeros(len(D))
        for tr, te in KFold(8, shuffle=True, random_state=seed).split(D):
            best = None
            for j, cfg in enumerate(cfgs):
                C = fit_C(S[tr, j], D[tr])
                e = mape(D[tr], S[tr, j] / C)
                key = (e, complexity(cfg), abs(cfg["m"] - 5))
                if best is None or key < best[0]:
                    best = (key, j, C)
            # prefer simplest config within tolerance of the best training MAPE
            tol = best[0][0] + 0.002
            cands = [(complexity(cfg), abs(cfg["m"] - 5), j) for j, cfg in enumerate(cfgs)
                     if mape(D[tr], S[tr, j] / fit_C(S[tr, j], D[tr])) <= tol]
            _, _, j = min(cands)
            C = fit_C(S[tr, j], D[tr])
            oof[te] = S[te, j] / C
            chosen.append(dict(seed=seed, **cfgs[j], C=C))
        cv_scores.append(1 - mape(D, oof))
        print(f"seed {seed}: CV score (1-MAPE) = {cv_scores[-1]:.4f}")
    print(f"CV score mean±std = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    ch = pd.DataFrame(chosen)
    print("chosen configs across folds:\n", ch.groupby(["m", "residue", "cutoff", "goodman_su"], dropna=False).size().sort_values(ascending=False).head(8))

    # ---- baselines for the write-up: plain m=5 half-cycle, and log-LS fitted C
    j5 = next(j for j, c in enumerate(cfgs) if c["m"] == 5.0 and c["residue"] == "half" and c["cutoff"] == 0 and c["goodman_su"] is None)
    C_ls = float(np.exp(np.mean(np.log(S[:, j5]) - np.log(D))))
    print(f"m=5 half/no-cutoff: in-sample MAPE with MAPE-optimal C = {mape(D, S[:, j5] / fit_C(S[:, j5], D)):.4f}, with log-LS C = {mape(D, S[:, j5] / C_ls):.4f}")

    # ---- final: choose on all 64 with the same rule
    errs = np.array([mape(D, S[:, j] / fit_C(S[:, j], D)) for j in range(len(cfgs))])
    tol = errs.min() + 0.002
    j = min(((complexity(cfg), abs(cfg["m"] - 5), jj) for jj, cfg in enumerate(cfgs) if errs[jj] <= tol))[2]
    final = DamageModel(**{**cfgs[j], "C": fit_C(S[:, j], D)})
    resid = D / (S[:, j] / final.C)
    print("FINAL", final.to_dict(), f"in-sample MAPE={errs[j]:.4f}", f"worst-file APE={np.max(np.abs(1 - 1/resid)):.3f}")
    top = np.argsort(errs)[:10]
    print("top-10 configs by in-sample MAPE:")
    for jj in top:
        print(f"   {cfgs[jj]} MAPE={errs[jj]:.4f}")
    report = dict(cv_scores=cv_scores, cv_mean=float(np.mean(cv_scores)), cv_std=float(np.std(cv_scores)),
                  chosen_per_fold=chosen, final=final.to_dict(), final_insample_mape=float(errs[j]),
                  m5_insample_mape_mapeC=float(mape(D, S[:, j5] / fit_C(S[:, j5], D))),
                  m5_insample_mape_logC=float(mape(D, S[:, j5] / C_ls)),
                  per_file_ape=dict(zip(lab.filename, (np.abs(D - S[:, j] / final.C) / D).tolist())))
    (WEIGHTS / "cv_report.json").write_text(json.dumps(report, indent=1, default=float))
    (MODEL / "shm_model.json").write_text(json.dumps(final.to_dict(), indent=1))
    joblib.dump(final.to_dict(), WEIGHTS / "shm_model.joblib")
    print("saved", MODEL / "shm_model.json")


if __name__ == "__main__":
    main()
