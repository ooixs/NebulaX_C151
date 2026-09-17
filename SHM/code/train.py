"""SHM calibration: grid over (m, residue, magnitude, cutoff, Goodman) minimising MAPE, 8-fold CV × 3 seeds."""
from __future__ import annotations

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


def main():
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
