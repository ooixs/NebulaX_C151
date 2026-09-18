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
from SHM.code.pipeline import (DamageModel, cycles, damage_sum, fit_C, load_series, mape,  # noqa: E402
                               sg_series_features)

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


def grouped_C_predict(S, D, groups, tr, te):
    """Per-group MAPE-optimal C, fitted on training indices only; groups with <3 training files fall back to the pooled C."""
    fallback = fit_C(S[tr], D[tr])
    out = np.empty(len(te), dtype=float)
    for value in np.unique(groups[te]):
        trg = tr[groups[tr] == value]
        C = fit_C(S[trg], D[trg]) if len(trg) >= 3 else fallback
        out[groups[te] == value] = S[te][groups[te] == value] / C
    return out


def sg_feature_table(cyc, lab, jobs):
    """Per-file features in the spirit of the sg-experiments branch, computed with the validated rainflow."""
    def one(name, c):
        return sg_series_features(load_series(DATA / "Train" / name), c)
    return pd.DataFrame(Parallel(n_jobs=jobs)(delayed(one)(name, c) for name, c in zip(lab.filename, cyc)))


SG_CORE_FEATURES = ["log_rf_energy_range_m_4.25", "log_rf_energy_range_m_3.5", "log_rf_energy_range_m_3.0",
                    "log_rf_goodman_su_600_m_3.5", "p95_p05", "p99_p01", "std", "ptp", "rf_p95_range",
                    "p01", "crest_factor", "spec_alpha2", "spec_zero_crossing"]


def sg_families(args, D, s5, T):
    """Candidate families derived from the sg-experiments branch, evaluated with fold-local fitting."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.preprocessing import StandardScaler

    X_all = T.to_numpy(float)
    X_core = T[SG_CORE_FEATURES].to_numpy(float)
    log_D = np.log(D)

    def ensemble_fn(tr, te, seed):
        scaler = StandardScaler().fit(X_core[tr])
        pred = np.zeros(len(te))
        for weight, model in ((0.5521, HuberRegressor(alpha=0.01, max_iter=20000)),
                              (0.4479, ExtraTreesRegressor(n_estimators=150, max_depth=5, random_state=42))):
            model.fit(scaler.transform(X_core[tr]), log_D[tr])
            pred += weight * np.exp(model.predict(scaler.transform(X_core[te])))
        return np.clip(pred, 1e-3, 1.5)

    def residual_fn(make_model, columns=None):
        X = X_all if columns is None else T[list(columns)].to_numpy(float)
        def fn(tr, te, seed):
            C = fit_C(s5[tr], D[tr])
            base = s5 / C
            scaler = StandardScaler().fit(X[tr])
            model = make_model(seed).fit(scaler.transform(X[tr]), np.log(D[tr] / base[tr]))
            return base[te] * np.exp(model.predict(scaler.transform(X[te])))
        return fn

    def residual_final(make_model, columns=None):
        def final():
            X = X_all if columns is None else T[list(columns)].to_numpy(float)
            C = fit_C(s5, D)
            scaler = StandardScaler().fit(X)
            model = make_model(0).fit(scaler.transform(X), np.log(D / (s5 / C)))
            return dict(kind="physics_plus_residual", base=DamageModel(m=5.0, C=C).to_dict(),
                        feature_names=list(T.columns) if columns is None else list(columns),
                        scaler=scaler, model=model)
        return final

    def group_fn(groups):
        return lambda tr, te, seed: grouped_C_predict(s5, D, groups, tr, te)

    def make_lgbm(seed):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                             colsample_bytree=0.7, reg_lambda=1.0, random_state=seed, verbosity=-1)

    class HuberEtAverage:
        def __init__(self, seed):
            self.a = HuberRegressor(alpha=0.01, max_iter=20000)
            self.b = ExtraTreesRegressor(n_estimators=150, max_depth=5, random_state=seed)

        def fit(self, X, y):
            self.a.fit(X, y); self.b.fit(X, y); return self

        def predict(self, X):
            return 0.5 * (self.a.predict(X) + self.b.predict(X))

    make_huber = lambda alpha: (lambda seed: HuberRegressor(alpha=alpha, max_iter=20000))
    aw_groups = (T["mean"].to_numpy() > 0).astype(int)
    skew_groups = (T["skewness"].to_numpy() > 0).astype(int)
    return [
        ("sg_huber_extratrees_ensemble", dict(fn=ensemble_fn, description="SG app ensemble replicated on validated rainflow features")),
        ("aw_proxy_grouped_C", dict(fn=group_fn(aw_groups), description="per-load-condition C (SG is_aw4_proxy: series mean > 0)",
                                    groups=aw_groups.tolist())),
        ("skew_sign_grouped_C", dict(fn=group_fn(skew_groups), description="per-skew-sign C (strongest residual correlate)",
                                     groups=skew_groups.tolist())),
        ("physics_plus_huber_residual", dict(fn=residual_fn(make_huber(0.01)), final=residual_final(make_huber(0.01)),
                                             description="S5/C plus Huber log-residual on SG features")),
        ("physics_plus_extratrees_residual", dict(fn=residual_fn(lambda seed: ExtraTreesRegressor(n_estimators=150, max_depth=5, random_state=seed)),
                                                  final=residual_final(lambda seed: ExtraTreesRegressor(n_estimators=150, max_depth=5, random_state=seed)),
                                                  description="S5/C plus shallow ExtraTrees log-residual on SG features")),
        ("physics_plus_ridge_residual", dict(fn=residual_fn(lambda seed: Ridge(alpha=1.0)), final=residual_final(lambda seed: Ridge(alpha=1.0)),
                                             description="same features, squared loss: does the robust loss matter?")),
        ("physics_plus_huber_core13", dict(fn=residual_fn(make_huber(0.01), columns=SG_CORE_FEATURES),
                                           final=residual_final(make_huber(0.01), columns=SG_CORE_FEATURES),
                                           description="Huber log-residual on the 13 SG core features only")),
        ("physics_plus_lgbm_residual", dict(fn=residual_fn(make_lgbm), final=residual_final(make_lgbm),
                                            description="S5/C plus LightGBM log-residual on SG features")),
        ("physics_plus_huber_alpha1", dict(fn=residual_fn(make_huber(1.0)), final=residual_final(make_huber(1.0)),
                                           description="strongly regularised Huber log-residual")),
        ("physics_plus_huber_et_average", dict(fn=residual_fn(lambda seed: HuberEtAverage(seed)),
                                               final=residual_final(lambda seed: HuberEtAverage(seed)),
                                               description="average of Huber and ExtraTrees log-residuals")),
        ("physics_plus_huber_alpha3", dict(fn=residual_fn(make_huber(3.0)), final=residual_final(make_huber(3.0)),
                                           description="Huber log-residual, alpha=3")),
        ("physics_plus_huber_alpha10", dict(fn=residual_fn(make_huber(10.0)), final=residual_final(make_huber(10.0)),
                                            description="Huber log-residual, alpha=10")),
        ("physics_plus_huber_alpha03", dict(fn=residual_fn(make_huber(0.3)), final=residual_final(make_huber(0.3)),
                                            description="Huber log-residual, alpha=0.3")),
        ("physics_plus_ridge_alpha10", dict(fn=residual_fn(lambda seed: Ridge(alpha=10.0)), final=residual_final(lambda seed: Ridge(alpha=10.0)),
                                            description="Ridge log-residual, alpha=10")),
        ("physics_plus_ridge_alpha30", dict(fn=residual_fn(lambda seed: Ridge(alpha=30.0)), final=residual_final(lambda seed: Ridge(alpha=30.0)),
                                            description="Ridge log-residual, alpha=30")),
        ("physics_plus_huber_alpha1_core13", dict(fn=residual_fn(make_huber(1.0), columns=SG_CORE_FEATURES),
                                                  final=residual_final(make_huber(1.0), columns=SG_CORE_FEATURES),
                                                  description="regularised Huber on the 13 SG core features")),
        ("physics_plus_huber_alpha1_eps2", dict(fn=residual_fn(lambda seed: HuberRegressor(alpha=1.0, epsilon=2.0, max_iter=20000)),
                                                final=residual_final(lambda seed: HuberRegressor(alpha=1.0, epsilon=2.0, max_iter=20000)),
                                                description="regularised Huber with a wider quadratic zone")),
    ]


def research(args):
    from common.metrics import shm_score
    from common.research import ResearchRun, fingerprint

    run = ResearchRun("SHM", args.run_id, args.min_delta, args.patience)
    lab = pd.read_csv(DATA / "Train_Labels.csv")
    D = lab.damage.to_numpy()
    cache = WEIGHTS / "train_cycles.joblib"
    if cache.exists():
        cyc = joblib.load(cache)
    else:
        cyc = Parallel(n_jobs=args.jobs)(delayed(lambda f: cycles(load_series(DATA / "Train" / f)))(f) for f in lab.filename)
        joblib.dump(cyc, cache)
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
    if (WEIGHTS / "S_matrix.npy").exists():
        old_S = np.load(WEIGHTS / "S_matrix.npy", allow_pickle=False)
        if old_S.shape != (len(D), len(old_configs)):
            raise ValueError("S-matrix cache does not match the configuration grid")
        for j, cfg in enumerate(old_configs):
            columns[key({**base, **cfg})] = old_S[:, j]
    s5 = np.array([damage_sum(c, 5.0) for c in cyc])
    if key(base) in columns and not np.allclose(s5, columns[key(base)], rtol=1e-12):
        raise ValueError("stale S-matrix cache; rebuild before research")
    columns.setdefault(key(base), s5)
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
    if getattr(args, "families", "default") == "sg":
        T = sg_feature_table(cyc, lab, args.jobs)
        T.to_csv(run.path / "sg_features.csv", index=False)
        families = [("corrected_m5", [base])] + sg_families(args, D, s5, T)
        print(f"SG-branch evaluation: {len(T.columns)} features per file for the custom candidates", flush=True)
    else:
        all_configs = {key(cfg): cfg for _, configs in families for cfg in configs}
        families += [(f"joint_grid_tolerance_{tolerance:g}", list(all_configs.values()))
                     for tolerance in (0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005)]
    best = None
    for trial, (name, payload) in enumerate(families):
        oof = np.zeros((args.repeats, len(D)))
        fold_reports = []
        if isinstance(payload, list):
            configs = payload
            tolerance = float(name.rsplit("_", 1)[1]) if name.startswith("joint_grid_tolerance_") else 0.0002
            missing = [cfg for cfg in configs if key(cfg) not in columns]
            def column(cfg):
                return key(cfg), np.array([damage_sum(c, **cfg) for c in cyc])
            columns.update(Parallel(n_jobs=args.jobs, prefer="threads")(delayed(column)(cfg) for cfg in missing))
            S = np.column_stack([columns[key(cfg)] for cfg in configs])
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
                best = dict(kind="grid", config=configs[j], C=C, oof=oof, scores=scores, folds=fold_reports)
        else:
            for seed, tr, te in splits:
                oof[seed, te] = payload["fn"](tr, te, seed)
                fold_reports.append(dict(seed=seed, validation_files=lab.filename.iloc[te].tolist(),
                                         score=shm_score(D[te], oof[seed, te])["score"]))
            scores = [shm_score(D, p)["score"] for p in oof]
            result = run.record(name, scores, kind="custom", description=payload["description"], folds=fold_reports,
                                worst_file_ape=float(np.max(np.abs(1 - oof / D))))
            np.savez_compressed(run.path / f"trial_{trial:02d}.npz", oof=oof, target=D)
            if result["accepted"]:
                best = dict(kind="custom", name=name, payload=payload, oof=oof, scores=scores, folds=fold_reports)
        if run.tracker.stop_reason:
            break
    if run.tracker.stop_reason is None:
        run.finish(published=False)
        raise RuntimeError("SHM candidate list exhausted before convergence; extend the search")
    errors = np.abs(1 - best["oof"] / D)
    run.save_json("per_file_errors.json", dict(zip(lab.filename, errors.mean(axis=0).tolist())))
    if best["kind"] == "custom":
        seeds = (100, 101, 102)
        confirm_splits = [(i, tr, te) for i, s in enumerate(seeds)
                          for tr, te in KFold(args.folds, shuffle=True, random_state=s).split(D)]
        base_oof = np.zeros((len(seeds), len(D)))
        champion_oof = np.zeros((len(seeds), len(D)))
        for i, tr, te in confirm_splits:
            base_oof[i, te] = s5[te] / fit_C(s5[tr], D[tr])
            champion_oof[i, te] = best["payload"]["fn"](tr, te, 100 + i)
        base_scores = [shm_score(D, p)["score"] for p in base_oof]
        champion_scores = [shm_score(D, p)["score"] for p in champion_oof]
        gain = float(np.mean(champion_scores) - np.mean(base_scores))
        required = max(args.min_delta, float(np.std(base_scores)))
        confirmed = bool(all(c > b for c, b in zip(champion_scores, base_scores)) and gain > required)
        confirmation = dict(name=best["name"], seeds=list(seeds), baseline_scores=base_scores,
                            champion_scores=champion_scores, gain=gain, required_gain=required,
                            accepted=confirmed, description=best["payload"]["description"])
        run.save_json("confirmation.json", confirmation)
        print(f"confirmation on fresh seeds: champion {np.mean(champion_scores):.6f} vs baseline "
              f"{np.mean(base_scores):.6f} -> {'CONFIRMED' if confirmed else 'REJECTED'}", flush=True)
        if not confirmed or "final" not in best["payload"]:
            run.finish(published=False, model=None, champion=best["name"], champion_confirmed=confirmed,
                       fallback="incumbent shipped model retained",
                       legacy_cv_mean=float(np.mean(legacy_scores)),
                       validation="file-level CV; all convention and scale selection inside training folds",
                       p50_ape=float(np.median(errors)), p90_ape=float(np.quantile(errors, 0.9)), worst_ape=float(errors.max()))
            return
        artifact = best["payload"]["final"]()
        artifact.update(research_run=run.run_id, cv_scores=best["scores"], confirmation=confirmation,
                        validation="file-level CV with fold-local scaler/model/C fitting; fresh-seed confirmation")
        joblib.dump(artifact, run.path / "shm_model.joblib")
        if args.ship:
            run.publish(run.path / "shm_model.joblib", MODEL / "shm_model.joblib")
        run.finish(published=args.ship, model="shm_model.joblib", champion=best["name"], champion_confirmed=True,
                   confirmation_gain=gain, legacy_cv_mean=float(np.mean(legacy_scores)),
                   validation="file-level CV; fold-local scaler/model/C fitting; fresh-seed confirmation",
                   p50_ape=float(np.median(errors)), p90_ape=float(np.quantile(errors, 0.9)), worst_ape=float(errors.max()))
        return
    model = DamageModel(C=best["C"], **best["config"])
    run.save_json("shm_model.json", model.to_dict())
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
    ap.add_argument("--families", default="default", choices=["default", "sg"])
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
    from common.artifacts import publish_model
    (WEIGHTS / "shm_model.json").write_text(json.dumps(final.to_dict(), indent=1))
    joblib.dump(final.to_dict(), WEIGHTS / "shm_model.joblib")
    publish_model(WEIGHTS / "shm_model.json", MODEL / "shm_model.json")
    print("saved", MODEL / "shm_model.json")


if __name__ == "__main__":
    main()
