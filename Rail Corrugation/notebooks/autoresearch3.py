"""Rail improvement phase 3 (Side I focus): seed averaging, per-side thresholds, window augmentation.

All on the same RepeatedStratifiedKFold(5x3, seed 0) as previous phases; ship only if
mean macro-F1 > 0.8336 + 0.0139.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rail_corrugation.pipeline import (NormalReference, aggregate, channel_features, load_file,  # noqa: E402
                           side_relative_rows, speed)
from rail_corrugation.train import decode_side, make_detector  # noqa: E402
from common.metrics import rail_macro_f1  # noqa: E402

W = ROOT / "weights/Rail Corrugation"
D = ROOT / "data/Rail_Corrugation"
lab = pd.read_csv(D / "Train_Labels.csv"); y = lab.label.to_numpy(); n = len(y)
ct_v = joblib.load(W / "train_channel_tables.joblib"); CT, V = [c for c, _ in ct_v], [v for _, v in ct_v]
SPLITS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(n), y))
BASE = (0.8336, 0.0139)


def windows_of(path, k=2):
    pulse, vib, shock = load_file(path)
    m = len(pulse) // k
    out = []
    for w in range(k):
        sl = slice(w * m, (w + 1) * m)
        v = speed(pulse[sl])
        out.append((channel_features(vib[sl], shock[sl], v), v))
    return out


wcache = W / "train_window_tables.joblib"
if wcache.exists():
    WT = joblib.load(wcache)
else:
    WT = Parallel(n_jobs=-1)(delayed(windows_of)(D / "Train" / f) for f in lab.filename)
    joblib.dump(WT, wcache)
print("window tables ready")


def decode_2tau(p1, p2, t1, t2):
    out = []
    for a, b in zip(p1, p2):
        if a < t1 and b < t2:
            out.append("Normal")
        else:
            out.append("Side I" if (a - t1) >= (b - t2) else "Side II")
    return out


def tune_2tau(oof_rep):
    grid = np.linspace(0.05, 0.9, 25)
    return max(((rail_macro_f1(y, decode_2tau(oof_rep[:, 0], oof_rep[:, 1], t1, t2))["macro_f1"], t1, t2)
                for t1 in grid for t2 in grid))


def evaluate(oof, name, two_tau=False):
    scores = []
    for rep in range(3):
        if two_tau:
            sc, *_ = tune_2tau(oof[rep])
        else:
            sc = max(rail_macro_f1(y, decode_side(oof[rep][:, 0], oof[rep][:, 1], t))["macro_f1"] for t in np.linspace(0.1, 0.9, 33))
        scores.append(sc)
    m, s = float(np.mean(scores)), float(np.std(scores))
    verdict = "SHIP?" if m > BASE[0] + BASE[1] else "no"
    print(f"{name:58s} {m:.4f}±{s:.4f}  [{verdict}]", flush=True)
    return m, s


def run_files(seeds=(0,)):
    oof = np.zeros((3, n, 2))
    for k, (tr, te) in enumerate(SPLITS):
        ref = NormalReference().fit([CT[i] for i in tr], [V[i] for i in tr], [y[i] for i in tr])
        F = pd.DataFrame([aggregate(ref.excess(CT[i], V[i]), V[i]) for i in range(n)])
        rows, ys = [], []
        for i in tr:
            r1, r2 = side_relative_rows(F.iloc[i].to_dict()); rows += [r1, r2]; ys += [int(y[i] == "Side I"), int(y[i] == "Side II")]
        R = pd.DataFrame(rows).fillna(-9)
        p = np.zeros((len(te), 2, 2))
        prob = np.zeros((n, 2))
        for s_i, seed in enumerate(seeds):
            det = make_detector(1000 * seed + k).fit(R, ys)
            for j, i in enumerate(te):
                r1, r2 = side_relative_rows(F.iloc[i].to_dict())
                prob[i] += det.predict_proba(pd.DataFrame([r1, r2]).fillna(-9)[R.columns])[:, 1] / len(seeds)
        oof[k // 5][te] = prob[te]
    return oof


def run_windows():
    oof = np.zeros((3, n, 2))
    for k, (tr, te) in enumerate(SPLITS):
        # reference fitted on WINDOW tables of training Normal files
        wct = [w[0] for i in tr for w in WT[i]]
        wv = [w[1] for i in tr for w in WT[i]]
        wy = [y[i] for i in tr for _ in WT[i]]
        ref = NormalReference().fit(wct, wv, wy)
        feats = {i: [aggregate(ref.excess(c, v), v) for c, v in WT[i]] for i in range(n)}
        rows, ys = [], []
        for i in tr:
            for fw in feats[i]:
                r1, r2 = side_relative_rows(fw); rows += [r1, r2]; ys += [int(y[i] == "Side I"), int(y[i] == "Side II")]
        R = pd.DataFrame(rows).fillna(-9)
        det = make_detector(k).fit(R, ys)
        for i in te:
            ps = []
            for fw in feats[i]:
                r1, r2 = side_relative_rows(fw)
                ps.append(det.predict_proba(pd.DataFrame([r1, r2]).fillna(-9)[R.columns])[:, 1])
            oof[k // 5][i] = np.mean(ps, axis=0)
        if (k + 1) % 5 == 0:
            print(f"  window folds {k+1}/15", flush=True)
    return oof


res = {}
oof1 = run_files(seeds=(0,))
res["14. single-seed ET (reference rerun)"] = evaluate(oof1, "14. single-seed ET (reference rerun)")
res["15. per-side thresholds"] = evaluate(oof1, "15. per-side thresholds (tau1, tau2) on same OOF", two_tau=True)
oof5 = run_files(seeds=(0, 1, 2, 3, 4))
res["16. 5-seed averaged ET"] = evaluate(oof5, "16. 5-seed averaged ET")
res["17. 5-seed + per-side thresholds"] = evaluate(oof5, "17. 5-seed averaged ET + per-side thresholds", two_tau=True)
oofw = run_windows()
res["18. window augmentation (2x0.5s)"] = evaluate(oofw, "18. window-level training + window-averaged inference")
res["19. windows + per-side thresholds"] = evaluate(oofw, "19. windows + per-side thresholds", two_tau=True)
oofc = (oof5 + oofw) / 2
res["20. (files+windows)/2 + 2tau"] = evaluate(oofc, "20. average of file- and window-level probs + per-side tau", two_tau=True)
joblib.dump(res, W / "autoresearch3_results.joblib")
