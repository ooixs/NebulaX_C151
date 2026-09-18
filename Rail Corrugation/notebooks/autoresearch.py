"""Rail improvement phase: candidate per-side detectors on the same repeated stratified folds."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rail_corrugation.pipeline import CLASSES, NormalReference, aggregate, side_relative_rows  # noqa: E402
from rail_corrugation.train import decode_side  # noqa: E402
from common.metrics import rail_macro_f1  # noqa: E402

W = ROOT / "weights/Rail Corrugation"
lab = pd.read_csv(ROOT / "data/Rail_Corrugation/Train_Labels.csv"); y = lab.label.to_numpy(); n = len(y)
ct_v = joblib.load(W / "train_channel_tables.joblib"); CT, V = [c for c, _ in ct_v], [v for _, v in ct_v]
SPLITS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(n), y))


def fold_features(tr, ref_cls=NormalReference):
    ref = ref_cls().fit([CT[i] for i in tr], [V[i] for i in tr], [y[i] for i in tr])
    return pd.DataFrame([aggregate(ref.excess(CT[i], V[i]), V[i]) for i in range(n)])


FEATS_CACHE = {}


def features_for(tr):
    key = tuple(tr)
    if key not in FEATS_CACHE:
        FEATS_CACHE[key] = fold_features(tr)
    return FEATS_CACHE[key]


def side_rows(F, idx, mirror=False):
    rows, ys = [], []
    for i in idx:
        r1, r2 = side_relative_rows(F.iloc[i].to_dict())
        rows += [r1, r2]; ys += [int(y[i] == "Side I"), int(y[i] == "Side II")]
    R = pd.DataFrame(rows).fillna(-9)
    return R, np.array(ys)


def run(name, make_clf, cols_fn=lambda R: list(R.columns), mirror=False, decode="max"):
    oof = np.zeros((3, n, 2))
    for k, (tr, te) in enumerate(SPLITS):
        F = features_for(tr)
        Rtr, ytr = side_rows(F, tr)
        cols = cols_fn(Rtr)
        clf = make_clf(k).fit(Rtr[cols], ytr)
        Rte, _ = side_rows(F, te)
        p = clf.predict_proba(Rte[cols])[:, 1]
        oof[k // 5][te] = p.reshape(-1, 2)
    scores, taus = [], []
    for rep in range(3):
        best = max(((rail_macro_f1(y, decode_side(oof[rep][:, 0], oof[rep][:, 1], t))["macro_f1"], t) for t in np.linspace(0.1, 0.9, 33)))
        scores.append(best[0]); taus.append(best[1])
    pm = oof.mean(0); tau = float(np.mean(taus)); pred = decode_side(pm[:, 0], pm[:, 1], tau)
    pc = rail_macro_f1(y, pred)["per_class"]
    print(f"{name:52s} macroF1={np.mean(scores):.4f}±{np.std(scores):.4f}  tau={tau:.2f}  "
          f"per-class N/I/II={pc['Normal']:.3f}/{pc['Side I']:.3f}/{pc['Side II']:.3f}", flush=True)
    return float(np.mean(scores)), float(np.std(scores))


rf = lambda k: RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=k, n_jobs=-1)
results = {}
results["baseline RF per-side"] = run("0. baseline RF per-side detector", rf)
results["ET"] = run("1. ExtraTrees per-side", lambda k: ExtraTreesClassifier(n_estimators=800, class_weight="balanced_subsample", random_state=k, n_jobs=-1))


def lgb(k):
    import lightgbm as lgbm
    return lgbm.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", random_state=k, verbose=-1)


results["LGBM"] = run("2. LightGBM per-side", lgb)
results["rel-only RF"] = run("3. RF on side-RELATIVE features only", rf, cols_fn=lambda R: [c for c in R.columns if c.startswith("rel_") or c.startswith("speed")])
results["own-only RF"] = run("4. RF on OWN-side features only (no cross-rail)", rf, cols_fn=lambda R: [c for c in R.columns if c.startswith("own_") or c.startswith("speed")])
results["LR"] = run("5. logistic regression (relative + own, standardised)", lambda k: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)))
results["RF deeper leaf"] = run("6. RF min_samples_leaf=3, max_features=0.3", lambda k: RandomForestClassifier(n_estimators=800, class_weight="balanced_subsample", min_samples_leaf=3, max_features=0.3, random_state=k, n_jobs=-1))
# 7. excess-only features (reference-normalised) + speed
results["excess-only RF"] = run("7. RF on excess-over-reference features only", rf, cols_fn=lambda R: [c for c in R.columns if "_ex_" in c or c.startswith("rel_ex_") or c.startswith("speed")])
# 8. RF + LGBM probability average
class Avg:
    def __init__(self, k): self.a, self.b = rf(k), lgb(k)
    def fit(self, X, y): self.a.fit(X, y); self.b.fit(X, y); return self
    def predict_proba(self, X): return 0.5 * (self.a.predict_proba(X) + self.b.predict_proba(X))
results["RF+LGBM avg"] = run("8. RF + LightGBM probability average", Avg)
joblib.dump(results, W / "autoresearch_results.joblib")
print("\nsummary:"); [print(f"  {k:30s} {m:.4f} ± {s:.4f}") for k, (m, s) in results.items()]
