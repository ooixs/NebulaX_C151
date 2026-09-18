"""Rail improvement phase, round 2: variants around the ExtraTrees per-side detector."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Rail Corrugation/notebooks"))
import autoresearch as ar  # noqa: E402  (reuses folds, caches, run())

W = ROOT / "weights/Rail Corrugation"
et = lambda k, **kw: ExtraTreesClassifier(n_estimators=800, class_weight="balanced_subsample", random_state=k, n_jobs=-1, **kw)
res = {}
res["ET base"] = ar.run("ET (reference, 800 trees)", et)
res["ET mf0.3"] = ar.run("9. ET max_features=0.3", lambda k: et(k, max_features=0.3))
res["ET leaf2"] = ar.run("10. ET min_samples_leaf=2", lambda k: et(k, min_samples_leaf=2))
res["ET 2000"] = ar.run("11. ET 2000 trees", lambda k: ExtraTreesClassifier(n_estimators=2000, class_weight="balanced_subsample", random_state=k, n_jobs=-1))


class ETRF:
    def __init__(self, k): self.a, self.b = et(k), RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=k, n_jobs=-1)
    def fit(self, X, y): self.a.fit(X, y); self.b.fit(X, y); return self
    def predict_proba(self, X): return 0.5 * (self.a.predict_proba(X) + self.b.predict_proba(X))


res["ET+RF"] = ar.run("12. ET + RF probability average", ETRF)


# 13. mirror augmentation: add swapped (own<->other) rows with swapped labels for fault files
def run_mirror(name):
    oof = np.zeros((3, ar.n, 2))
    for k, (tr, te) in enumerate(ar.SPLITS):
        F = ar.features_for(tr)
        Rtr, ytr = ar.side_rows(F, tr)
        # mirrored copy: swap own_ and rel_ sign (rel = own - other -> other - own needs other's own values)
        rows, ys = [], []
        for i in tr:
            r1, r2 = ar.side_relative_rows(F.iloc[i].to_dict())
            if ar.y[i] == "Normal":
                continue
            # a Side I fault seen "from side II's perspective" is a Side II fault: swap the two rows' labels
            rows += [r2, r1]; ys += [int(ar.y[i] == "Side I"), int(ar.y[i] == "Side II")]
        Raug = pd.concat([Rtr, pd.DataFrame(rows).fillna(-9)[Rtr.columns]], ignore_index=True)
        yaug = np.concatenate([ytr, ys])
        clf = et(k).fit(Raug, yaug)
        Rte, _ = ar.side_rows(F, te)
        oof[k // 5][te] = clf.predict_proba(Rte[Rtr.columns])[:, 1].reshape(-1, 2)
    scores = []
    for rep in range(3):
        scores.append(max(ar.rail_macro_f1(ar.y, ar.decode_side(oof[rep][:, 0], oof[rep][:, 1], t))["macro_f1"] for t in np.linspace(0.1, 0.9, 33)))
    print(f"{name:52s} macroF1={np.mean(scores):.4f}±{np.std(scores):.4f}", flush=True)
    return float(np.mean(scores)), float(np.std(scores))


res["ET mirror"] = run_mirror("13. ET + side-mirroring augmentation")
joblib.dump(res, W / "autoresearch2_results.joblib")
print("\nsummary:"); [print(f"  {k:20s} {m:.4f} ± {s:.4f}") for k, (m, s) in res.items()]
