"""ACV EDA: per-car features per training case and single-feature ranking power."""
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ACV.code.pipeline import car_features, load_case  # noqa: E402
from common.metrics import acv_rank_score  # noqa: E402

lab = pd.read_csv(ROOT / "data/ACV/Train_Labels.csv", dtype=str).set_index("filename")["faulty_car"]
feats = {}
for f in sorted(glob.glob(str(ROOT / "data/ACV/Train/*.xlsx"))):
    name = Path(f).name
    long, cars, rm = load_case(f)
    F = car_features(long, cars)
    feats[name] = F
    print(name, "faulty", lab[name], "roles", {k: v[:24] for k, v in rm.items()})
    print(F[["cool_frac", "full_cool_frac", "t_in_mean", "peer_dev_mean", "peer_dev_p90", "peer_dev_persist",
             "set_err_mean", "set_err_p95", "dT_mean_cool", "invalid_frac"]].round(3).to_string())
cols = [c for c in feats["acv_case_01.xlsx"].columns if c not in ("n", "n_valid")]
res = {}
for c in cols:
    sc = []
    for name, F in feats.items():
        r = F[c].fillna(F[c].min() - 1).sort_values(ascending=False).index.tolist()
        sc.append(acv_rank_score(r, lab[name]))
    res[c] = (np.mean(sc), [round(s, 3) for s in sc])
print("\nsingle-feature ranking (descending order) mean rank-decay over 6 cases:")
for c, (m, s) in sorted(res.items(), key=lambda x: -x[1][0]):
    print(f"{c:22s} mean={m:.3f} {s}")
pickle.dump(feats, open(ROOT / "weights/ACV/train_features.pkl", "wb"))
