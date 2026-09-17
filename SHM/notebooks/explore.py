"""SHM EDA: series statistics, cycle counts, and the m-scan on the full training set."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SHM.code.pipeline import cycles, damage_sum, fit_C, load_series, mape  # noqa: E402

D = ROOT / "data/SHM"
lab = pd.read_csv(D / "Train_Labels.csv")
print(lab.damage.describe())
x = load_series(D / "Train/train01.csv")
print("train01: n", len(x), "mean", x.mean().round(3), "std", x.std().round(3), "min", x.min().round(2), "max", x.max().round(2))
c = cycles(x)
print("cycles", len(c), "half", (c[:, 2] == 0.5).sum(), "range quantiles", np.percentile(c[:, 0], [50, 90, 99, 100]).round(2))
lengths = [len(load_series(D / "Train" / f)) for f in lab.filename[:8]] + [len(load_series(f)) for f in sorted((D / "Test").glob("*.csv"))[:4]]
print("lengths (train[:8], test[:4]):", lengths)

cyc = {f: cycles(load_series(D / "Train" / f)) for f in lab.filename}
Dtrue = lab.damage.to_numpy()
print("\nm scan (half-cycle residue, amplitude):")
for m in [2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10]:
    S = np.array([damage_sum(cyc[f], m) for f in lab.filename])
    C = fit_C(S, Dtrue)
    print(f"  m={m:<4} MAPE={mape(Dtrue, S / C):.4f}  corr(logS,logD)={np.corrcoef(np.log(S), np.log(Dtrue))[0,1]:.4f}")
# simple stats correlation with damage
stats = pd.DataFrame({f: dict(rms=np.sqrt((load_series(D / 'Train' / f) ** 2).mean()), n_cyc=len(cyc[f]),
                              max_range=cyc[f][:, 0].max()) for f in lab.filename}).T
stats["damage"] = Dtrue
print(stats.corr()["damage"].round(3))
