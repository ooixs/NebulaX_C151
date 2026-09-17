"""Rail EDA: speed pulse decoding, per-class spectral differences, side asymmetry."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/Rail_Corrugation"
lab = pd.read_csv(D / "Train_Labels.csv")
print(lab.label.value_counts())

FS = 10000


def load(fn):
    a = pd.read_csv(D / "Train" / fn, header=0).to_numpy(float)
    return a[:, 0], a[:, 1:]


# ---- speed pulse
for fn in ["Train1.csv", "Train2.csv", lab[lab.label == "Side I"].filename.iloc[0], lab[lab.label == "Side II"].filename.iloc[0]]:
    p, X = load(fn)
    vals, cnt = np.unique(p, return_counts=True)
    trans = int((np.diff(p) != 0).sum())
    # run lengths of high and low
    runs = np.diff(np.flatnonzero(np.diff(p) != 0))
    print(fn, "pulse values", dict(zip(vals, cnt)), "transitions", trans,
          "run-length median", np.median(runs) if len(runs) else None,
          "v(1 trans/tooth) m/s", trans / 90 * np.pi * 0.85, " v(2 trans/tooth)", trans / 180 * np.pi * 0.85, "shape", X.shape)

# ---- speed distribution across files
speeds = []
for fn in lab.filename:
    p, _ = load(fn)
    speeds.append((np.diff(p) != 0).sum())
lab["trans"] = speeds
print(lab.groupby("label").trans.describe())

# ---- spectra: mean PSD by class for Side I vs Side II vibration channels
def side_channels(side):
    pos = [1, 3, 5, 7] if side == 1 else [2, 4, 6, 8]
    return [2 * (8 * c + (pp - 1)) for c in range(8) for pp in pos]  # vibration col index within X


f = None
psd = {k: {1: [], 2: []} for k in ["Normal", "Side I", "Side II"]}
for fn, y in zip(lab.filename[:120].tolist() + lab[lab.label != "Normal"].filename.tolist(), lab.label[:120].tolist() + lab[lab.label != "Normal"].label.tolist()):
    p, X = load(fn)
    for s in (1, 2):
        ch = X[:, side_channels(s)]
        f, P = welch(ch, fs=FS, nperseg=1024, axis=0)
        psd[y][s].append(np.log10(P.mean(axis=1) + 1e-12))
bands = [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 5000)]
print("\nmean log10 band power (Side I channels | Side II channels) by class:")
for y in psd:
    r1 = np.mean(psd[y][1], axis=0); r2 = np.mean(psd[y][2], axis=0)
    row = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        row.append(f"{lo}-{hi}: {r1[m].mean():.2f}|{r2[m].mean():.2f}")
    print(f"{y:8s} n={len(psd[y][1])}", "  ".join(row))
# RMS per side
print("\nRMS vib per side by class:")
for y in ["Normal", "Side I", "Side II"]:
    fns = lab[lab.label == y].filename.tolist()[:40]
    r = np.array([[np.sqrt((load(fn)[1][:, side_channels(s)] ** 2).mean()) for s in (1, 2)] for fn in fns])
    print(y, "sideI", r[:, 0].mean().round(3), "sideII", r[:, 1].mean().round(3), " ratio I/II median", np.median(r[:, 0] / r[:, 1]).round(3))
