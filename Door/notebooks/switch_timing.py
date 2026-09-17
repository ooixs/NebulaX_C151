"""Independent evidence for the 7 borderline test cycles: DCS/DLS switch actuation timing.

Abnormal resistance slows the door, so the door-close switches (DCS*) and lock switches (DLS*)
should actuate later in the cycle. This axis was not used by the classifier, so it provides
independent corroboration (or refutation) of the Normal calls on the borderline cycles.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Door.code.pipeline import load_stream, match_labels, operation_of, segment  # noqa: E402

pd.set_option("display.width", 200)


def first_transition(v: np.ndarray, kind: str) -> float:
    """Row index of the first 0->1 ('act') or 1->0 ('rel') transition; NaN if none."""
    d = np.diff(v.astype(int))
    idx = np.flatnonzero(d == (1 if kind == "act" else -1))
    return float(idx[0] + 1) if len(idx) else np.nan


def switch_features(seg: pd.DataFrame) -> dict:
    op = operation_of(seg)
    n = len(seg)
    f = {"op": op, "n": n}
    kind = "act" if op == "Close" else "rel"  # Close: switches actuate; Open: they release
    for c in ["dcsr", "dcsl", "dlsr", "dlsl", "locked", "opened"]:
        k = kind
        if c == "locked":
            k = "act" if op == "Close" else "rel"
        if c == "opened":
            k = "act" if op == "Open" else "rel"
        f[f"t_{c}"] = first_transition(seg[c].to_numpy(), k) * 0.02  # seconds from cycle start
    dcs = np.nanmean([f["t_dcsr"], f["t_dcsl"]])
    dls = np.nanmean([f["t_dlsr"], f["t_dlsl"]])
    f["t_dcs"] = dcs
    f["t_dls"] = dls
    f["dcs_to_dls"] = dls - dcs if np.isfinite(dcs) and np.isfinite(dls) else np.nan
    f["t_end_minus_dls"] = n * 0.02 - dls if np.isfinite(dls) else np.nan
    # position when DCS actuates/releases (Close: should be near 0)
    if np.isfinite(dcs):
        f["pos_at_dcs"] = float(seg["pos"].iloc[min(int(dcs / 0.02), n - 1)])
    return f


def table(segs, labels=None):
    t = pd.DataFrame([switch_features(s) for s in segs])
    if labels is not None:
        t["y"] = labels
    return t


tr = load_stream(ROOT / "data/Door/Train.csv")
ans = pd.read_csv(ROOT / "data/Door/Train_Segments_Answer.csv")
segs = segment(tr)
T = table(segs, match_labels(segs, ans))
te = load_stream(ROOT / "data/Door/Test.csv")
st = segment(te)
E = table(st)
E["idx"] = range(len(st))

cols = ["t_dcs", "t_dls", "dcs_to_dls", "t_end_minus_dls", "t_locked", "t_opened", "pos_at_dcs"]
print("=== TRAIN switch timing (seconds from cycle start) ===")
print(T.groupby(["op", "y"])[cols].agg(["min", "median", "max"]).round(2).to_string())

BORDER = [4, 20, 36, 8, 27, 28, 34]  # 3 Close + 4 Open borderline test cycles
print("\n=== TEST borderline cycles ===")
print(E[E.idx.isin(BORDER)][["idx", "op"] + cols].round(2).to_string(index=False))
print("\n=== TEST clear-normal (per op median) and clear-abnormal cycles for context ===")
clear_ab = [30, 10, 37, 33, 22, 19, 18, 35]
print("clear-abnormal:")
print(E[E.idx.isin(clear_ab)][["idx", "op"] + cols].round(2).to_string(index=False))
rest = E[~E.idx.isin(BORDER + clear_ab)]
print("clear-normal medians:")
print(rest.groupby("op")[cols].median().round(2).to_string())
