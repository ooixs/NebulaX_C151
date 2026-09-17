"""ACV improvement phase: candidate scorers evaluated leave-one-case-out with NO fitted weights."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ACV.code.pipeline import load_case, prepare, robust_z  # noqa: E402
from common.metrics import acv_rank_score  # noqa: E402

D = ROOT / "data/ACV"
lab = pd.read_csv(D / "Train_Labels.csv", dtype=str).set_index("filename")["faulty_car"]
cases = {Path(f).name: load_case(f) for f in sorted(glob.glob(str(D / "Train/*.xlsx")))}


def peer_dev(d, mask_col="cooling", value="t_in", min_peers=3):
    c = d[d[mask_col] & d[value].notna()]
    med = c.groupby("time")[value].transform("median")
    cnt = c.groupby("time")[value].transform("count")
    dev = (c[value] - med).where(cnt >= min_peers)
    return dev.groupby(c["car"]).mean()


def evaluate(fn, name):
    sc = {}
    for n, (long, cars, _) in cases.items():
        d = prepare(long)
        s = fn(d, cars)
        s = s.reindex(cars).fillna(-np.inf)
        sc[n] = acv_rank_score(s.sort_values(ascending=False).index.tolist(), lab[n])
    m = np.mean(list(sc.values()))
    print(f"{name:55s} mean={m:.4f}  " + " ".join(f"{v:.3f}" for v in sc.values()))
    return m


# 0. baseline
evaluate(lambda d, cars: peer_dev(d), "baseline: mean peer deviation (cooling mode)")
# 1. setpoint-adjusted deviation: compare (t_in - t_set) with peers' (t_in - t_set)
def setpoint_adj(d, cars):
    d = d.copy(); d["se"] = d["t_in"] - d["t_set_cool"]
    return peer_dev(d, value="se")
evaluate(setpoint_adj, "1. setpoint-adjusted peer deviation")
# 2. deviation only when outdoor is hot (top half of outdoor temp) -> high thermal load
def hot_only(d, cars):
    d = d.copy(); thr = d["t_out"].median(); d["hot"] = d["cooling"] & (d["t_out"] >= thr)
    return peer_dev(d, mask_col="hot")
evaluate(hot_only, "2. peer deviation during high outdoor temperature only")
# 3. rank aggregation of several indicators (Borda)
def borda(d, cars):
    ind = pd.DataFrame({"dev": peer_dev(d)})
    c = d[d["cooling"]]
    ind["p90"] = (c["t_in"] - c.groupby("time")["t_in"].transform("median")).groupby(c["car"]).quantile(0.9)
    ind["se"] = c["set_err"].groupby(c["car"]).mean() if "set_err" in c else np.nan
    ind["dT"] = -c["dT_10"].groupby(c["car"]).mean()
    return ind.rank().mean(axis=1)
evaluate(borda, "3. Borda rank aggregation (dev, p90, setpoint err, -dT)")
# 4. trimmed-mean robust deviation (drop top/bottom 10% of each car's deviations)
def trimmed(d, cars):
    c = d[d["cooling"] & d["t_in"].notna()]
    dev = c["t_in"] - c.groupby("time")["t_in"].transform("median")
    return dev.groupby(c["car"]).apply(lambda x: x[(x > x.quantile(0.1)) & (x < x.quantile(0.9))].mean())
evaluate(trimmed, "4. trimmed-mean peer deviation")
# 5. deviation of daily maxima (afternoon peak load) vs peers
def daily_max(d, cars):
    c = d[d["cooling"] & d["t_in"].notna()].copy(); c["day"] = c["time"].dt.date
    dm = c.groupby(["day", "car"])["t_in"].max().unstack()
    return (dm.sub(dm.median(axis=1), axis=0)).mean()
evaluate(daily_max, "5. deviation of daily max temperature vs peers")
# 6. fraction of time hotter than ALL peers
def hottest_frac(d, cars):
    c = d[d["cooling"] & d["t_in"].notna()]
    mx = c.groupby("time")["t_in"].transform("max"); cnt = c.groupby("time")["t_in"].transform("count")
    return ((c["t_in"] >= mx) & (cnt >= 3)).groupby(c["car"]).mean()
evaluate(hottest_frac, "6. fraction of timestamps as the hottest car")
# 7. leave-one-out z: deviation vs median of OTHER cars (excludes self from the median)
def loo_median(d, cars):
    c = d[d["cooling"] & d["t_in"].notna()]
    piv = c.pivot_table(index="time", columns="car", values="t_in")
    out = {}
    for car in piv.columns:
        others = piv.drop(columns=car).median(axis=1)
        out[car] = (piv[car] - others).mean()
    return pd.Series(out)
evaluate(loo_median, "7. deviation vs median of the OTHER cars")
# 8. mode-based: full-cooling fraction + deviation (physics: leaky unit runs flat out)
def combo_mode(d, cars):
    return robust_z(peer_dev(d)) + 0.5 * robust_z(d.groupby("car")["full_cool"].mean())
evaluate(combo_mode, "8. peer deviation + 0.5 * full-cooling fraction (fixed weights)")
