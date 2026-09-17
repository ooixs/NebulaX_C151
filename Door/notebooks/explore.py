"""Quick EDA for Door: timestamp gaps vs. labelled segment boundaries."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.metrics import parse_door_time  # noqa: E402

df = pd.read_csv(ROOT / "data/Door/Train.csv")
ans = pd.read_csv(ROOT / "data/Door/Train_Segments_Answer.csv")
df["t"] = df["Datetime"].map(parse_door_time)
dt = df["t"].diff().dt.total_seconds()
print("rows", len(df), "segments", len(ans), "sum n_rows", ans.n_rows.sum())
print("dt quantiles:\n", dt.describe(percentiles=[.5, .9, .99, .999]))
print("dt > 1s count:", (dt > 1).sum(), " dt > 0.5s:", (dt > 0.5).sum(), " dt>0.1:", (dt > 0.1).sum())
print("top gaps:\n", dt.sort_values(ascending=False).head(15).to_list())

# are labelled boundaries at gaps?
starts = ans.start_time.map(parse_door_time)
ends = ans.end_time.map(parse_door_time)
idx_by_t = pd.Series(df.index.values, index=df.t)
s_idx = idx_by_t.reindex(starts).values
e_idx = idx_by_t.reindex(ends).values
print("start rows found:", np.isfinite(s_idx).sum(), "end rows found:", np.isfinite(e_idx).sum())
print("dt before start rows:", pd.Series(dt.values[s_idx[np.isfinite(s_idx)].astype(int)]).describe())
print("dt after end rows:", pd.Series(dt.values[np.clip(e_idx[np.isfinite(e_idx)].astype(int) + 1, 0, len(df) - 1)]).describe())
# gap rows between segments
inside = np.zeros(len(df), bool)
for s, e in zip(s_idx, e_idx):
    if np.isfinite(s) and np.isfinite(e):
        inside[int(s):int(e) + 1] = True
print("rows inside segments:", inside.sum(), "outside:", (~inside).sum())
out = df[~inside]
print("outside rows sample:\n", out.head(10).to_string())
print("segment durations (s):", (ends - starts).dt.total_seconds().describe())
print(ans.groupby(["operation", "status"]).size())
seg0 = df.iloc[int(s_idx[0]):int(e_idx[0]) + 1]
print(seg0.iloc[::15].to_string())
