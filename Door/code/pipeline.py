"""Door subsystem: segmentation of the continuous stream + per-cycle feature extraction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.metrics import parse_door_time, format_door_time  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Column positions per the Info Kit (names in the CSV differ slightly from the doc -> map by position)
COLS = ["dt", "current", "voltage", "bemf", "t_open", "t_close", "cmd_close", "cmd_open",
        "dcsr", "dcsl", "dlsr", "dlsl", "opened", "locked", "opening", "closing", "pos"]
GAP_SECONDS = 1.0
N_BINS = 20


def load_stream(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert df.shape[1] == 17, f"expected 17 columns, got {df.shape[1]}"
    df.columns = COLS
    df["t"] = df["dt"].map(parse_door_time)
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    return df


def segment(df: pd.DataFrame, gap_seconds: float = GAP_SECONDS) -> list[pd.DataFrame]:
    """Cut the stream wherever the inter-row gap exceeds gap_seconds."""
    dt = df["t"].diff().dt.total_seconds().fillna(0).to_numpy()
    seg_id = np.cumsum(dt > gap_seconds)
    return [g for _, g in df.groupby(seg_id, sort=True)]


def operation_of(seg: pd.DataFrame) -> str:
    """Open if position increases over the cycle, else Close."""
    return "Open" if seg["pos"].iloc[-1] > seg["pos"].iloc[0] else "Close"


def resample_profile(seg: pd.DataFrame, col: str, n_bins: int = N_BINS) -> np.ndarray:
    """Mean of `col` in n_bins equal bins of normalised time."""
    x = np.linspace(0, 1, len(seg), endpoint=False)
    bins = np.minimum((x * n_bins).astype(int), n_bins - 1)
    v = seg[col].to_numpy(float)
    out = np.array([v[bins == b].mean() if np.any(bins == b) else np.nan for b in range(n_bins)])
    return pd.Series(out).interpolate(limit_direction="both").to_numpy()


def basic_features(seg: pd.DataFrame) -> dict:
    cur = seg["current"].to_numpy(float)
    pos = seg["pos"].to_numpy(float)
    vel = np.abs(np.diff(pos, prepend=pos[0]))
    volt = seg["voltage"].to_numpy(float)
    bemf = seg["bemf"].to_numpy(float)
    n = len(seg)
    dur = (seg["t"].iloc[-1] - seg["t"].iloc[0]).total_seconds()
    peak = cur.max()
    # exclude the final locking spike (last ~10% of rows) for "travel" current statistics
    k = max(1, int(0.9 * n))
    trav = cur[:k]
    f = dict(
        n_rows=n, duration=dur, op_is_open=float(operation_of(seg) == "Open"),
        cur_peak=peak, cur_mean=cur.mean(), cur_rms=np.sqrt((cur ** 2).mean()), cur_std=cur.std(),
        cur_integral=cur.sum() * 0.02, cur_p90=np.percentile(cur, 90), cur_median=np.median(cur),
        trav_mean=trav.mean(), trav_p90=np.percentile(trav, 90), trav_max=trav.max(), trav_std=trav.std(),
        trav_above_300=(trav > 300).mean(), trav_above_500=(trav > 500).mean(),
        # middle-of-travel (uniform-speed phase) current: bins 30%-80% of time
        mid_mean=cur[int(0.3 * n):int(0.8 * n)].mean(), mid_max=cur[int(0.3 * n):int(0.8 * n)].max(),
        t_open_col=seg["t_open"].iloc[0], t_close_col=seg["t_close"].iloc[0],
        vel_mean=vel.mean(), vel_max=vel.max(), vel_min_mid=vel[int(0.2 * n):int(0.8 * n)].min(),
        stall_rows=int(((vel == 0) & (cur > 200)).sum()),
        volt_mean=volt.mean(), volt_max=volt.max(), bemf_mean=bemf.mean(), bemf_max=bemf.max(),
        energy=(volt * 0.01 * cur * 0.001 * 0.02).sum(),
        bemf_per_volt=(bemf.mean() / volt.mean()) if volt.mean() else 0.0,
        pos_range=abs(pos[-1] - pos[0]), pos_start=pos[0], pos_end=pos[-1],
        cur_above_bemf_slope=np.polyfit(bemf[:k], trav, 1)[0] if k > 2 else 0.0,
    )
    return f


class ReferenceProfiles:
    """Per-operation median/IQR current profile from Normal cycles (fit on training fold only)."""

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins
        self.ref: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, segs: list[pd.DataFrame], labels: list[str]) -> "ReferenceProfiles":
        by_op: dict[str, list[np.ndarray]] = {"Open": [], "Close": []}
        for s, y in zip(segs, labels):
            if y == "Normal":
                by_op[operation_of(s)].append(resample_profile(s, "current", self.n_bins))
        for op, arr in by_op.items():
            a = np.stack(arr) if arr else np.zeros((1, self.n_bins))
            med = np.median(a, axis=0)
            iqr = np.subtract(*np.percentile(a, [75, 25], axis=0)) + 1e-6
            self.ref[op] = (med, iqr)
        return self

    def excess_features(self, seg: pd.DataFrame) -> dict:
        med, iqr = self.ref[operation_of(seg)]
        prof = resample_profile(seg, "current", self.n_bins)
        z = (prof - med) / iqr
        pos = np.clip(z, 0, None)
        above = z > 2
        # longest run of bins above +2 IQR
        run = best = 0
        for a in above:
            run = run + 1 if a else 0
            best = max(best, run)
        f = dict(exc_max=z.max(), exc_mean=z.mean(), exc_pos_integral=pos.sum(), exc_argmax=int(z.argmax()),
                 exc_run=best, exc_n_above2=int(above.sum()), exc_l2=float(np.sqrt((z ** 2).mean())),
                 exc_mid_mean=z[int(0.3 * self.n_bins):int(0.8 * self.n_bins)].mean())
        f.update({f"prof_{i}": prof[i] for i in range(self.n_bins)})
        f.update({f"z_{i}": z[i] for i in range(self.n_bins)})
        return f


def features_table(segs: list[pd.DataFrame], ref: ReferenceProfiles) -> pd.DataFrame:
    rows = []
    for s in segs:
        f = basic_features(s)
        f.update(ref.excess_features(s))
        rows.append(f)
    return pd.DataFrame(rows)


def segments_to_frame(segs: list[pd.DataFrame], labels: list[str] | None = None) -> pd.DataFrame:
    out = pd.DataFrame({
        "start_time": [format_door_time(s["t"].iloc[0]) for s in segs],
        "end_time": [format_door_time(s["t"].iloc[-1]) for s in segs],
    })
    if labels is not None:
        out["prediction"] = labels
    return out


def match_labels(segs: list[pd.DataFrame], answer: pd.DataFrame) -> list[str]:
    """Assign the ground-truth status to each predicted segment by start_time (train only)."""
    ans = answer.copy()
    ans["s"] = ans["start_time"].map(parse_door_time)
    lookup = dict(zip(ans["s"], ans["status"]))
    return [lookup.get(s["t"].iloc[0], None) for s in segs]


class RfLrEnsemble:
    """Mean of RF probability on all features and logistic-regression probability on 4 physical features."""

    LR_FEATURES = ["mid_mean", "trav_mean", "exc_mid_mean", "op_is_open"]

    def __init__(self, seed: int = 0):
        self.rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=2,
                                         random_state=seed, n_jobs=-1)
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced"))

    def fit(self, X, y):
        self.rf.fit(X, y); self.lr.fit(X[self.LR_FEATURES], y); return self

    def predict_proba(self, X):
        p = 0.5 * (self.rf.predict_proba(X)[:, 1] + self.lr.predict_proba(X[self.LR_FEATURES])[:, 1])
        return np.stack([1 - p, p], axis=1)

    @property
    def feature_importances_(self):
        return self.rf.feature_importances_


class DoorPipeline:
    """End-to-end inference pipeline for Door abnormal resistance detection."""
    def __init__(self, feature_columns: list[str], model: Any, threshold: float = 0.35):
        self.feature_columns = feature_columns
        self.model = model
        self.threshold = threshold

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[self.feature_columns].values
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df_features: pd.DataFrame) -> list[str]:
        probs = self.predict_proba(df_features)
        preds = []
        for p in probs:
            if p >= self.threshold:
                preds.append("Abnormal resistance")
            else:
                preds.append("Normal")
        return preds
