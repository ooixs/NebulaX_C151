"""ACV subsystem: header-driven loading + per-car peer-relative features."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CAR_RE = re.compile(r"^Car (\d{2}) - (.+)$")

# keyword rules -> canonical role (first match wins, evaluated in order)
ROLE_RULES = [
    ("t_in", lambda p: ("indoor" in p and "temp" in p) or ("cabin" in p and "temp" in p and "detected" in p)),
    ("t_out", lambda p: ("outdoor" in p and "temp" in p) or ("outside" in p and "temp" in p) or ("fresh air" in p and "temp" in p)),
    ("t_set_cool", lambda p: ("control temp" in p and "cool" in p) or p == "target temperature value"),
    ("t_set_heat", lambda p: "control temp" in p and "heat" in p),
    ("mode_run", lambda p: "running mode" in p),
    ("mode_set", lambda p: "setting mode" in p or "control mode" in p),
    ("load_halved", lambda p: "load halved" in p or "load shedding" in p),
    ("info_valid", lambda p: "information valid" in p),
    # rich-format only: discharge (high-side) pressures; undercharge lowers them
    ("p_high_1", lambda p: "system 1" in p and "high pressure" in p),
    ("p_high_2", lambda p: "system 2" in p and "high pressure" in p),
    ("p_low_1", lambda p: "system 1" in p and "low pressure" in p),
    ("p_low_2", lambda p: "system 2" in p and "low pressure" in p),
]
RESPONSE_FEATURES = ["t_in", "t_out", "t_set_cool", "full_cool"]


def load_case(path: str | Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Return (long table indexed by [time, car] with canonical role columns, car ids, role->param map)."""
    df = pd.read_excel(path)
    time_col = next((c for c in df.columns if str(c).strip().lower() == "time"), df.columns[1])
    cars, params = [], {}
    for c in df.columns:
        m = CAR_RE.match(str(c))
        if m:
            cars.append(m.group(1))
            params.setdefault(m.group(2), []).append(m.group(1))
    cars = sorted(set(cars))
    role_map: dict[str, str] = {}
    for role, rule in ROLE_RULES:
        for p in params:
            if role not in role_map and rule(p.lower()):
                role_map[role] = p
    frames = []
    for car in cars:
        d = pd.DataFrame({"time": pd.to_datetime(df[time_col]), "car": car})
        for role, p in role_map.items():
            col = f"Car {car} - {p}"
            d[role] = df[col].values if col in df.columns else np.nan
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)
    for c in ["t_in", "t_out", "t_set_cool", "t_set_heat", "p_high_1", "p_high_2", "p_low_1", "p_low_2"]:
        if c in long:
            long[c] = pd.to_numeric(long[c], errors="coerce")
            long.loc[long[c] <= 0, c] = np.nan  # 0.0 = invalid reading
    for c in ["mode_run", "mode_set", "load_halved", "info_valid"]:
        if c in long:
            long[c] = long[c].astype(str).str.strip()
    if "info_valid" in long:
        bad = long["info_valid"].str.lower().eq("invalid")
        long.loc[bad, ["t_in", "t_out", "t_set_cool"]] = np.nan
    return long, cars, role_map


def _is_cooling(mode: pd.Series) -> pd.Series:
    m = mode.str.lower()
    return m.str.contains("cool") & ~m.str.contains("stop")


def _is_full_cooling(mode: pd.Series) -> pd.Series:
    return mode.str.lower().str.contains("full cool")


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826 + 1e-6
    return (x - med) / mad


def prepare(long: pd.DataFrame) -> pd.DataFrame:
    d = long.copy()
    d["cooling"] = _is_cooling(d["mode_run"]) if "mode_run" in d else True
    d["full_cool"] = _is_full_cooling(d["mode_run"]).astype(float) if "mode_run" in d else 0.0
    if "t_set_cool" in d:
        d["set_err"] = d["t_in"] - d["t_set_cool"]
    d = d.sort_values(["car", "time"])
    d["dT_10"] = d.groupby("car")["t_in"].shift(-10) - d["t_in"]
    return d


class ResponseModel:
    """Healthy-response model: predicts 5-min indoor temperature change from state (fit on healthy cars)."""

    def __init__(self):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
        self.cols = RESPONSE_FEATURES

    def _X(self, d: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=d.index)
        for c in self.cols:
            X[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        return X

    def fit(self, tables: list[pd.DataFrame]) -> "ResponseModel":
        d = pd.concat(tables, ignore_index=True)
        d = d[d["cooling"]]
        X = self._X(d)
        m = X.notna().all(axis=1) & d["dT_10"].notna()
        self.means = X[m].mean()
        self.model.fit(X[m].fillna(self.means), d.loc[m, "dT_10"])
        return self

    def residual(self, d: pd.DataFrame) -> pd.Series:
        X = self._X(d).fillna(getattr(self, "means", 0.0))
        pred = self.model.predict(X)
        r = d["dT_10"] - pred
        r[~d["cooling"]] = np.nan
        return r


def car_features(long: pd.DataFrame, cars: list[str], response: "ResponseModel | None" = None) -> pd.DataFrame:
    """Per-car aggregate features computed relative to the other cars at the same timestamp."""
    d = prepare(long)
    if response is not None:
        d["resid"] = response.residual(d)
    # peer statistics at each timestamp among cars in cooling mode with valid readings
    cool = d[d["cooling"] & d["t_in"].notna()]
    peer_med = cool.groupby("time")["t_in"].transform("median")
    peer_cnt = cool.groupby("time")["t_in"].transform("count")
    d.loc[cool.index, "peer_dev"] = cool["t_in"] - peer_med
    d.loc[cool.index, "peer_cnt"] = peer_cnt
    d.loc[d["peer_cnt"] < 3, "peer_dev"] = np.nan
    # optional pressure deficit vs peers (rich format only): positive = lower discharge pressure than peers
    for p in ["p_high_1", "p_high_2"]:
        if p in d and d[p].notna().any():
            pm = d[d[p].notna()].groupby("time")[p].transform("median")
            d.loc[d[p].notna(), f"{p}_def"] = pm - d.loc[d[p].notna(), p]
    rows = []
    for car in cars:
        c = d[d["car"] == car]
        cc = c[c["cooling"]]
        f = dict(car=car,
                 n=len(c), n_valid=int(c["t_in"].notna().sum()), cool_frac=float(c["cooling"].mean()),
                 full_cool_frac=float(c["full_cool"].mean()),
                 t_in_mean=cc["t_in"].mean(), t_in_p95=cc["t_in"].quantile(0.95), t_in_std=cc["t_in"].std(),
                 peer_dev_mean=cc["peer_dev"].mean(), peer_dev_p90=cc["peer_dev"].quantile(0.9),
                 peer_dev_pos_frac=float((cc["peer_dev"] > 1.0).mean()),
                 peer_dev_persist=_longest_run(cc["peer_dev"].to_numpy() > 1.0),
                 set_err_mean=cc["set_err"].mean() if "set_err" in cc else np.nan,
                 set_err_p95=cc["set_err"].quantile(0.95) if "set_err" in cc else np.nan,
                 set_err_pos_frac=float((cc["set_err"] > 1.0).mean()) if "set_err" in cc else np.nan,
                 dT_mean_cool=cc["dT_10"].mean(), dT_p90_cool=cc["dT_10"].quantile(0.9),
                 invalid_frac=float(c["t_in"].isna().mean()),
                 resid_mean=cc["resid"].mean() if "resid" in cc else np.nan,
                 resid_p90=cc["resid"].quantile(0.9) if "resid" in cc else np.nan,
                 p_high_def=max([cc[f"{p}_def"].mean() for p in ["p_high_1", "p_high_2"]
                                 if f"{p}_def" in cc and cc[f"{p}_def"].notna().any()], default=np.nan))
        rows.append(f)
    return pd.DataFrame(rows).set_index("car")


def score_cars(F: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted sum of robust-z-standardised components; cars with no data go to the bottom."""
    total = pd.Series(0.0, index=F.index)
    for col, w in weights.items():
        if col not in F or w == 0:
            continue
        x = F[col]
        if x.notna().sum() < 2:
            continue
        z = robust_z(x).clip(-5, 5)
        total = total + w * z.fillna(0.0)
    total[F["n_valid"] == 0] = -np.inf
    return total.sort_values(ascending=False)


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)
