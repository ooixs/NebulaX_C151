"""Rail Corrugation: speed decode, dual-domain per-channel spectral features, side aggregation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

FS = 10_000
TEETH, EDGES_PER_TOOTH, WHEEL_D = 90, 2, 0.85
NPERSEG = 1024
FREQ_BANDS = [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 5000)]
# corrugation wavelength bands in metres (third-octave-like, 25 mm .. 1 m)
LAMBDA_BANDS = [(0.025, 0.05), (0.05, 0.08), (0.08, 0.125), (0.125, 0.2), (0.2, 0.315), (0.315, 0.5), (0.5, 1.0)]
SIDE_POS = {1: [1, 3, 5, 7], 2: [2, 4, 6, 8]}
CLASSES = ["Normal", "Side I", "Side II"]


def load_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pulse (n,), vib (n,64), shock (n,64); channel k = car k//8, position k%8 + 1."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"input CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"input CSV is empty: {path.name}") from exc
    if frame.shape[0] == 0:
        raise ValueError(f"input CSV has no data rows: {path.name}")
    if frame.shape[1] != 129:
        raise ValueError(f"expected 129 columns, found {frame.shape[1]} in {path.name}")
    expected = ["Rotating speed"] + [f"{kind} of bearing in position {position} of car {car}"
              for car in range(1, 9) for position in range(1, 9) for kind in ("Vibration", "Shock")]
    if frame.columns.tolist() != expected:
        raise ValueError(f"unexpected Rail header names or ordering in {path.name}")
    try:
        a = frame.to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"nonnumeric value in {path.name}") from exc
    if not np.isfinite(a).all():
        raise ValueError(f"NaN or infinite value in {path.name}")
    return a[:, 0], a[:, 1::2], a[:, 2::2]


def speed(pulse: np.ndarray) -> float:
    trans = int((np.diff(pulse) != 0).sum())
    return trans / (EDGES_PER_TOOTH * TEETH) * np.pi * WHEEL_D * (FS / len(pulse))


def channel_side(k: int) -> int:
    return 1 if (k % 8 + 1) in SIDE_POS[1] else 2


def channel_features(vib: np.ndarray, shock: np.ndarray, v: float) -> pd.DataFrame:
    """One row per channel (64) with time, frequency and wavelength-domain features."""
    f, P = welch(vib, fs=FS, nperseg=NPERSEG, axis=0)  # P: (nfreq, 64)
    logP = np.log10(P + 1e-15)
    rows = {}
    for lo, hi in FREQ_BANDS:
        m = (f >= lo) & (f < hi)
        rows[f"fb_{lo}_{hi}"] = logP[m].mean(axis=0)
    if v > 1.0:
        lam = np.divide(v, f, out=np.full_like(f, np.inf), where=f > 0)
        for lo, hi in LAMBDA_BANDS:
            m = (lam >= lo) & (lam < hi)
            rows[f"lb_{int(lo*1000)}_{int(hi*1000)}"] = logP[m].mean(axis=0) if m.any() else np.full(64, np.nan)
        # dominant wavelength in the corrugation range and its prominence
        m = (lam >= 0.025) & (lam < 1.0)
        Pm = P[m]
        idx = Pm.argmax(axis=0)
        rows["dom_lambda"] = lam[m][idx]
        rows["dom_prom"] = np.log10(Pm.max(axis=0) + 1e-15) - np.log10(np.median(Pm, axis=0) + 1e-15)
    else:
        for lo, hi in LAMBDA_BANDS:
            rows[f"lb_{int(lo*1000)}_{int(hi*1000)}"] = np.full(64, np.nan)
        rows["dom_lambda"] = np.full(64, np.nan)
        rows["dom_prom"] = np.full(64, np.nan)
    m = (f >= 20) & (f < 2000)
    Pn = P[m] / (P[m].sum(axis=0) + 1e-15)
    rows["spec_entropy"] = -(Pn * np.log(Pn + 1e-15)).sum(axis=0)
    rows["spec_centroid"] = (f[m][:, None] * Pn).sum(axis=0)
    rows["rms"] = np.sqrt((vib ** 2).mean(axis=0))
    rows["log_rms"] = np.log10(rows["rms"] + 1e-9)
    rows["kurt"] = kurtosis(vib, axis=0)
    rows["crest"] = np.abs(vib).max(axis=0) / (rows["rms"] + 1e-9)
    rows["shock_rms"] = np.sqrt((shock ** 2).mean(axis=0))
    rows["shock_log_rms"] = np.log10(rows["shock_rms"] + 1e-9)
    rows["shock_kurt"] = kurtosis(shock, axis=0)
    sd = shock.std(axis=0) + 1e-9
    rows["shock_peaks"] = (np.abs(shock - shock.mean(axis=0)) > 4 * sd).sum(axis=0)
    df = pd.DataFrame(rows)
    df["car"] = np.arange(64) // 8
    df["position"] = np.arange(64) % 8 + 1
    df["side"] = [channel_side(k) for k in range(64)]
    return df


REF_FEATURES = [f"fb_{lo}_{hi}" for lo, hi in FREQ_BANDS] + ["log_rms", "shock_log_rms"]
SPEED_BINS = np.array([0, 3, 6, 9, 12, 15, 30])


def speed_bin(v: float) -> int:
    return int(np.clip(np.digitize(v, SPEED_BINS) - 1, 0, len(SPEED_BINS) - 2))


class NormalReference:
    """Fold-local robust Normal reference at side, position, or channel granularity."""

    def __init__(self, granularity: str = "side", smooth_speed: bool = False):
        if granularity not in {"side", "position", "channel"}:
            raise ValueError(f"unknown reference granularity: {granularity}")
        self.granularity = granularity
        self.smooth_speed = smooth_speed
        self.ref: dict[tuple[int, tuple], tuple[pd.Series, pd.Series]] = {}
        self.global_ref: dict[tuple, tuple[pd.Series, pd.Series]] = {}

    def _groups(self, ct: pd.DataFrame):
        if self.granularity == "side":
            return [(s, ct.side == s) for s in (1, 2)]
        if self.granularity == "position":
            return [(p, ct.position == p) for p in range(1, 9)]
        return [((c, p), (ct.car == c) & (ct.position == p))
                for c in range(8) for p in range(1, 9)]

    def fit(self, chan_tables: list[pd.DataFrame], speeds: list[float], labels: list[str]) -> "NormalReference":
        pool: dict[tuple[int, tuple], list] = {}
        gpool: dict[tuple, list] = {}
        for ct, v, y in zip(chan_tables, speeds, labels):
            if y != "Normal":
                continue
            b = speed_bin(v)
            for group, mask in self._groups(ct):
                sub = ct.loc[mask, REF_FEATURES]
                pool.setdefault((b, group), []).append(sub)
                gpool.setdefault(group, []).append(sub)
        for key, lst in pool.items():
            a = pd.concat(lst)
            self.ref[key] = (a.median(), a.quantile(0.75) - a.quantile(0.25) + 1e-3)
        for s, lst in gpool.items():
            a = pd.concat(lst)
            self.global_ref[s] = (a.median(), a.quantile(0.75) - a.quantile(0.25) + 1e-3)
        return self

    def excess(self, ct: pd.DataFrame, v: float) -> pd.DataFrame:
        b = speed_bin(v)
        out = ct.copy()
        for group, m in self._groups(out):
            if self.smooth_speed:
                available = [(abs(speed_bin_value - b), value) for (speed_bin_value, key), value in self.ref.items()
                             if key == group]
                med, iqr = min(available, key=lambda item: item[0])[1] if available else self.global_ref[group]
            else:
                med, iqr = self.ref.get((b, group), self.global_ref[group])
            for c in REF_FEATURES:
                out.loc[m, f"ex_{c}"] = (out.loc[m, c] - med[c]) / iqr[c]
        return out


AGG_FEATURES = None  # filled lazily


def aggregate(ct: pd.DataFrame, v: float) -> dict:
    """Side-level aggregates + cross-side differences + per-car max excess."""
    feats = {"speed": v, "speed_bin": speed_bin(v)}
    cols = [c for c in ct.columns if c not in ("car", "position", "side")]
    per_side = {}
    for s in (1, 2):
        sub = ct[ct.side == s][cols]
        agg = pd.concat([sub.mean().add_suffix("_mean"), sub.max().add_suffix("_max"),
                         sub.quantile(0.75).add_suffix("_p75"), sub.std().add_suffix("_std")])
        ex_cols = [c for c in cols if c.startswith("ex_")]
        if ex_cols:
            agg = pd.concat([agg, (sub[ex_cols] > 2).sum().add_suffix("_nover2")])
        per_side[s] = agg
        feats.update({f"s{s}_{k}": val for k, val in agg.items()})
    diff = per_side[1] - per_side[2]
    feats.update({f"d_{k}": val for k, val in diff.items()})
    feats["absd_log_rms_mean"] = abs(diff["log_rms_mean"])
    # same-side agreement of dominant wavelength
    for s in (1, 2):
        dl = ct[ct.side == s]["dom_lambda"].dropna()
        if len(dl) > 2:
            modal = dl.median()
            feats[f"s{s}_agree"] = float(((dl > 0.9 * modal) & (dl < 1.1 * modal)).mean())
        else:
            feats[f"s{s}_agree"] = np.nan
    # per-car max of side excess (keeps local evidence)
    ex = "ex_log_rms" if "ex_log_rms" in ct else "log_rms"
    for s in (1, 2):
        carmax = ct[ct.side == s].groupby("car")[ex].max()
        feats[f"s{s}_carmax_top1"] = carmax.max()
        feats[f"s{s}_carmax_top2"] = carmax.nlargest(2).mean()
    return feats


def file_channel_table(path: str | Path) -> tuple[pd.DataFrame, float]:
    pulse, vib, shock = load_file(path)
    v = speed(pulse)
    return channel_features(vib, shock, v), v


def side_relative_rows(feats: dict) -> list[dict]:
    """Two rows per file (own side vs other side) for the shared per-side detector."""
    rows = []
    for own, oth in ((1, 2), (2, 1)):
        r = {"speed": feats["speed"], "speed_bin": feats["speed_bin"]}
        for k, val in feats.items():
            if k.startswith(f"s{own}_"):
                base = k[3:]
                r[f"own_{base}"] = val
                other = feats.get(f"s{oth}_{base}", np.nan)
                r[f"rel_{base}"] = val - other if isinstance(val, (int, float, np.floating)) else np.nan
        rows.append(r)
    return rows
