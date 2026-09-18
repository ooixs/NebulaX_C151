"""SHM: rainflow counting + Miner's rule damage model with MAPE-calibrated S-N constants."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rainflow


def load_series(path: str | Path) -> np.ndarray:
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy(float)


def cycles(x: np.ndarray) -> np.ndarray:
    """(n_cycles, 3) array of [range, mean, count] per ASTM E1049 (count 0.5 for residue half cycles)."""
    return np.array([(r, m, c) for r, m, c, _, _ in rainflow.extract_cycles(x)], float).reshape(-1, 3)


def damage_sum(cyc: np.ndarray, m: float, residue: str = "half", magnitude: str = "amplitude",
               goodman_su: float | None = None, cutoff: float = 0.0, range_bins: int | None = None,
               range_bin_width: float | None = None, bin_mode: str = "ceil") -> float:
    """S_m = sum(count_i * a_i^m) with the given conventions; D = S_m / C."""
    if range_bins is not None and (range_bins < 1 or int(range_bins) != range_bins):
        raise ValueError("range_bins must be a positive integer")
    if range_bin_width is not None and range_bin_width <= 0:
        raise ValueError("range_bin_width must be positive")
    if range_bins is not None and range_bin_width is not None:
        raise ValueError("choose range_bins or range_bin_width, not both")
    if bin_mode not in ("ceil", "midpoint", "nearest"):
        raise ValueError("unknown range-bin convention")
    rng, mean, cnt = cyc[:, 0], cyc[:, 1], cyc[:, 2].copy()
    width = (float(rng.max()) / range_bins if len(rng) else 0.0) if range_bins is not None else range_bin_width
    if width:
        bins = np.ceil(rng / width - 1e-12)
        if bin_mode == "midpoint":
            bins = np.maximum(0.0, bins - 0.5)
        elif bin_mode == "nearest":
            bins = np.floor(rng / width + 0.5)
        rng = bins * width
    if residue == "full":
        cnt = np.where(cnt == 0.5, 1.0, cnt)
    elif residue == "drop":
        cnt = np.where(cnt == 0.5, 0.0, cnt)
    a = rng / 2.0 if magnitude == "amplitude" else rng
    if goodman_su:
        a = a / np.clip(1.0 - mean / goodman_su, 1e-3, None)
    if cutoff > 0:
        cnt = np.where(a < cutoff, 0.0, cnt)
    return float(np.sum(cnt * a ** m))


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / np.abs(y)))


def fit_C(S: np.ndarray, D: np.ndarray) -> float:
    """MAPE-optimal C for D_hat = S / C: weighted median of S/D with weights S/D."""
    r = S / D
    w = r
    order = np.argsort(r)
    cw = np.cumsum(w[order])
    return float(r[order][np.searchsorted(cw, cw[-1] / 2.0)])


class DamageModel:
    def __init__(self, m: float, C: float, residue: str = "half", magnitude: str = "amplitude",
                 goodman_su: float | None = None, cutoff: float = 0.0, range_bins: int | None = None,
                 range_bin_width: float | None = None, bin_mode: str = "ceil"):
        self.m, self.C, self.residue, self.magnitude, self.goodman_su, self.cutoff = m, C, residue, magnitude, goodman_su, cutoff
        self.range_bins, self.range_bin_width, self.bin_mode = range_bins, range_bin_width, bin_mode

    def S(self, cyc: np.ndarray) -> float:
        return damage_sum(cyc, self.m, self.residue, self.magnitude, self.goodman_su, self.cutoff,
                          self.range_bins, self.range_bin_width, self.bin_mode)

    def predict_from_cycles(self, cyc: np.ndarray) -> float:
        return self.S(cyc) / self.C

    def predict(self, x: np.ndarray) -> float:
        return self.predict_from_cycles(cycles(x))

    def to_dict(self) -> dict:
        return dict(m=self.m, C=self.C, residue=self.residue, magnitude=self.magnitude,
                    goodman_su=self.goodman_su, cutoff=self.cutoff, range_bins=self.range_bins,
                    range_bin_width=self.range_bin_width, bin_mode=self.bin_mode)

    @classmethod
    def from_dict(cls, d: dict) -> "DamageModel":
        return cls(**d)


CORE_FEATURES = [
    "log_rf_energy_range_m_4.25",
    "log_rf_energy_range_m_3.5",
    "log_rf_energy_range_m_3.0",
    "log_rf_goodman_su_600_m_3.5",
    "p95_p05",
    "p99_p01",
    "std",
    "ptp",
    "rf_p95_range",
    "p01",
    "crest_factor",
    "spec_alpha2",
    "spec_zero_crossing",
]


class SHMPipeline:
    """End-to-end inference pipeline for SHM cumulative fatigue damage."""
    def __init__(self, features: list[str], scaler: Any, models: dict[str, Any], weights: dict[str, float]):
        self.features = features
        self.scaler = scaler
        self.models = models
        self.weights = weights

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[self.features].values
        X_scaled = self.scaler.transform(X)

        preds = np.zeros(len(df_features), dtype=np.float64)
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w > 0:
                pred_log = model.predict(X_scaled)
                pred_val = np.exp(pred_log)
                preds += w * pred_val

        return np.clip(preds, 0.001, 1.5)
