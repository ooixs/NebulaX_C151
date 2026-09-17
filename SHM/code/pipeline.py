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
               goodman_su: float | None = None, cutoff: float = 0.0) -> float:
    """S_m = sum(count_i * a_i^m) with the given conventions; D = S_m / C."""
    rng, mean, cnt = cyc[:, 0], cyc[:, 1], cyc[:, 2].copy()
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
    """MAPE-optimal scalar C for D_hat = S / C  ==  weighted median of S/D with weights 1/D."""
    r = S / D
    w = 1.0 / D
    order = np.argsort(r)
    cw = np.cumsum(w[order])
    return float(r[order][np.searchsorted(cw, cw[-1] / 2.0)])


class DamageModel:
    def __init__(self, m: float, C: float, residue: str = "half", magnitude: str = "amplitude",
                 goodman_su: float | None = None, cutoff: float = 0.0):
        self.m, self.C, self.residue, self.magnitude, self.goodman_su, self.cutoff = m, C, residue, magnitude, goodman_su, cutoff

    def S(self, cyc: np.ndarray) -> float:
        return damage_sum(cyc, self.m, self.residue, self.magnitude, self.goodman_su, self.cutoff)

    def predict_from_cycles(self, cyc: np.ndarray) -> float:
        return self.S(cyc) / self.C

    def predict(self, x: np.ndarray) -> float:
        return self.predict_from_cycles(cycles(x))

    def to_dict(self) -> dict:
        return dict(m=self.m, C=self.C, residue=self.residue, magnitude=self.magnitude,
                    goodman_su=self.goodman_su, cutoff=self.cutoff)

    @classmethod
    def from_dict(cls, d: dict) -> "DamageModel":
        return cls(**d)
