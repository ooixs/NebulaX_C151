"""
Feature Extraction for Door Electromechanical Time Series
Extracts physical motor dynamics, steady-state glide statistics, and kinematic indicators.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "is_open",
    "n_rows",
    "cur_mean",
    "cur_std",
    "cur_max",
    "cur_mid_mean",
    "cur_mid_median",
    "cur_mid_std",
    "cur_mid_max",
    "cur_mid_min",
    "cur_mid_p90",
    "cur_inrush_max",
    "cur_sum",
    "volt_mean",
    "power_mean",
    "power_mid_mean",
    "power_sum",
    "emf_mean",
    "emf_mid_mean",
    "emf_std",
    "pos_delta",
]


def extract_features_from_segment(seg_df: pd.DataFrame, operation: str = None) -> Dict[str, Any]:
    """
    Extracts electromechanical domain features from an isolated door opening or closing segment.
    """
    cur = seg_df["Motor current(mA)"].values.astype(np.float64)
    volt = seg_df["Motor Voltage(10mV)"].values.astype(np.float64) * 0.01  # convert to Volts
    emf = seg_df["Motor electrodynamic force"].values.astype(np.float64)
    pos = seg_df["Door leaf position"].values.astype(np.float64)
    n = len(cur)

    # Determine operation if not provided
    if operation is None:
        is_opening_mode = seg_df["Door is opening"].sum() > seg_df["Door is closing"].sum()
        pos_start = pos[0] if n > 0 else 0
        pos_end = pos[-1] if n > 0 else 0
        operation = "Open" if (is_opening_mode or (pos_end > pos_start)) else "Close"

    # Central steady-state gliding window (25% to 75% of travel stroke)
    idx_25 = int(n * 0.25)
    idx_75 = int(n * 0.75)
    if idx_75 <= idx_25:
        idx_25 = 0
        idx_75 = n

    cur_mid = cur[idx_25:idx_75]
    volt_mid = volt[idx_25:idx_75]
    emf_mid = emf[idx_25:idx_75]

    # Electrical power: P(t) = (I / 1000) * V in Watts
    power = (cur / 1000.0) * volt
    power_mid = (cur_mid / 1000.0) * volt_mid

    return {
        "is_open": 1.0 if operation == "Open" else 0.0,
        "n_rows": float(n),
        "cur_mean": float(np.mean(cur)),
        "cur_std": float(np.std(cur)),
        "cur_max": float(np.max(cur)),
        "cur_mid_mean": float(np.mean(cur_mid)),
        "cur_mid_median": float(np.median(cur_mid)),
        "cur_mid_std": float(np.std(cur_mid)),
        "cur_mid_max": float(np.max(cur_mid)),
        "cur_mid_min": float(np.min(cur_mid)),
        "cur_mid_p90": float(np.percentile(cur_mid, 90)),
        "cur_inrush_max": float(np.max(cur[:idx_25])) if idx_25 > 0 else float(np.max(cur)),
        "cur_sum": float(np.sum(cur)),
        "volt_mean": float(np.mean(volt)),
        "power_mean": float(np.mean(power)),
        "power_mid_mean": float(np.mean(power_mid)),
        "power_sum": float(np.sum(power)),
        "emf_mean": float(np.mean(emf)),
        "emf_mid_mean": float(np.mean(emf_mid)),
        "emf_std": float(np.std(emf)),
        "pos_delta": float(abs(pos[-1] - pos[0])) if n > 0 else 0.0,
    }
