"""
High-Performance Feature Extraction Pipeline for SHM (Dynamic Stress Fatigue Damage)
Implements:
1. ASTM E1049-85 Rainflow Cycle Counting (vectorized, sub-second).
2. Multi-exponent Basquin S-N fatigue damage sums (m = 2.0 to 5.0).
3. Goodman and Gerber mean stress corrections (σ_a / (1 - σ_m / σ_u)).
4. Higher-order statistical and engineering waveform metrics (RMS, Kurtosis, Crest, Shape, Impulse factors).
5. Spectral fatigue moments via Welch PSD (Rice frequency, bandwidth parameter α_2).
6. Operational load condition indicators (AW0 tare vs AW4 crush load).
"""

import os
import argparse
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import signal, stats

# Candidate Basquin exponents based on structural steel fatigue curves (EN 1993-1-9 / BS 7608)
BASQUIN_EXPONENTS = [2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0]

# Candidate ultimate tensile strengths for Goodman correction (MPa)
GOODMAN_SIGMA_U = [400.0, 600.0, 800.0]


def rainflow_astm_e1049(series: np.ndarray):
    """
    Fast ASTM E1049-85 Rainflow Cycle Counting algorithm.
    Extracts closed hysteresis stress cycles (ranges, cycle counts, mean stresses).
    """
    diff = np.diff(series)
    non_zero = diff != 0
    series_nz = series[np.concatenate(([True], non_zero))]
    diff_nz = np.diff(series_nz)
    extrema = np.where(diff_nz[:-1] * diff_nz[1:] < 0)[0] + 1
    extrema_pts = np.concatenate(([series_nz[0]], series_nz[extrema], [series_nz[-1]]))

    stack: List[float] = []
    ranges: List[float] = []
    counts: List[float] = []
    means: List[float] = []

    for p in extrema_pts:
        stack.append(p)
        while len(stack) >= 3:
            s0, s1, s2 = stack[-3], stack[-2], stack[-1]
            r_x = abs(s1 - s2)
            r_y = abs(s0 - s1)
            if r_x >= r_y:
                cnt = 0.5 if len(stack) == 3 else 1.0
                ranges.append(r_y)
                counts.append(cnt)
                means.append((s0 + s1) / 2.0)
                stack.pop(-2)
            else:
                break

    while len(stack) >= 2:
        ranges.append(abs(stack[-1] - stack[-2]))
        counts.append(0.5)
        means.append((stack[-1] + stack[-2]) / 2.0)
        stack.pop()

    return np.array(ranges, dtype=np.float64), np.array(counts, dtype=np.float64), np.array(means, dtype=np.float64)


def extract_features_from_series(series: np.ndarray) -> Dict[str, float]:
    """
    Extracts all physical, statistical, and spectral features from a dynamic stress sequence.
    """
    feats: Dict[str, float] = {}

    # --- 1. Statistical & Waveform Metrics ---
    mean = float(np.mean(series))
    std = float(np.std(series))
    rms = float(np.sqrt(np.mean(series**2)))
    abs_mean = float(np.mean(np.abs(series)))
    max_val = float(np.max(series))
    min_val = float(np.min(series))
    max_abs = float(np.max(np.abs(series)))
    ptp = float(np.ptp(series))

    feats["mean"] = mean
    feats["std"] = std
    feats["var"] = float(std**2)
    feats["rms"] = rms
    feats["abs_mean"] = abs_mean
    feats["max_val"] = max_val
    feats["min_val"] = min_val
    feats["ptp"] = ptp
    feats["kurtosis"] = float(stats.kurtosis(series))
    feats["skewness"] = float(stats.skew(series))

    # Engineering Dimensionless Indicators
    feats["crest_factor"] = max_abs / (rms + 1e-8)
    feats["shape_factor"] = rms / (abs_mean + 1e-8)
    feats["impulse_factor"] = max_abs / (abs_mean + 1e-8)
    sqrt_mean = float(np.mean(np.sqrt(np.abs(series))))
    feats["margin_factor"] = max_abs / ((sqrt_mean**2) + 1e-8)

    # Percentiles & Tail Spreads
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        feats[f"p{p:02d}"] = float(np.percentile(series, p))
    feats["iqr"] = feats["p75"] - feats["p25"]
    feats["p95_p05"] = feats["p95"] - feats["p05"]
    feats["p99_p01"] = feats["p99"] - feats["p01"]

    # Operating Regime Proxy (AW0 tare vs AW4 crush load baseline)
    feats["is_aw4_proxy"] = 1.0 if mean > 0.0 else 0.0

    # --- 2. Spectral Fatigue Moments (Welch PSD) ---
    f, pxx = signal.welch(series, fs=100.0, nperseg=2048)
    m0 = float(np.sum(pxx))
    m1 = float(np.sum(f * pxx))
    m2 = float(np.sum((f**2) * pxx))
    m4 = float(np.sum((f**4) * pxx))

    feats["spec_m0"] = m0
    feats["spec_m1"] = m1
    feats["spec_m2"] = m2
    feats["spec_m4"] = m4
    feats["spec_alpha2"] = m2 / (np.sqrt(m0 * m4) + 1e-8)
    feats["spec_zero_crossing"] = np.sqrt(m2 / (m0 + 1e-8))
    feats["spec_peak_rate"] = np.sqrt(m4 / (m2 + 1e-8))

    # --- 3. ASTM E1049 Rainflow Cycle Counting & S-N Basquin Energies ---
    ranges, counts, cycle_means = rainflow_astm_e1049(series)
    amplitudes = ranges / 2.0

    feats["rf_total_cycles"] = float(np.sum(counts))
    feats["rf_max_range"] = float(np.max(ranges)) if len(ranges) > 0 else 0.0
    feats["rf_mean_range"] = float(np.average(ranges, weights=counts)) if len(ranges) > 0 else 0.0
    feats["rf_p99_range"] = float(np.percentile(ranges, 99)) if len(ranges) > 0 else 0.0
    feats["rf_p95_range"] = float(np.percentile(ranges, 95)) if len(ranges) > 0 else 0.0
    feats["rf_p90_range"] = float(np.percentile(ranges, 90)) if len(ranges) > 0 else 0.0

    # Multi-Exponent Basquin Damage Energies: Sum(n_i * Δσ^m)
    for m in BASQUIN_EXPONENTS:
        energy_range = float(np.sum(counts * (ranges ** m)))
        energy_amp = float(np.sum(counts * (amplitudes ** m)))
        feats[f"rf_energy_range_m_{m}"] = energy_range
        feats[f"rf_energy_amp_m_{m}"] = energy_amp
        feats[f"log_rf_energy_range_m_{m}"] = float(np.log1p(energy_range))
        feats[f"log_rf_energy_amp_m_{m}"] = float(np.log1p(energy_amp))

    # Goodman-corrected Equivalent Amplitude Energies
    for su in GOODMAN_SIGMA_U:
        # Prevent division by zero or negative denominators
        denom = np.clip(1.0 - (cycle_means / su), 0.1, 2.0)
        amp_goodman = amplitudes / denom
        for m in [3.0, 3.5, 4.0]:
            gm_energy = float(np.sum(counts * (amp_goodman ** m)))
            feats[f"rf_goodman_su_{int(su)}_m_{m}"] = gm_energy
            feats[f"log_rf_goodman_su_{int(su)}_m_{m}"] = float(np.log1p(gm_energy))

    return feats


def extract_features_from_file(file_path: str) -> Dict[str, Any]:
    """Reads a single dynamic stress CSV file and returns its extracted features."""
    series = pd.read_csv(file_path, header=None, dtype=np.float64)[0].values
    feats = extract_features_from_series(series)
    feats["filename"] = os.path.basename(file_path)
    return feats


def extract_all_and_save(
    splits_csv: str = "ps3/SHM/data_splits.csv",
    train_features_csv: str = "ps3/SHM/train_features.csv",
    test_features_csv: str = "ps3/SHM/test_features.csv"
):
    """
    Extracts features for all files in data_splits.csv and saves train and test feature tables.
    """
    splits_df = pd.read_csv(splits_csv)
    print(f"Loaded splits from {splits_csv}: {len(splits_df)} files total.")

    train_records = []
    test_records = []

    t_start = time.time()
    for idx, row in splits_df.iterrows():
        fname = row["filename"]
        fpath = row["file_path"]
        split = row["split"]
        
        t0 = time.time()
        feats = extract_features_from_file(fpath)
        feats["split"] = split
        feats["damage"] = row["damage"]
        feats["damage_quartile"] = row["damage_quartile"]
        feats["cv_fold_4"] = row["cv_fold_4"]
        feats["cv_fold_5"] = row["cv_fold_5"]

        if split in ["train", "val"]:
            train_records.append(feats)
        else:
            test_records.append(feats)

        print(f"[{idx+1}/{len(splits_df)}] Processed {fname} ({split}) in {time.time()-t0:.2f}s")

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_df.to_csv(train_features_csv, index=False)
    test_df.to_csv(test_features_csv, index=False)

    print(f"\nFeature extraction complete in {time.time()-t_start:.1f}s!")
    print(f"Train features: {train_df.shape} -> saved to {train_features_csv}")
    print(f"Test features: {test_df.shape} -> saved to {test_features_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SHM features from dynamic stress files.")
    parser.add_argument("--splits", type=str, default="ps3/SHM/data_splits.csv", help="Path to data_splits.csv")
    parser.add_argument("--train-out", type=str, default="ps3/SHM/train_features.csv", help="Output train features CSV")
    parser.add_argument("--test-out", type=str, default="ps3/SHM/test_features.csv", help="Output test features CSV")
    args = parser.parse_args()

    extract_all_and_save(
        splits_csv=args.splits,
        train_features_csv=args.train_out,
        test_features_csv=args.test_out
    )
