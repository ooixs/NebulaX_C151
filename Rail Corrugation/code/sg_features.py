"""
Feature Extraction Engine for Rail Corrugation Detection & Localization
Transforms 129-channel axle-box vibration/shock recordings (10 kHz, 1 s) into
physics-informed kinematic, bilateral asymmetry, spatial propagation, and spectral features.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import signal, stats

# Constants
SAMPLING_RATE_HZ = 10000
WHEEL_DIAMETER_M = 0.85
TEETH_COUNT = 90
SIDE1_POSITIONS = [1, 3, 5, 7]
SIDE2_POSITIONS = [2, 4, 6, 8]


def extract_features_from_df(df: pd.DataFrame, filename: str = "") -> Dict[str, Any]:
    """
    Extracts comprehensive kinematic, bilateral asymmetry, spatial, and spectral features
    from an axle-box vibration and shock recording DataFrame.

    Args:
        df: DataFrame containing 129 columns (Rotating speed + 128 vibration/shock channels)
        filename: Optional source filename identifier

    Returns:
        Dictionary of scalar numerical features.
    """
    # 1. Kinematic Wheel Speed from Toothed Wheel Sensor
    speed_raw = df.iloc[:, 0].values
    diffs = np.diff(speed_raw)
    pulses = np.sum(diffs > 0)
    rev_per_sec = pulses / float(TEETH_COUNT)
    speed_mps = rev_per_sec * np.pi * WHEEL_DIAMETER_M
    speed_kmh = speed_mps * 3.6

    # 2. Channel Identification
    cols = df.columns
    s1_vib_cols = [c for c in cols if 'Vibration' in c and any(f'position {p}' in c for p in SIDE1_POSITIONS)]
    s2_vib_cols = [c for c in cols if 'Vibration' in c and any(f'position {p}' in c for p in SIDE2_POSITIONS)]
    s1_shk_cols = [c for c in cols if 'Shock' in c and any(f'position {p}' in c for p in SIDE1_POSITIONS)]
    s2_shk_cols = [c for c in cols if 'Shock' in c and any(f'position {p}' in c for p in SIDE2_POSITIONS)]

    v1 = df[s1_vib_cols].values  # (N, 32)
    v2 = df[s2_vib_cols].values  # (N, 32)
    k1 = df[s1_shk_cols].values  # (N, 32)
    k2 = df[s2_shk_cols].values  # (N, 32)

    # 3. Channel-wise RMS & Peak-to-Peak Metrics
    v1_rms_ch = np.sqrt(np.mean(v1**2, axis=0))
    v2_rms_ch = np.sqrt(np.mean(v2**2, axis=0))
    k1_rms_ch = np.sqrt(np.mean(k1**2, axis=0))
    k2_rms_ch = np.sqrt(np.mean(k2**2, axis=0))

    v1_ptp_ch = np.ptp(v1, axis=0)
    v2_ptp_ch = np.ptp(v2, axis=0)

    # Aggregate summaries
    feats: Dict[str, Any] = {
        'filename': filename,
        'speed_mps': float(speed_mps),
        'speed_kmh': float(speed_kmh),
        'is_moving': 1.0 if speed_kmh > 1.0 else 0.0,

        # Side I Vibration Metrics
        's1_vib_rms_mean': float(np.mean(v1_rms_ch)),
        's1_vib_rms_std': float(np.std(v1_rms_ch)),
        's1_vib_rms_max': float(np.max(v1_rms_ch)),
        's1_vib_rms_p90': float(np.percentile(v1_rms_ch, 90)),
        's1_vib_rms_min': float(np.min(v1_rms_ch)),

        # Side II Vibration Metrics
        's2_vib_rms_mean': float(np.mean(v2_rms_ch)),
        's2_vib_rms_std': float(np.std(v2_rms_ch)),
        's2_vib_rms_max': float(np.max(v2_rms_ch)),
        's2_vib_rms_p90': float(np.percentile(v2_rms_ch, 90)),
        's2_vib_rms_min': float(np.min(v2_rms_ch)),

        # Shock Channel Metrics
        's1_shk_rms_mean': float(np.mean(k1_rms_ch)),
        's1_shk_rms_std': float(np.std(k1_rms_ch)),
        's1_shk_rms_max': float(np.max(k1_rms_ch)),
        's2_shk_rms_mean': float(np.mean(k2_rms_ch)),
        's2_shk_rms_std': float(np.std(k2_rms_ch)),
        's2_shk_rms_max': float(np.max(k2_rms_ch)),

        # Combined Full-Train Vibration Level
        'vib_rms_all_mean': float(0.5 * (np.mean(v1_rms_ch) + np.mean(v2_rms_ch))),
        'shk_rms_all_mean': float(0.5 * (np.mean(k1_rms_ch) + np.mean(k2_rms_ch))),

        # Bilateral Asymmetry & Differential Diagnostics (Core Corrugation Localizers)
        'vib_ratio_mean': float(np.mean(v1_rms_ch) / (np.mean(v2_rms_ch) + 1e-6)),
        'vib_diff_mean': float(np.mean(v1_rms_ch) - np.mean(v2_rms_ch)),
        'vib_norm_diff': float((np.mean(v1_rms_ch) - np.mean(v2_rms_ch)) / (np.mean(v1_rms_ch) + np.mean(v2_rms_ch) + 1e-6)),

        'vib_ratio_max': float(np.max(v1_rms_ch) / (np.max(v2_rms_ch) + 1e-6)),
        'vib_diff_max': float(np.max(v1_rms_ch) - np.max(v2_rms_ch)),

        'shk_ratio_mean': float(np.mean(k1_rms_ch) / (np.mean(k2_rms_ch) + 1e-6)),
        'shk_diff_mean': float(np.mean(k1_rms_ch) - np.mean(k2_rms_ch)),
        'shk_norm_diff': float((np.mean(k1_rms_ch) - np.mean(k2_rms_ch)) / (np.mean(k1_rms_ch) + np.mean(k2_rms_ch) + 1e-6)),

        # Peak-to-Peak Amplitudes
        's1_v_ptp_mean': float(np.mean(v1_ptp_ch)),
        's2_v_ptp_mean': float(np.mean(v2_ptp_ch)),
        'v_ptp_ratio': float(np.mean(v1_ptp_ch) / (np.mean(v2_ptp_ch) + 1e-6)),
        'v_ptp_diff': float(np.mean(v1_ptp_ch) - np.mean(v2_ptp_ch)),

        # Speed-Normalized Intensities
        's1_vib_speed_norm': float(np.mean(v1_rms_ch) / (speed_mps + 1.0)),
        's2_vib_speed_norm': float(np.mean(v2_rms_ch) / (speed_mps + 1.0)),
        'all_vib_speed_norm': float(0.5 * (np.mean(v1_rms_ch) + np.mean(v2_rms_ch)) / (speed_mps + 1.0)),
    }

    # 4. Per-Car Spatial Propagation Metrics (8 Cars)
    car_vib_diffs = []
    car_vib_ratios = []
    car_shk_diffs = []
    car_max_v1 = []
    car_max_v2 = []
    cars_s1_dominant = 0
    cars_s2_dominant = 0

    for c in range(1, 9):
        c_s1_cols = [col for col in s1_vib_cols if f'car {c}' in col]
        c_s2_cols = [col for col in s2_vib_cols if f'car {c}' in col]
        c_k1_cols = [col for col in s1_shk_cols if f'car {c}' in col]
        c_k2_cols = [col for col in s2_shk_cols if f'car {c}' in col]

        v1_c = np.sqrt(np.mean(df[c_s1_cols].values**2, axis=0))
        v2_c = np.sqrt(np.mean(df[c_s2_cols].values**2, axis=0))
        k1_c = np.sqrt(np.mean(df[c_k1_cols].values**2, axis=0))
        k2_c = np.sqrt(np.mean(df[c_k2_cols].values**2, axis=0))

        c_diff = float(np.mean(v1_c) - np.mean(v2_c))
        c_ratio = float(np.mean(v1_c) / (np.mean(v2_c) + 1e-6))
        c_shk_diff = float(np.mean(k1_c) - np.mean(k2_c))

        feats[f'car_{c}_vib_diff'] = c_diff
        feats[f'car_{c}_vib_ratio'] = c_ratio
        feats[f'car_{c}_shk_diff'] = c_shk_diff
        feats[f'car_{c}_v1_max'] = float(np.max(v1_c))
        feats[f'car_{c}_v2_max'] = float(np.max(v2_c))

        car_vib_diffs.append(c_diff)
        car_vib_ratios.append(c_ratio)
        car_shk_diffs.append(c_shk_diff)
        car_max_v1.append(float(np.max(v1_c)))
        car_max_v2.append(float(np.max(v2_c)))

        if c_diff > 0.04:
            cars_s1_dominant += 1
        elif c_diff < -0.04:
            cars_s2_dominant += 1

    feats['cars_s1_dominant'] = cars_s1_dominant
    feats['cars_s2_dominant'] = cars_s2_dominant
    feats['car_diffs_mean'] = float(np.mean(car_vib_diffs))
    feats['car_diffs_std'] = float(np.std(car_vib_diffs))
    feats['car_diffs_max'] = float(np.max(car_vib_diffs))
    feats['car_diffs_min'] = float(np.min(car_vib_diffs))
    feats['car_ratios_max'] = float(np.max(car_vib_ratios))
    feats['car_ratios_min'] = float(np.min(car_vib_ratios))
    feats['car_max_v1_top'] = float(np.max(car_max_v1))
    feats['car_max_v2_top'] = float(np.max(car_max_v2))
    feats['car_max_ratio_top'] = float(np.max(car_max_v1) / (np.max(car_max_v2) + 1e-6))
    feats['car_max_diff_top'] = float(np.max(car_max_v1) - np.max(car_max_v2))

    # 5. Multi-Band Welch Power Spectral Density (PSD)
    v1_avg = np.mean(v1, axis=1)
    v2_avg = np.mean(v2, axis=1)

    f_ax, pxx1 = signal.welch(v1_avg, fs=SAMPLING_RATE_HZ, nperseg=1024)
    _, pxx2 = signal.welch(v2_avg, fs=SAMPLING_RATE_HZ, nperseg=1024)

    bands = [
        ('low', 10, 100),
        ('mid1', 100, 300),
        ('corrug_classic', 300, 600),
        ('corrug_high', 600, 1200),
        ('impact', 1200, 3000),
    ]
    for bname, flow, fhigh in bands:
        mask = (f_ax >= flow) & (f_ax < fhigh)
        e1 = float(np.sum(pxx1[mask]))
        e2 = float(np.sum(pxx2[mask]))
        feats[f'psd_s1_{bname}'] = e1
        feats[f'psd_s2_{bname}'] = e2
        feats[f'psd_ratio_{bname}'] = float(e1 / (e2 + 1e-9))
        feats[f'psd_diff_{bname}'] = float(e1 - e2)
        feats[f'psd_sum_{bname}'] = float(e1 + e2)

    # Spectral centroid and high-order kurtosis
    feats['peak_freq_s1'] = float(f_ax[np.argmax(pxx1)])
    feats['peak_freq_s2'] = float(f_ax[np.argmax(pxx2)])
    feats['spec_centroid_s1'] = float(np.sum(f_ax * pxx1) / (np.sum(pxx1) + 1e-9))
    feats['spec_centroid_s2'] = float(np.sum(f_ax * pxx2) / (np.sum(pxx2) + 1e-9))
    feats['spec_centroid_diff'] = float(feats['spec_centroid_s1'] - feats['spec_centroid_s2'])

    feats['s1_kurtosis'] = float(stats.kurtosis(v1_avg))
    feats['s2_kurtosis'] = float(stats.kurtosis(v2_avg))
    feats['kurtosis_diff'] = float(feats['s1_kurtosis'] - feats['s2_kurtosis'])

    return feats


def extract_features_from_file(file_path: str) -> Dict[str, Any]:
    """Reads a CSV recording file and extracts its feature vector."""
    filename = os.path.basename(file_path)
    df = pd.read_csv(file_path)
    return extract_features_from_df(df, filename=filename)
