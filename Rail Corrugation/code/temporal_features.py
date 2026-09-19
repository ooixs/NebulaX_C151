"""Recording-local variation in channel-pooled spectral power.

Inputs are the existing 2.5 kHz, own-side recordings. Five consecutive windows
describe variation within each recording; this does not infer travel delays or
assert that different cars observe an identical track location simultaneously.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch


BANDS = ((20, 100), (100, 300), (300, 600), (600, 1200))


def _summaries(values):
    """Summarize an eight-car by five-window matrix without losing its axes."""
    return dict(
        mean=float(values.mean()),
        std=float(values.std()),
        p90=float(np.percentile(values, 90)),
        max=float(values.max()),
        min=float(values.min()),
        car_temporal_std_mean=float(values.std(axis=1).mean()),
        car_peak_excess_mean=float((values.max(axis=1) - values.mean(axis=1)).mean()),
        window_car_mean_std=float(values.mean(axis=0).std()),
    )


def temporal_features(raw):
    """Return 80 float32 features per side from an array shaped (2N, 32, 2500).

    Consecutive pairs belong to one file. Each side has four channels per car,
    ordered across eight cars, and 2,500 consecutive time samples. No statistics
    are fitted across files, so extracting files together or separately agrees.
    """
    values = np.asarray(raw)
    if values.ndim != 3 or values.shape[1:] != (32, 2500) or not len(values) or len(values) % 2:
        raise ValueError("Expected nonempty paired recordings shaped (2N, 32, 2500)")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Recordings must contain real numeric values")
    if not np.isfinite(values).all():
        raise ValueError("Recordings must contain only finite values")

    rows = []
    # Process one file at a time to bound temporary FFT memory.
    for index in range(0, len(values), 2):
        windows = values[index:index + 2].astype(np.float64).reshape(2, 8, 4, 5, 500)
        frequencies, powers = welch(windows, fs=2500, nperseg=500, axis=-1)
        car_powers = powers.mean(axis=2)
        sides = [{}, {}]
        for low, high in BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            log_power = np.log10(np.maximum(car_powers[..., mask].sum(axis=-1), 1e-15))
            for own in (0, 1):
                delta = log_power[own] - log_power[1 - own]
                prefix = f"temporal_{low}_{high}"
                for source, matrix in [("own", log_power[own]), ("relative", delta)]:
                    for name, value in _summaries(matrix).items():
                        sides[own][f"{prefix}_{source}_{name}"] = value
                positive = delta > 0
                window_fraction = positive.mean(axis=0)
                sides[own].update({
                    f"{prefix}_relative_positive_fraction": float(positive.mean()),
                    f"{prefix}_window_positive_car_fraction_mean": float(window_fraction.mean()),
                    f"{prefix}_window_positive_car_fraction_std": float(window_fraction.std()),
                    f"{prefix}_window_positive_car_fraction_max": float(window_fraction.max()),
                })
        rows.extend(sides)
    result = pd.DataFrame(rows).astype(np.float32)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Recording magnitude produced non-finite spectral features")
    return result
