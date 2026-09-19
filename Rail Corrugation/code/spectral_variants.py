"""A narrow SG ablation: average channel powers instead of channel waveforms.

Only the 30 existing PSD-derived fields change. Pooling powers does not require
channels to be phase aligned or to observe the same track location at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch


BANDS = (
    ("low", 10, 100),
    ("mid1", 100, 300),
    ("corrug_classic", 300, 600),
    ("corrug_high", 600, 1200),
    ("impact", 1200, 3000),
)
PSD_COLUMNS = tuple(
    f"psd_{stat}_{band}" for band, _, _ in BANDS
    for stat in ("s1", "s2", "ratio", "diff", "sum")
) + ("peak_freq_s1", "peak_freq_s2", "spec_centroid_s1", "spec_centroid_s2", "spec_centroid_diff")


def pooled_sg_from_array(a, sg_rows):
    """Replace PSD fields in two canonical SG rows; return a float32 DataFrame.

    ``a`` has the original 129-channel recording order: pulse, then alternating
    vibration/shock channels in car and position order. ``sg_rows`` contains
    the existing Side I view followed by the existing Side II view.
    """
    values = np.asarray(a, dtype=float)
    if values.ndim != 2 or values.shape[1] != 129 or not len(values):
        raise ValueError("Expected a nonempty recording with 129 columns")
    if not np.isfinite(values).all():
        raise ValueError("Recording must contain only finite values")
    if not isinstance(sg_rows, pd.DataFrame) or sg_rows.shape != (2, 121):
        raise ValueError("Expected a DataFrame containing two canonical rows and 121 SG columns")
    if not sg_rows.columns.is_unique:
        raise ValueError("SG column names must be unique")
    missing = set(PSD_COLUMNS) - set(sg_rows.columns)
    if missing:
        raise ValueError(f"Missing SG spectral columns: {sorted(missing)}")

    frequencies, channel_powers = welch(values[:, 1::2], fs=10000, nperseg=1024, axis=0)
    side_powers = np.stack([channel_powers[:, side::2].mean(axis=1) for side in (0, 1)])
    replacements = []
    for own in (0, 1):
        first, second = side_powers[own], side_powers[1 - own]
        fields = {}
        for band, low, high in BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            e1, e2 = float(first[mask].sum()), float(second[mask].sum())
            fields.update({
                f"psd_s1_{band}": e1,
                f"psd_s2_{band}": e2,
                f"psd_ratio_{band}": e1 / (e2 + 1e-9),
                f"psd_diff_{band}": e1 - e2,
                f"psd_sum_{band}": e1 + e2,
            })
        c1 = float(np.sum(frequencies * first) / (first.sum() + 1e-9))
        c2 = float(np.sum(frequencies * second) / (second.sum() + 1e-9))
        fields.update(peak_freq_s1=float(frequencies[np.argmax(first)]),
                      peak_freq_s2=float(frequencies[np.argmax(second)]),
                      spec_centroid_s1=c1, spec_centroid_s2=c2, spec_centroid_diff=c1 - c2)
        replacements.append(fields)

    result = sg_rows.astype(np.float32).copy()
    columns = list(PSD_COLUMNS)
    result.loc[:, columns] = pd.DataFrame(replacements)[columns].to_numpy(dtype=np.float32)
    return result


def pooled_sg_rows(path, sg_rows):
    """Read a recording and apply the same pure spectral-pooling ablation."""
    return pooled_sg_from_array(pd.read_csv(path).to_numpy(float), sg_rows)
