"""Recording-local representations; no labels or test-batch statistics are used."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch

from rail_corrugation.pipeline import FS, FREQ_BANDS, speed
from rail_corrugation.sg_features import extract_features_from_df


def extract(path):
    df = pd.read_csv(path)
    a = df.to_numpy(float)
    assert a.shape == (10000, 129), a.shape
    vib = a[:, 1::2]
    v = speed(a[:, 0])
    # Exact SG implementation, canonicalized to each candidate side. Swapping
    # values, not headers, lets one shared detector learn both orientations.
    swapped = a.copy()
    swapped[:, 1:] = a[:, 1:].reshape(len(a), 32, 2, 2)[:, :, ::-1, :].reshape(len(a), 128)
    sg = []
    for data in (df, pd.DataFrame(swapped, columns=df.columns)):
        row = extract_features_from_df(data)
        row.pop('filename')
        sg.append(row)
    f, psd = welch(vib, fs=FS, nperseg=1024, axis=0)
    # Average *powers*, avoiding phase cancellation from averaged waveforms.
    car = psd.reshape(len(f), 8, 4, 2).mean(axis=2)
    features = []
    for own in (0, 1):
        other = 1-own
        row = {'speed': v}
        for lo, hi in FREQ_BANDS:
            power = car[(f >= lo) & (f < hi)].sum(axis=0) + 1e-15
            log_power = np.log10(power)
            delta = log_power[:, own] - log_power[:, other]
            for name, values in [('power', log_power[:, own]), ('bilateral', delta)]:
                for stat, val in [('mean', values.mean()), ('std', values.std()),
                                  ('min', values.min()), ('max', values.max()),
                                  ('median', np.median(values))]:
                    row[f'{lo}_{hi}_{name}_{stat}'] = float(val)
            row[f'{lo}_{hi}_positive_fraction'] = float((delta > 0).mean())
            row[f'{lo}_{hi}_adjacent_difference'] = float(np.abs(np.diff(log_power[:, own])).mean())
        mask = (f >= 20) & (f < 2000)
        spectra = car[mask, :, own]
        norms = np.linalg.norm(spectra, axis=0) + 1e-15
        sim = (spectra.T @ spectra) / (norms[:, None] * norms[None, :])
        row['spectral_cosine_mean'] = float(sim[np.triu_indices(8, 1)].mean())
        peaks = f[mask][spectra.argmax(axis=0)]
        row['peak_frequency_std'] = float(peaks.std())
        row['peak_frequency_median'] = float(np.median(peaks))
        wavelengths = v / np.maximum(peaks, 1)
        row['wavelength_std'] = float(wavelengths.std())
        row['wavelength_median'] = float(np.median(wavelengths))
        features.append(row)
    # Anti-aliased 2.5 kHz representation; 32 vibration channels per side.
    down = resample_poly(vib, 1, 4, axis=0).astype(np.float32)
    raw = np.stack([down[:, own::2].T for own in (0, 1)])
    return pd.DataFrame(sg).astype(np.float32), pd.DataFrame(features).astype(np.float32), raw


def normalized_raw(raw):
    centered = raw - raw.mean(axis=-1, keepdims=True)
    return centered / np.maximum(centered.std(axis=-1, keepdims=True), 1e-6)
