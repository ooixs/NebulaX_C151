"""Physics and feature-boundary tests for the SG power-pooling ablation."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import welch

from rail_corrugation.sg_features import extract_features_from_df
from rail_corrugation.spectral_variants import PSD_COLUMNS, pooled_sg_from_array, pooled_sg_rows


HEADERS = ["Rotating speed"] + [
    f"{kind} of bearing in position {position} of car {car}"
    for car in range(1, 9) for position in range(1, 9) for kind in ["Vibration", "Shock"]
]


def swap_sides(values):
    result = values.copy()
    result[:, 1:] = values[:, 1:].reshape(len(values), 32, 2, 2)[:, :, ::-1, :].reshape(len(values), 128)
    return result


def canonical_rows(values):
    rows = []
    for data in [values, swap_sides(values)]:
        features = extract_features_from_df(pd.DataFrame(data, columns=HEADERS))
        features.pop("filename")
        rows.append(features)
    return pd.DataFrame(rows).astype(np.float32)


def recording():
    values = np.random.default_rng(13).normal(size=(2048, 129))
    values[:, 0] = np.arange(len(values)) // 20 % 2
    # Distinct side powers make side-order mistakes visible.
    values[:, 1::4] *= 3
    return values


def test_side_swap_exchanges_canonical_rows():
    values = recording()
    rows = canonical_rows(values)
    pooled = pooled_sg_from_array(values, rows)
    swapped = pooled_sg_from_array(swap_sides(values), canonical_rows(swap_sides(values)))
    pd.testing.assert_frame_equal(swapped, pooled.iloc[::-1].reset_index(drop=True))
    np.testing.assert_array_equal(pooled["psd_s1_corrug_classic"], pooled["psd_s2_corrug_classic"][::-1])
    assert pooled.iloc[0]["psd_ratio_corrug_classic"] > 1
    assert pooled.iloc[1]["psd_ratio_corrug_classic"] < 1


def test_only_thirty_psd_columns_change_and_input_is_not_mutated():
    values = recording()
    rows = canonical_rows(values)
    rows.index = ["first", "second"]
    original = rows.copy(deep=True)
    pooled = pooled_sg_from_array(values, rows)
    assert len(PSD_COLUMNS) == 30
    assert pooled.shape == (2, 121)
    assert list(pooled.columns) == list(rows.columns)
    assert list(pooled.index) == list(rows.index)
    assert set(pooled.dtypes) == {np.dtype("float32")}
    unchanged = [column for column in rows if column not in PSD_COLUMNS]
    pd.testing.assert_frame_equal(pooled[unchanged], rows[unchanged])
    pd.testing.assert_frame_equal(rows, original)
    assert not np.array_equal(pooled[list(PSD_COLUMNS)].to_numpy(), rows[list(PSD_COLUMNS)].to_numpy())


def test_mean_power_retains_antiphase_signal_energy():
    values = recording()
    rows = canonical_rows(values)
    times = np.arange(len(values)) / 10000
    wave = np.sin(2 * np.pi * 390.625 * times)
    # Equal positive/negative channels on Side I cancel in the waveform mean.
    values[:, 1::4] = wave[:, None] * np.tile([1, -1], 16)[None, :]
    values[:, 3::4] = 0
    pooled = pooled_sg_from_array(values, rows)
    frequencies, power = welch(wave, fs=10000, nperseg=1024)
    band = (frequencies >= 300) & (frequencies < 600)
    expected = power[band].sum()
    _, cancelled_power = welch(values[:, 1::4].mean(axis=1), fs=10000, nperseg=1024)
    assert cancelled_power[band].sum() == 0
    assert expected > 0
    np.testing.assert_allclose(pooled.iloc[0]["psd_s1_corrug_classic"], expected, rtol=1e-6)
    assert pooled.iloc[0]["psd_s2_corrug_classic"] == 0
    assert pooled.iloc[0]["peak_freq_s1"] == 390.625


def test_path_wrapper_matches_array_function(tmp_path):
    values = recording()
    rows = canonical_rows(values)
    path = tmp_path / "recording.csv"
    pd.DataFrame(values, columns=HEADERS).to_csv(path, index=False)
    pd.testing.assert_frame_equal(pooled_sg_rows(path, rows), pooled_sg_from_array(values, rows))


def test_missing_spectral_column_is_rejected():
    values = recording()
    rows = canonical_rows(values).rename(columns={"peak_freq_s1": "unexpected"})
    with pytest.raises(ValueError, match="Missing SG spectral columns"):
        pooled_sg_from_array(values, rows)
