"""Canonical grouping and temporal sensitivity without fitting any model."""
import numpy as np
import pandas as pd
import pytest

from rail_corrugation.temporal_features import temporal_features


def sample_raw(files=1):
    raw = np.random.default_rng(27).normal(size=(2 * files, 32, 2500)).astype(np.float32)
    raw[::2] *= 2
    return raw


def test_side_swap_exchanges_feature_rows_and_preserves_source():
    raw = sample_raw(2)
    original = raw.copy()
    features = temporal_features(raw)
    swapped = raw.reshape(2, 2, 32, 2500)[:, ::-1].reshape(raw.shape)
    expected = features.iloc[[1, 0, 3, 2]].reset_index(drop=True)
    pd.testing.assert_frame_equal(temporal_features(swapped), expected)
    np.testing.assert_array_equal(raw, original)
    assert features.shape == (4, 80)
    assert features.columns.is_unique
    assert set(features.dtypes) == {np.dtype("float32")}
    assert np.isfinite(features.to_numpy()).all()


def test_single_window_burst_has_more_temporal_variation_than_stationary_signal():
    times = np.arange(2500) / 2500
    wave = np.sin(2 * np.pi * 400 * times)
    stationary = np.broadcast_to(wave, (2, 32, 2500)).copy()
    burst = stationary.copy()
    burst[0] = 0
    burst[0, :, 1000:1500] = wave[1000:1500] * 5
    stable_features = temporal_features(stationary)
    burst_features = temporal_features(burst)
    prefix = "temporal_300_600_own_"
    for stat in ["car_temporal_std_mean", "car_peak_excess_mean", "window_car_mean_std"]:
        assert stable_features.loc[0, prefix + stat] < 1e-10
        assert burst_features.loc[0, prefix + stat] > 1
    # Side II's own-side statistics are unaffected by a Side I burst.
    own_columns = [name for name in stable_features if "_own_" in name]
    pd.testing.assert_series_equal(stable_features.loc[1, own_columns], burst_features.loc[1, own_columns])


def test_file_grouping_is_independent_of_other_recordings():
    raw = sample_raw(2)
    together = temporal_features(raw)
    separate = pd.concat([temporal_features(raw[:2]), temporal_features(raw[2:])], ignore_index=True)
    pd.testing.assert_frame_equal(together, separate)
    changed = raw.copy()
    changed[2:] *= 100
    pd.testing.assert_frame_equal(temporal_features(changed).iloc[:2], together.iloc[:2])


def test_zero_recording_has_finite_floor_and_no_side_dominance():
    features = temporal_features(np.zeros((2, 32, 2500), dtype=np.float32))
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "temporal_300_600_own_mean"] == -15
    assert features.loc[0, "temporal_300_600_relative_mean"] == 0
    assert features.loc[0, "temporal_300_600_relative_positive_fraction"] == 0
    pd.testing.assert_series_equal(features.iloc[0], features.iloc[1], check_names=False)


def test_car_and_window_axes_remain_distinct():
    times = np.arange(2500) / 2500
    wave = np.sin(2 * np.pi * 400 * times)
    raw = np.broadcast_to(wave, (2, 32, 2500)).copy()
    # Two cars on Side I are stronger throughout all five windows.
    raw[0, :8] *= 2
    features = temporal_features(raw)
    prefix = "temporal_300_600_"
    assert features.loc[0, prefix + "relative_positive_fraction"] == 0.25
    assert features.loc[0, prefix + "window_positive_car_fraction_mean"] == 0.25
    assert features.loc[0, prefix + "window_positive_car_fraction_max"] == 0.25
    assert features.loc[0, prefix + "window_positive_car_fraction_std"] == 0
    assert features.loc[0, prefix + "own_car_temporal_std_mean"] < 1e-10


@pytest.mark.parametrize("raw", [
    np.zeros((0, 32, 2500)),
    np.zeros((3, 32, 2500)),
    np.zeros((2, 31, 2500)),
    np.zeros((2, 32, 2499)),
    np.zeros((2, 32)),
])
def test_invalid_shapes_are_rejected(raw):
    with pytest.raises(ValueError, match="paired recordings"):
        temporal_features(raw)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_samples_are_rejected(invalid):
    raw = sample_raw()
    raw[1, 2, 3] = invalid
    with pytest.raises(ValueError, match="finite"):
        temporal_features(raw)
