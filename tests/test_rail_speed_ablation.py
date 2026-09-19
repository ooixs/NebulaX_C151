"""Feature-selection and split-isolation checks without fitting any forests."""

import numpy as np
import pandas as pd
import pytest

from rail_corrugation import experiment_speed_ablation as experiment
from rail_corrugation import predict_candidate
from rail_corrugation.pipeline import NormalReference, aggregate, channel_features, side_relative_rows
from rail_corrugation.sg_features import extract_features_from_df
from rail_corrugation.train import research_columns


def test_masks_match_real_feature_schema_and_preserve_order():
    rng = np.random.default_rng(17)
    values = rng.normal(size=(1024, 129))
    values[:, 0] = np.arange(1024) // 20 % 2
    headers = ["Rotating speed"] + [
        f"{kind} of bearing in position {position} of car {car}"
        for car in range(1, 9)
        for position in range(1, 9)
        for kind in ["Vibration", "Shock"]
    ]
    sg = list(extract_features_from_df(pd.DataFrame(values, columns=headers)))
    sg.remove("filename")
    ct = channel_features(values[:, 1::2], values[:, 2::2], 4.0)
    reference = NormalReference().fit([ct], [4.0], ["Normal"])
    table = pd.DataFrame(side_relative_rows(aggregate(reference.excess(ct, 4.0), 4.0, paired=True), engineered=True))
    legacy = research_columns(table.columns, "legacy")

    assert len(sg) == 121
    assert len(legacy) == 324
    sg_kept = [sg[i] for i in experiment.kept_indices(sg, experiment.SG_DROP)]
    legacy_kept = [legacy[i] for i in experiment.kept_indices(legacy, experiment.LEGACY_DROP)]
    assert len(sg_kept) == 115
    assert len(legacy_kept) == 322
    assert sg_kept == [name for name in sg if "speed" not in name and name != "is_moving"]
    assert legacy_kept == [name for name in legacy if name not in ("speed", "speed_bin")]
    assert "own_lb_25_50_mean" in legacy_kept
    assert "own_ex_log_rms_mean" in legacy_kept
    assert "own_dom_lambda_mean" in legacy_kept
    assert len(legacy) + len(sg_kept) == 439
    assert len(legacy_kept) + len(sg_kept) == 437


def test_mask_rejects_missing_expected_input():
    with pytest.raises(ValueError, match="speed_bin"):
        experiment.kept_indices(["speed", "amplitude"], experiment.LEGACY_DROP)


class RecordingModel:
    classes_ = np.array([1, 0])

    def fit(self, design, labels):
        self.fit_design = np.array(design, copy=True)
        self.fit_labels = np.array(labels, copy=True)
        return self

    def predict_proba(self, design):
        self.prediction_design = np.array(design, copy=True)
        return np.tile([0.7, 0.3], (len(design), 1))


class ReferenceStub:
    def excess(self, channels, speed):
        return channels


@pytest.mark.parametrize("configuration", ["legacy_sg_drop_sg", "legacy_sg_drop_both", "sg_drop_sg"])
def test_training_and_artifact_inference_use_identical_masks(monkeypatch, configuration):
    sg_columns = ["s1_vib_rms_mean", "speed_kmh", "is_moving", "s2_vib_speed_norm", "peak_freq_s1",
                  "speed_mps", "s1_vib_speed_norm", "all_vib_speed_norm"]
    legacy_columns = ["own_lb_25_50_mean", "speed", "own_ex_log_rms_mean", "speed_bin"]
    sg = np.arange(6 * len(sg_columns), dtype=float).reshape(6, -1)
    legacy = np.arange(6 * len(legacy_columns), dtype=float).reshape(6, -1) + 100
    test_sg = sg[2:4] + 1000
    test_legacy = legacy[2:4] + 2000
    campaign = experiment.SpeedCampaign.__new__(experiment.SpeedCampaign)
    campaign.jobs = 1
    campaign.data = {"sg": sg}
    campaign.sg_columns = sg_columns
    campaign.sy = np.array([0, 0, 1, 0, 0, 1])
    campaign.table = lambda indices: (ReferenceStub(), legacy_columns, legacy)
    model = RecordingModel()
    monkeypatch.setattr(experiment.base, "research_model", lambda config, seed, jobs: model)
    cfg = experiment.CONFIGS[configuration]
    probabilities, artifact = campaign.fit_predict(
        cfg, np.array([0, 2]), np.array([0]), 7, final=True,
        testdata={"sg": test_sg}, testlegacy=test_legacy,
    )

    expected_train = [sg[[0, 1, 4, 5]][:, [0, 4]]]
    expected_test = [test_sg[:, [0, 4]]]
    if "legacy" in cfg["rep"]:
        legacy_indices = [0, 2] if cfg.get("drop_legacy") else [0, 1, 2, 3]
        expected_train.insert(0, legacy[[0, 1, 4, 5]][:, legacy_indices])
        expected_test.insert(0, test_legacy[:, legacy_indices])
        assert artifact["cols"] == [legacy_columns[i] for i in legacy_indices]
    expected_train = np.concatenate(expected_train, axis=1)
    expected_test = np.concatenate(expected_test, axis=1)
    np.testing.assert_array_equal(model.fit_design, expected_train)
    np.testing.assert_array_equal(model.fit_labels, [0, 0, 0, 1])
    np.testing.assert_array_equal(model.prediction_design, expected_test)
    np.testing.assert_array_equal(probabilities, [[0.7, 0.7]])
    assert artifact["sg_indices"] == [0, 4]

    monkeypatch.setattr(predict_candidate, "aggregate", lambda *args, **kwargs: {})
    monkeypatch.setattr(predict_candidate, "side_relative_rows",
                        lambda *args, **kwargs: pd.DataFrame(test_legacy, columns=legacy_columns).to_dict("records"))
    data = (pd.DataFrame(test_sg, columns=sg_columns), None, None)
    inferred = predict_candidate.component_probability(artifact, data, None, 4.0)
    np.testing.assert_array_equal(model.prediction_design, expected_test)
    np.testing.assert_array_equal(inferred, [0.7, 0.7])


def test_legacy_artifact_without_sg_mask_retains_all_sg_columns():
    model = RecordingModel()
    sg = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["amplitude", "speed_mps", "peak_frequency"])
    artifact = {"config": {"rep": "sg"}, "model": model}
    predict_candidate.component_probability(artifact, (sg, None, None), None, 4.0)
    np.testing.assert_array_equal(model.prediction_design, sg.to_numpy())


def test_speed_protocol_keeps_cross_boundary_duplicate_family_together():
    # The repeated recording family straddles the 9 m/s boundary, so grouping
    # must precede speed holdout assignment and remain intact in inner splits.
    speeds = np.array([1, 4, 7, 8.9, 9.1, 10, 11, 12.5, 13, 14, 16, 18, 20, 22], dtype=float)
    groups = np.array([f"file_{i}" for i in range(len(speeds))], dtype=object)
    groups[[3, 4]] = "same_recording"
    folds = experiment.speed_protocol(speeds, groups)
    assert [fold["speed_bin"] for fold in folds] == [2, 3, 4, 5]
    assert sorted(np.concatenate([fold["validation"] for fold in folds])) == list(range(len(speeds)))
    for fold in folds:
        for train, validation in [(fold["train"], fold["validation"])] + fold["inner"]:
            assert set(groups[train]).isdisjoint(groups[validation])
            assert bool(3 in validation) == bool(4 in validation)
        assert sorted(np.concatenate([validation for _, validation in fold["inner"]])) == sorted(fold["train"])
        assert all(len(train) and len(validation) for train, validation in fold["inner"])
    family_fold = next(fold for fold in folds if 3 in fold["validation"])
    assert family_fold["speed_bin"] == 3
    np.testing.assert_array_equal(experiment.base.rows([3, 4]), [6, 7, 8, 9])
