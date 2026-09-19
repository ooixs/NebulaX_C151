"""Fold-local file weighting and isolated pooled-SG feature integration."""
import numpy as np
import pandas as pd
import pytest

from rail_corrugation import experiment_followup as experiment
from rail_corrugation import predict_candidate


def test_file_weights_balance_original_classes_and_have_unit_mean():
    labels = np.array(["Normal"] * 5 + ["Side I"] * 2 + ["Side II"])
    weights = experiment.file_weights(labels)
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.mean(), 1)
    np.testing.assert_allclose([weights[labels == label].sum() for label in np.unique(labels)], [8 / 3] * 3)
    np.testing.assert_allclose(weights, [8 / 15] * 5 + [4 / 3] * 2 + [8 / 3])


@pytest.mark.parametrize("labels,expected", [
    (["Normal", "Normal", "Side I"], [0.75, 0.75, 1.5]),
    (["Side II", "Side II"], [1, 1]),
])
def test_file_weights_handle_classes_absent_from_training_fold(labels, expected):
    np.testing.assert_allclose(experiment.file_weights(labels), expected)


class RecordingModel:
    classes_ = np.array([1, 0])

    def fit(self, design, labels, sample_weight=None):
        self.fit_design = np.array(design, copy=True)
        self.fit_labels = np.array(labels, copy=True)
        self.fit_weights = None if sample_weight is None else np.array(sample_weight, copy=True)
        return self

    def predict_proba(self, design):
        self.predict_design = np.array(design, copy=True)
        positive = 1 / (1 + np.exp(-np.asarray(design).mean(axis=1) / 100))
        return np.stack([positive, 1 - positive], axis=1)


@pytest.mark.parametrize("name", list(experiment.CONFIGS))
def test_fold_fit_uses_local_file_weights_and_correct_sg_representation(monkeypatch, name):
    campaign = experiment.Campaign.__new__(experiment.Campaign)
    campaign.jobs = 1
    campaign.y = np.array(["Normal", "Normal", "Normal", "Normal", "Side I", "Side II"])
    campaign.sy = (campaign.y[:, None] == np.array(["Side I", "Side II"])).astype(int).ravel()
    sg = np.arange(36, dtype=float).reshape(12, 3)
    pooled = sg + 1000
    legacy = np.arange(24, dtype=float).reshape(12, 2) + 100
    campaign.data = {"sg": sg}
    campaign.pooled = pooled
    reference = object()
    table_calls = []

    def table(train):
        table_calls.append(np.array(train, copy=True))
        return reference, ["legacy_a", "legacy_b"], legacy

    campaign.table = table
    factory_calls = []
    model = RecordingModel()

    def factory(config, seed, jobs):
        factory_calls.append((dict(config), seed, jobs))
        return model

    monkeypatch.setattr(experiment.base, "research_model", factory)
    cfg = experiment.CONFIGS[name]
    train = np.array([4, 0, 5, 2])
    testdata = {"sg": sg[:4] + 5000, "sg_pooled": pooled[:4] + 6000}
    testlegacy = legacy[:4] + 7000
    snapshots = [x.copy() for x in [sg, pooled, legacy, testdata["sg"], testdata["sg_pooled"], testlegacy]]
    probabilities, artifact = campaign.fit_predict(
        cfg, train, np.array([1]), seed=73, final=True, testdata=testdata, testlegacy=testlegacy
    )
    selected = pooled if cfg.get("pooled") else sg
    selected_test = testdata["sg_pooled" if cfg.get("pooled") else "sg"]
    expected_train = selected[[8, 9, 0, 1, 10, 11, 4, 5]]
    expected_test = selected_test[[2, 3]]
    if "legacy" in cfg["rep"]:
        expected_train = np.concatenate([legacy[[8, 9, 0, 1, 10, 11, 4, 5]], expected_train], axis=1)
        expected_test = np.concatenate([testlegacy[[2, 3]], expected_test], axis=1)
        np.testing.assert_array_equal(table_calls, [train])
        assert artifact["ref"] is reference
        assert artifact["cols"] == ["legacy_a", "legacy_b"]
    else:
        assert table_calls == []
    np.testing.assert_array_equal(model.fit_design, expected_train)
    np.testing.assert_array_equal(model.predict_design, expected_test)
    np.testing.assert_array_equal(model.fit_labels, [1, 0, 0, 0, 0, 1, 0, 0])
    assert factory_calls == [(cfg, 73, 1)]
    assert artifact["config"] == cfg
    assert artifact["model"] is model
    if cfg.get("file_balanced"):
        # Two Normal files and one of each fault are in this fold. Using all
        # four Normal files from the full dataset would yield different weights.
        expected_weights = np.repeat([4 / 3, 2 / 3, 4 / 3, 2 / 3], 2)
        np.testing.assert_allclose(model.fit_weights, expected_weights)
        global_weights = np.repeat(experiment.file_weights(campaign.y)[train], 2)
        assert not np.allclose(model.fit_weights, global_weights)
        assert factory_calls[0][0]["class_weight"] is None
    else:
        assert model.fit_weights is None
    expected_probabilities = 1 / (1 + np.exp(-expected_test.mean(axis=1) / 100))
    np.testing.assert_allclose(probabilities, expected_probabilities.reshape(1, 2))
    for actual, unchanged in zip([sg, pooled, legacy, testdata["sg"], testdata["sg_pooled"], testlegacy], snapshots):
        np.testing.assert_array_equal(actual, unchanged)


def test_nested_existing_and_pooled_artifacts_receive_their_own_features(monkeypatch):
    sg = pd.DataFrame([[1, 2], [3, 4]], columns=["sg_a", "sg_b"])
    pooled = pd.DataFrame([[21, 42], [63, 84]], columns=sg.columns)
    legacy = pd.DataFrame([[5, 6], [7, 8]], columns=["legacy_a", "legacy_b"])
    originals = [frame.copy(deep=True) for frame in [sg, pooled, legacy]]
    models = [RecordingModel() for _ in range(3)]

    class Reference:
        def excess(self, channels, speed):
            return channels

    monkeypatch.setattr(predict_candidate, "aggregate", lambda *args, **kwargs: "aggregated")
    monkeypatch.setattr(predict_candidate, "side_relative_rows", lambda *args, **kwargs: legacy.to_dict("records"))
    old_sg = dict(config=dict(rep="sg"), model=models[0])
    old_legacy = dict(config=dict(rep="legacy_sg"), model=models[1],
                      ref=Reference(), cols=["legacy_b", "legacy_a"])
    pooled_legacy = dict(config=dict(rep="legacy_sg", pooled=True), model=models[2],
                         ref=Reference(), cols=["legacy_b", "legacy_a"])
    old_ensemble = dict(components=[old_legacy, old_sg], weight=0.75)
    nested = dict(components=[pooled_legacy, old_ensemble], weight=0.25)
    inferred = predict_candidate.component_probability(nested, (sg, None, None, pooled), None, 4.0)

    legacy_values = legacy[["legacy_b", "legacy_a"]].to_numpy()
    designs = [sg.to_numpy(), np.concatenate([legacy_values, sg.to_numpy()], axis=1),
               np.concatenate([legacy_values, pooled.to_numpy()], axis=1)]
    for model, expected in zip(models, designs):
        np.testing.assert_array_equal(model.predict_design, expected)
    probabilities = [1 / (1 + np.exp(-design.mean(axis=1) / 100)) for design in designs]
    expected = 0.25 * probabilities[2] + 0.75 * (0.75 * probabilities[1] + 0.25 * probabilities[0])
    np.testing.assert_allclose(inferred, expected)
    for actual, original in zip([sg, pooled, legacy], originals):
        pd.testing.assert_frame_equal(actual, original)
