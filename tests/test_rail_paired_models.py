"""Fault-label semantics and side exchange behavior of paired architectures."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from rail_corrugation import paired_models as paired
from rail_corrugation import predict_candidate


def sample_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(36, 4))
    y = np.tile(["Normal", "Side I", "Side II"], 6)
    own = (y[:, None] == np.array(["Side I", "Side II"])).ravel()
    x[:, 0] += own * 4
    return x, y


@pytest.mark.parametrize("architecture", ["multiclass", "hierarchical"])
def test_side_swap_exchanges_probabilities_and_fit_is_deterministic(architecture):
    x, y = sample_data()
    cfg = dict(architecture=architecture, trees=11, leaf=1)
    fitted = paired.fit_paired_model(x, y, cfg, seed=17, jobs=1)
    probabilities = paired.paired_probability(fitted, x)
    swapped = x.reshape(-1, 2, x.shape[1])[:, ::-1].reshape(x.shape)
    np.testing.assert_allclose(paired.paired_probability(fitted, swapped), probabilities[:, ::-1], atol=1e-15)
    repeated = paired.fit_paired_model(x, y, cfg, seed=17, jobs=1)
    np.testing.assert_array_equal(paired.paired_probability(repeated, x), probabilities)
    assert probabilities.shape == (len(y), 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.all(probabilities.sum(axis=1) <= 1 + 1e-15)
    # This synthetic signal makes both actual fault sides identifiable.
    assert probabilities[y == "Side I", 0].mean() > probabilities[y == "Side I", 1].mean()
    assert probabilities[y == "Side II", 1].mean() > probabilities[y == "Side II", 0].mean()


def test_multiclass_encodes_own_and_opposite_faults(monkeypatch):
    captured = []

    def recorder(design, labels, config, seed, jobs):
        captured.append((design.copy(), labels.copy()))
        return DummyClassifier(strategy="prior").fit(design, labels)

    monkeypatch.setattr(paired, "_fit_classifier", recorder)
    x = np.arange(18).reshape(6, 3)
    paired.fit_paired_model(x, ["Normal", "Side I", "Side II"],
                            dict(architecture="multiclass", trees=3), seed=1, jobs=1)
    np.testing.assert_array_equal(captured[0][0], x)
    np.testing.assert_array_equal(captured[0][1], [0, 0, 1, 2, 2, 1])


def test_hierarchical_detector_is_symmetric_and_localizer_uses_only_faults(monkeypatch):
    captured = []

    def recorder(design, labels, config, seed, jobs):
        captured.append((design.copy(), labels.copy()))
        return DummyClassifier(strategy="prior").fit(design, labels)

    monkeypatch.setattr(paired, "_fit_classifier", recorder)
    x = np.array([[1, 5], [3, 1], [2, 8], [4, 2], [6, 4], [8, 6]])
    artifact = paired.fit_paired_model(x, ["Normal", "Side I", "Side II"],
                                      dict(architecture="hierarchical", trees=3), seed=1, jobs=1)
    np.testing.assert_array_equal(captured[0][0], [[2, 3, 2, 4], [3, 5, 2, 6], [7, 5, 2, 2]])
    np.testing.assert_array_equal(captured[0][1], [0, 1, 1])
    np.testing.assert_array_equal(captured[1][0], x[2:])
    np.testing.assert_array_equal(captured[1][1], [1, 0, 0, 1])
    np.testing.assert_allclose(paired.paired_probability(artifact, x), np.full((3, 2), 1 / 3))


class ReorderedClasses:
    classes_ = np.array([2, 0, 1])

    def predict_proba(self, values):
        return np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])


def test_multiclass_looks_up_classes_instead_of_assuming_column_order():
    artifact = dict(config=dict(architecture="multiclass"), n_features=2, model=ReorderedClasses())
    np.testing.assert_allclose(paired.paired_probability(artifact, np.ones((2, 2))), [[0.55, 0.15]])


@pytest.mark.parametrize("architecture", ["multiclass", "hierarchical"])
def test_all_normal_training_has_zero_fault_probability(architecture):
    x = np.ones((4, 2))
    artifact = paired.fit_paired_model(x, ["Normal", "Normal"],
                                      dict(architecture=architecture, trees=3), seed=1, jobs=1)
    np.testing.assert_array_equal(paired.paired_probability(artifact, x), np.zeros((2, 2)))


@pytest.mark.parametrize("architecture", ["multiclass", "hierarchical"])
def test_all_fault_training_retains_unit_fault_probability(architecture):
    x = np.array([[1, 0], [0, 1], [0, 2], [2, 0]])
    artifact = paired.fit_paired_model(x, ["Side I", "Side II"],
                                      dict(architecture=architecture, trees=3), seed=1, jobs=1)
    np.testing.assert_allclose(paired.paired_probability(artifact, x).sum(axis=1), [1, 1])


@pytest.mark.parametrize("values,labels,message", [
    (np.empty((0, 2)), [], "nonempty"),
    (np.ones((3, 2)), ["Normal"], "two consecutive"),
    (np.ones((2, 0)), ["Normal"], "nonempty"),
    ([[np.nan], [1]], ["Normal"], "finite"),
    (np.ones((2, 2)), ["Side III"], "File labels"),
    (np.ones((2, 2)), ["Normal", "Normal"], "one file label"),
])
def test_invalid_training_input_is_rejected(values, labels, message):
    with pytest.raises(ValueError, match=message):
        paired.fit_paired_model(values, labels, dict(architecture="multiclass", trees=3), seed=1, jobs=1)


@pytest.mark.parametrize("architecture", ["multiclass", "hierarchical"])
def test_candidate_inference_preserves_paired_layout_and_sg_selection(architecture):
    x, y = sample_data()
    keep = [0, 2, 3]
    cfg = dict(rep="sg", architecture=architecture, trees=7)
    model = paired.fit_paired_model(x[:, keep], y, cfg, seed=23, jobs=1)
    artifact = dict(config=cfg, paired_model=model, sg_indices=keep)
    data = (pd.DataFrame(x, columns=["own", "unused", "f2", "f3"]), None, None)
    inferred = predict_candidate.component_probability(artifact, data, None, 4.0)
    expected = paired.paired_probability(model, x[:, keep]).ravel()
    assert inferred.shape == (2 * len(y),)
    np.testing.assert_array_equal(inferred, expected)


def test_recursive_candidate_blend_matches_manual_component_arithmetic():
    x, y = sample_data()
    own = (y[:, None] == np.array(["Side I", "Side II"])).ravel().astype(int)
    old_models = [RandomForestClassifier(n_estimators=7, min_samples_leaf=2, random_state=seed).fit(x, own)
                  for seed in [11, 37]]
    old_components = [dict(config=dict(rep="sg"), model=model) for model in old_models]
    old_ensemble = dict(components=old_components, weight=0.75)
    cfg = dict(rep="sg", architecture="multiclass", trees=7)
    paired_model = paired.fit_paired_model(x, y, cfg, seed=17, jobs=1)
    paired_artifact = dict(config=cfg, paired_model=paired_model)
    nested = dict(components=[paired_artifact, old_ensemble], weight=0.25)
    data = (pd.DataFrame(x), None, None)

    old_probabilities = [model.predict_proba(x)[:, list(model.classes_).index(1)] for model in old_models]
    old_expected = 0.75 * old_probabilities[0] + 0.25 * old_probabilities[1]
    expected = 0.25 * paired.paired_probability(paired_model, x).ravel() + 0.75 * old_expected
    actual = predict_candidate.component_probability(nested, data, None, 4.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_existing_legacy_sg_artifact_retains_feature_order_and_probability(monkeypatch):
    x, y = sample_data()
    legacy = pd.DataFrame({"legacy_a": x[:, 0] + 3, "legacy_b": x[:, 2] - 2})
    # Saved column order may differ from the table's natural order.
    cols = ["legacy_b", "legacy_a"]
    design = np.concatenate([legacy[cols].to_numpy(), x], axis=1)
    own = (y[:, None] == np.array(["Side I", "Side II"])).ravel().astype(int)
    model = RandomForestClassifier(n_estimators=7, random_state=11).fit(design, own)

    class Reference:
        def excess(self, channels, speed):
            assert channels == "channels"
            assert speed == 4.0
            return "excess"

    def aggregate(excess, speed, paired):
        assert (excess, speed, paired) == ("excess", 4.0, True)
        return "aggregate"

    def relative_rows(features, engineered):
        assert (features, engineered) == ("aggregate", True)
        return legacy.to_dict("records")

    monkeypatch.setattr(predict_candidate, "aggregate", aggregate)
    monkeypatch.setattr(predict_candidate, "side_relative_rows", relative_rows)
    artifact = dict(config=dict(rep="legacy_sg"), ref=Reference(), cols=cols, model=model)
    actual = predict_candidate.component_probability(artifact, (pd.DataFrame(x), None, None), "channels", 4.0)
    np.testing.assert_array_equal(actual, model.predict_proba(design)[:, list(model.classes_).index(1)])
