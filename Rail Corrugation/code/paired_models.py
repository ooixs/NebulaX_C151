"""File-level fault models from paired, own-side-canonical feature rows.

Every file contributes two consecutive rows: Side I's view and Side II's
view.  Inference averages both views so exchanging the two rows exchanges
the two returned fault probabilities exactly.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def _paired_rows(x_rows):
    values = np.asarray(x_rows, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Expected a nonempty, two-dimensional feature matrix")
    if len(values) % 2:
        raise ValueError("Expected two consecutive feature rows per file")
    if not np.isfinite(values).all():
        raise ValueError("Feature rows must contain only finite values")
    return values


def _symmetric_design(values):
    first, second = values[::2], values[1::2]
    return np.concatenate((first / 2 + second / 2, np.abs(first - second)), axis=1)


def _fit_classifier(design, labels, config, seed, jobs):
    if len(np.unique(labels)) == 1:
        model = DummyClassifier(strategy="constant", constant=labels[0])
    else:
        model = RandomForestClassifier(
            n_estimators=config.get("trees", 800),
            min_samples_leaf=config.get("leaf", 1),
            max_features=config.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=jobs,
        )
    return model.fit(design, labels)


def _class_probability(model, design, label):
    matches = np.flatnonzero(np.asarray(model.classes_) == label)
    if not len(matches):
        return np.zeros(len(design), dtype=float)
    return model.predict_proba(design)[:, matches[0]]


def fit_paired_model(x_rows, y_filelabels, config, seed, jobs):
    """Fit a multiclass or hierarchical forest and return a serializable artifact.

    ``config['architecture']`` must be ``'multiclass'`` or ``'hierarchical'``.
    Forest options are ``trees`` (800), ``leaf`` (1), and ``max_features``
    (``'sqrt'``). Additional experiment metadata in config is preserved.
    """
    values = _paired_rows(x_rows)
    labels = np.asarray(y_filelabels)
    if labels.ndim != 1 or len(labels) * 2 != len(values):
        raise ValueError("Expected one file label for each pair of feature rows")
    if not np.isin(labels, ["Normal", "Side I", "Side II"]).all():
        raise ValueError("File labels must be Normal, Side I, or Side II")
    architecture = config.get("architecture")
    if architecture not in ("multiclass", "hierarchical"):
        raise ValueError("Architecture must be multiclass or hierarchical")
    if config.get("kind", "rf") != "rf":
        raise ValueError("Paired models currently support only kind='rf'")

    own_fault = labels[:, None] == np.array(["Side I", "Side II"])
    any_fault = labels != "Normal"
    artifact = dict(config=dict(config), n_features=values.shape[1])
    if architecture == "multiclass":
        row_labels = np.where(any_fault[:, None], np.where(own_fault, 1, 2), 0).ravel()
        artifact["model"] = _fit_classifier(values, row_labels, config, seed, jobs)
    else:
        artifact["detector"] = _fit_classifier(
            _symmetric_design(values), any_fault.astype(int), config, seed, jobs
        )
        fault_rows = np.repeat(any_fault, 2)
        artifact["localizer"] = (
            _fit_classifier(values[fault_rows], own_fault.ravel()[fault_rows].astype(int),
                            config, seed + 1, jobs)
            if any_fault.any() else None
        )
    return artifact


def paired_probability(artifact, x_rows):
    """Return an N-by-2 array of mutually exclusive Side I/Side II probabilities."""
    values = _paired_rows(x_rows)
    if values.shape[1] != artifact["n_features"]:
        raise ValueError("Feature count does not match the fitted paired model")
    architecture = artifact["config"]["architecture"]
    if architecture == "multiclass":
        own = _class_probability(artifact["model"], values, 1).reshape(-1, 2)
        other = _class_probability(artifact["model"], values, 2).reshape(-1, 2)
        return 0.5 * (own + other[:, ::-1])
    if architecture == "hierarchical":
        detector = _class_probability(artifact["detector"], _symmetric_design(values), 1)
        if artifact["localizer"] is None:
            attribution = np.full((len(values) // 2, 2), 0.5)
        else:
            own = _class_probability(artifact["localizer"], values, 1).reshape(-1, 2)
            attribution = 0.5 * (own + (1.0 - own[:, ::-1]))
        return detector[:, None] * attribution
    raise ValueError("Artifact architecture must be multiclass or hierarchical")
