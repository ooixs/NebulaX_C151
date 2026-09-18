"""Decision layer for the per-side Rail detector: turn (p_side1, p_side2) into a 3-class label.

Every rule is a pure function of the two side probabilities and a small parameter dict, so the
same code is used for validation, the research harness and inference.
"""
from __future__ import annotations

import numpy as np

from common.metrics import rail_macro_f1

GRID = np.round(np.linspace(0.05, 0.9, 35), 4)
MARGIN_GRID = np.round(np.linspace(0.0, 0.3, 7), 4)
RULES = ("global", "per_side", "margin")


def decode(p1, p2, decision: dict) -> np.ndarray:
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    rule = decision["rule"]
    if rule == "global":
        flagged = np.maximum(p1, p2) >= decision["tau"]
        side1 = p1 >= p2
    elif rule == "per_side":
        excess1, excess2 = p1 - decision["tau1"], p2 - decision["tau2"]
        flagged = (excess1 >= 0) | (excess2 >= 0)
        side1 = excess1 >= excess2
    elif rule == "margin":
        flagged = (np.maximum(p1, p2) >= decision["tau"]) & (np.abs(p1 - p2) >= decision["delta"])
        side1 = p1 >= p2
    else:
        raise ValueError(f"unknown decision rule {rule!r}")
    return np.where(flagged, np.where(side1, "Side I", "Side II"), "Normal")


def legacy_decision(tau: float) -> dict:
    return dict(rule="global", tau=float(tau))


def _best(candidates, labels, p1, p2):
    """Highest macro-F1 candidate; ties go to the middle candidate of the tied set (stable, not extreme)."""
    scores = np.array([rail_macro_f1(labels, decode(p1, p2, c))["macro_f1"] for c in candidates])
    tied = np.flatnonzero(scores >= scores.max() - 1e-12)
    return dict(candidates[tied[(len(tied) - 1) // 2]]), float(scores.max())


def select(rule: str, labels, probabilities) -> tuple[dict, float]:
    """Choose rule parameters on calibration probabilities (must be out-of-fold for the labels)."""
    p1, p2 = np.asarray(probabilities)[:, 0], np.asarray(probabilities)[:, 1]
    if rule == "global":
        candidates = [dict(rule="global", tau=float(t)) for t in GRID]
    elif rule == "per_side":
        candidates = [dict(rule="per_side", tau1=float(a), tau2=float(b)) for a in GRID for b in GRID]
    elif rule == "margin":
        candidates = [dict(rule="margin", tau=float(t), delta=float(d)) for t in GRID for d in MARGIN_GRID]
    else:
        raise ValueError(f"unknown decision rule {rule!r}")
    return _best(candidates, labels, p1, p2)


def contiguous_blocks(order: np.ndarray, blocks: int, offset: int = 0) -> list[np.ndarray]:
    """Split an ordering into `blocks` contiguous validation groups, rotating by `offset` positions."""
    order = np.asarray(order)
    if blocks < 2 or blocks > len(order):
        raise ValueError("blocks must be between 2 and the number of files")
    rotated = np.roll(order, -offset)
    return [np.asarray(part) for part in np.array_split(rotated, blocks)]


def block_protocol(file_numbers, blocks: int, inner_blocks: int, offsets=(0,)) -> list[dict]:
    """Outer contiguous file-number blocks; inner contiguous blocks inside each outer training set.

    Both side rows of a file always share a fold because folds are defined on file indices.
    """
    order = np.argsort(np.asarray(file_numbers))
    protocol = []
    for repeat, offset in enumerate(offsets):
        for k, validation in enumerate(contiguous_blocks(order, blocks, offset)):
            train = np.array([i for i in order if i not in set(validation.tolist())])
            inner = [(np.setdiff1d(train, part), part) for part in contiguous_blocks(train, inner_blocks)]
            protocol.append(dict(fold=repeat * blocks + k, repeat=repeat, offset=int(offset),
                                 train=train, validation=validation, inner=inner))
    return protocol
