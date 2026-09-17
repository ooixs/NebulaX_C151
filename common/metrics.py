"""Exact re-implementations of the four PS3 judge metrics (see docs/references/*/..._Info_Kit.md)."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- Door


def parse_door_time(s: str | pd.Timestamp) -> pd.Timestamp:
    """Parse the native 'Y-M-D-h-m-s-ms' door timestamp (or anything pandas can parse)."""
    if isinstance(s, pd.Timestamp):
        return s
    parts = str(s).split("-")
    if len(parts) == 7 and all(p.isdigit() for p in parts):
        y, mo, d, h, mi, se, ms = (int(p) for p in parts)
        return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=se, microsecond=ms * 1000)
    return pd.Timestamp(s)


def format_door_time(t: pd.Timestamp) -> str:
    return f"{t.year}-{t.month}-{t.day}-{t.hour}-{t.minute}-{t.second}-{t.microsecond // 1000}"


def _to_seconds(df: pd.DataFrame) -> np.ndarray:
    s = df["start_time"].map(parse_door_time).astype("int64").to_numpy() / 1e9
    e = df["end_time"].map(parse_door_time).astype("int64").to_numpy() / 1e9
    return np.stack([s, e], axis=1)


def segment_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def door_iou_f1(true: pd.DataFrame, pred: pd.DataFrame, ignore_labels: bool = False) -> dict:
    """IoU-weighted F1 per Door Info Kit §4.

    true: columns start_time,end_time,status   pred: columns start_time,end_time,prediction
    Greedy one-to-one matching by highest IoU among same-label pairs with IoU>0.
    Returns dict(score, soft_recall, soft_precision, n_true, n_pred, n_match, mean_iou).
    """
    if len(true) == 0 and len(pred) == 0:
        return dict(score=0.0, soft_recall=0.0, soft_precision=0.0, n_true=0, n_pred=0, n_match=0, mean_iou=0.0)
    T = _to_seconds(true)
    P = _to_seconds(pred) if len(pred) else np.zeros((0, 2))
    tl = true["status"].to_numpy() if not ignore_labels else np.zeros(len(true))
    pl = pred["prediction"].to_numpy() if (not ignore_labels and len(pred)) else np.zeros(len(pred))
    cands = []
    for i in range(len(T)):
        for j in range(len(P)):
            if tl[i] != pl[j]:
                continue
            iou = segment_iou(T[i], P[j])
            if iou > 0:
                cands.append((iou, i, j))
    cands.sort(key=lambda x: -x[0])
    used_t, used_p, total = set(), set(), 0.0
    for iou, i, j in cands:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        total += iou
    n_match = len(used_t)
    sr = total / len(T) if len(T) else 0.0
    sp = total / len(P) if len(P) else 0.0
    score = 2 * sr * sp / (sr + sp) if (sr + sp) > 0 else 0.0
    return dict(score=score, soft_recall=sr, soft_precision=sp, n_true=len(T), n_pred=len(P),
                n_match=n_match, mean_iou=(total / n_match if n_match else 0.0))


# --------------------------------------------------------------------------- ACV


def acv_rank_score(ranked_cars: Iterable[str] | str, true_car: str) -> float:
    """(n - (r-1)) / n ; 0 if the true car is absent."""
    if isinstance(ranked_cars, str):
        ranked_cars = ranked_cars.split("|")
    ranked = [str(c).strip() for c in ranked_cars]
    n = len(ranked)
    if str(true_car).strip() not in ranked:
        return 0.0
    r = ranked.index(str(true_car).strip()) + 1
    return (n - (r - 1)) / n


# --------------------------------------------------------------------------- Rail


def rail_macro_f1(y_true: Sequence[str], y_pred: Sequence[str],
                  classes: Sequence[str] = ("Normal", "Side I", "Side II")) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    f1s = {}
    for c in classes:
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[c] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(macro_f1=float(np.mean(list(f1s.values()))), per_class=f1s)


# --------------------------------------------------------------------------- SHM


def shm_score(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(yt - yp) / np.abs(yt)
    mape = float(np.mean(ape))
    return dict(score=max(0.0, 1.0 - mape), mape=mape, ape=ape)
