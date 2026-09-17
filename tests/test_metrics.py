import numpy as np
import pandas as pd
import pytest

from common.metrics import acv_rank_score, door_iou_f1, rail_macro_f1, shm_score, parse_door_time, format_door_time


def test_shm_worked_example():
    r = shm_score([0.10, 0.30, 0.50, 0.70, 0.90], [0.15, 0.28, 0.55, 0.68, 0.85])
    assert r["score"] == pytest.approx(0.850, abs=1e-3)
    r2 = shm_score([0.10, 0.30, 0.50, 0.70, 0.90], [0.5] * 5)
    assert r2["score"] == 0.0 and r2["mape"] > 1.0


def test_acv_worked_example():
    ranked = "03|01|05|02|04|06|07|08"
    assert acv_rank_score(ranked, "03") == 1.0
    assert acv_rank_score(ranked, "01") == pytest.approx(0.875)
    assert acv_rank_score(ranked, "05") == pytest.approx(0.750)
    assert acv_rank_score(ranked, "08") == pytest.approx(0.125)
    assert acv_rank_score(ranked, "09") == 0.0


def test_rail_macro_f1_all_normal():
    yt = ["Normal"] * 9 + ["Side I"] + ["Side II"]
    r = rail_macro_f1(yt, ["Normal"] * 11)
    assert r["per_class"]["Side I"] == 0 and r["per_class"]["Side II"] == 0
    assert r["macro_f1"] == pytest.approx((2 * 9 / 11 / (1 + 9 / 11)) / 3)


def test_door_perfect_and_wrong_label():
    true = pd.DataFrame({"start_time": ["2023-7-5-0-0-0-0", "2023-7-5-0-0-30-0"],
                         "end_time": ["2023-7-5-0-0-4-500", "2023-7-5-0-0-33-800"],
                         "status": ["Normal", "Abnormal resistance"]})
    pred = true.rename(columns={"status": "prediction"})
    assert door_iou_f1(true, pred)["score"] == pytest.approx(1.0)
    bad = pred.copy(); bad.loc[1, "prediction"] = "Normal"
    r = door_iou_f1(true, bad)
    assert r["score"] == pytest.approx(0.5)  # one match of IoU 1 out of 2 true / 2 pred
    assert door_iou_f1(true, bad, ignore_labels=True)["score"] == pytest.approx(1.0)


def test_door_partial_overlap_and_extra():
    true = pd.DataFrame({"start_time": ["2023-7-5-0-0-0-0"], "end_time": ["2023-7-5-0-0-10-0"], "status": ["Normal"]})
    pred = pd.DataFrame({"start_time": ["2023-7-5-0-0-5-0", "2023-7-5-0-1-0-0"],
                         "end_time": ["2023-7-5-0-0-15-0", "2023-7-5-0-1-5-0"], "prediction": ["Normal", "Normal"]})
    r = door_iou_f1(true, pred)
    iou = 5 / 15
    sr, sp = iou / 1, iou / 2
    assert r["score"] == pytest.approx(2 * sr * sp / (sr + sp))


def test_door_time_roundtrip():
    s = "2023-7-5-0-11-17-664"
    assert format_door_time(parse_door_time(s)) == s
