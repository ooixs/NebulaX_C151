import numpy as np
import pandas as pd
import pytest

from common.metrics import acv_rank_score, door_iou_f1, rail_macro_f1, shm_score, parse_door_time, format_door_time


def test_shm_worked_example():
    r = shm_score([0.10, 0.30, 0.50, 0.70, 0.90], [0.15, 0.28, 0.55, 0.68, 0.85])
    assert r["score"] == pytest.approx(0.850, abs=1e-3)
    r2 = shm_score([0.10, 0.30, 0.50, 0.70, 0.90], [0.5] * 5)
    assert r2["score"] == 0.0 and r2["mape"] > 1.0


@pytest.mark.parametrize("S,D", [
    ([1.0, 10.0, 100.0], [1.0, 2.0, 10.0]),
    ([3.0, 15.0, 8.0, 60.0], [1.0, 5.0, 2.0, 10.0]),
    ([2.0, 4.0, 6.0], [1.0, 2.0, 3.0]),
])
def test_shm_scale_minimizes_mape(S, D):
    from SHM.code.pipeline import fit_C, mape

    S, D = np.asarray(S), np.asarray(D)
    C = fit_C(S, D)
    best = min(mape(D, S / candidate) for candidate in S / D)
    assert mape(D, S / C) == pytest.approx(best)
    assert fit_C(100 * S, D) == pytest.approx(100 * C)
    assert fit_C(S, 10 * D) == pytest.approx(C / 10)


def test_shm_range_binning_and_model_roundtrip():
    from SHM.code.pipeline import DamageModel, damage_sum

    cyc = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 0.5], [4.0, 0.0, 1.0]])
    original = cyc.copy()
    assert damage_sum(cyc, 2.0) == pytest.approx(4.75)
    assert damage_sum(cyc, 2.0, range_bins=2) == pytest.approx(5.5)
    assert damage_sum(cyc, 2.0, range_bins=2, bin_mode="midpoint") == pytest.approx(2.625)
    assert damage_sum(cyc, 2.0, range_bin_width=2.0) == pytest.approx(5.5)
    model = DamageModel(m=2.0, C=2.0, range_bins=2)
    restored = DamageModel.from_dict(model.to_dict())
    assert restored.predict_from_cycles(cyc) == pytest.approx(2.75)
    assert np.array_equal(cyc, original)
    with pytest.raises(ValueError):
        damage_sum(cyc, 2.0, range_bins=0)
    with pytest.raises(ValueError):
        damage_sum(cyc, 2.0, range_bins=2, range_bin_width=1.0)


def test_shm_research_starts_without_feature_caches(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from common import research as tracking
    from SHM.code import train
    from SHM.code.pipeline import cycles, damage_sum

    data = tmp_path / "data" / "SHM"
    (data / "Train").mkdir(parents=True)
    labels = []
    for i in range(8):
        series = np.array([0.0, 1.0, 0.0, -1.0, 0.0]) * (1 + i / 5)
        filename = f"train{i + 1:02d}.csv"
        pd.Series(series).to_csv(data / "Train" / filename, header=False, index=False)
        labels.append(dict(filename=filename, damage=damage_sum(cycles(series), 5.0) / 1e6))
    pd.DataFrame(labels).to_csv(data / "Train_Labels.csv", index=False)
    monkeypatch.setattr(train, "DATA", data)
    monkeypatch.setattr(train, "WEIGHTS", tmp_path / "weights" / "SHM")
    monkeypatch.setattr(train, "MODEL", tmp_path / "SHM" / "model")
    run_class = tracking.ResearchRun
    monkeypatch.setattr(tracking, "ResearchRun", lambda *args: run_class(*args, root=tmp_path))
    args = SimpleNamespace(run_id="cold-start", min_delta=0.002, patience=5, folds=2, repeats=1, jobs=1, ship=False)
    train.research(args)
    folder = train.WEIGHTS / "runs" / args.run_id
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["stop_reason"] == "score_ceiling"
    assert summary["best_score"] == pytest.approx(1.0)
    assert (folder / "shm_model.json").exists()
    assert not (train.MODEL / "shm_model.json").exists()


def test_shm_grouped_calibration_fits_only_training_files():
    from SHM.code.train import grouped_C_predict
    from SHM.code.pipeline import fit_C

    S = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 40.0])
    D = S / np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 2.0, 4.0])
    groups = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    tr, te = np.arange(6), np.array([6, 7])
    assert grouped_C_predict(S, D, groups, tr, te) == pytest.approx(D[te])
    # a group absent from training falls back to the pooled training scale
    lonely = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    pooled = fit_C(S[tr], D[tr])
    assert grouped_C_predict(S, D, lonely, tr, te) == pytest.approx(S[te] / pooled)


def test_shm_residual_model_roundtrip_and_features():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from SHM.code.pipeline import ResidualDamageModel, cycles, damage_sum, sg_series_features

    rng = np.random.default_rng(0)
    x = rng.normal(0, 3, 4096)
    cyc = cycles(x)
    feats = sg_series_features(x, cyc)
    assert feats["mean"] == pytest.approx(float(x.mean()))
    assert feats["ptp"] == pytest.approx(float(np.ptp(x)))
    assert feats["log_rf_energy_range_m_5.0"] == pytest.approx(np.log1p(damage_sum(cyc, 5.0, magnitude="range")))
    assert all(np.isfinite(v) for v in feats.values())
    names = ["std", "log_rf_energy_range_m_5.0"]
    X = np.array([[feats[n] for n in names], [feats["std"] * 2, feats["log_rf_energy_range_m_5.0"] + 1]])
    scaler = StandardScaler().fit(X)
    model = LinearRegression().fit(scaler.transform(X), np.array([0.0, 0.0]))
    base_C = damage_sum(cyc, 5.0) / 0.25
    residual = ResidualDamageModel(dict(m=5.0, C=base_C), names, scaler, model)
    assert residual.predict(x) == pytest.approx(0.25)


def test_research_plateau_requires_meaningful_improvement():
    from common.research import Plateau

    search = Plateau(min_delta=0.005, patience=2)
    assert search.update([0.79, 0.81])["accepted"]
    assert not search.update([0.805, 0.807])["accepted"]
    assert search.stop_reason is None
    assert search.update([0.83, 0.85])["accepted"]
    assert search.stale == 0
    search.update([0.84, 0.85])
    search.update([0.83, 0.84])
    assert search.stop_reason == "negligible_improvement"
    assert search.best_mean == pytest.approx(0.84)


def test_research_plateau_ceiling_and_invalid_scores():
    from common.research import Plateau

    search = Plateau(min_delta=0.002, patience=5)
    search.update([1.0, 1.0])
    assert search.stop_reason == "score_ceiling"
    with pytest.raises(ValueError):
        Plateau().update([np.nan])
    with pytest.raises(ValueError):
        Plateau(patience=0)


def test_research_preserves_previous_artifacts(tmp_path):
    from common.research import ResearchRun

    model = tmp_path / "Door" / "model" / "door_model.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"original")
    run = ResearchRun("Door", "test-run", root=tmp_path)
    candidate = run.path / "candidate.joblib"
    candidate.write_bytes(b"candidate")
    run.publish(candidate, model)
    assert model.read_bytes() == b"candidate"
    assert (run.path / "before" / "Door" / "model" / model.name).read_bytes() == b"original"
    with pytest.raises(FileExistsError):
        ResearchRun("Door", "test-run", root=tmp_path)


def test_door_nested_threshold_excludes_outer_validation_labels(monkeypatch):
    from Door.code import train

    class Reference:
        def fit(self, *args):
            return self

    class Classifier:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.column_stack([1 - X["p"], X["p"]])

    monkeypatch.setattr(train, "ReferenceProfiles", Reference)
    monkeypatch.setattr(train, "make_model", lambda kind: Classifier())
    monkeypatch.setattr(train, "features_table", lambda segs, ref: pd.DataFrame({"p": [s.p.iloc[0] for s in segs]}))
    start = pd.Timestamp("2023-01-01")
    segs = [pd.DataFrame({"t": [start + pd.Timedelta(seconds=10 * i),
                                start + pd.Timedelta(seconds=10 * i + 2)], "p": [(i + 1) / 13] * 2})
            for i in range(12)]
    labels = ["Normal"] * 6 + [train.POS] * 6
    before = train.run_cv(segs, labels, 3, "fake", nested=True)
    after = train.run_cv(segs, [train.POS] * 4 + labels[4:], 3, "fake", nested=True)
    assert before["folds"][0]["threshold"] == after["folds"][0]["threshold"]
    for fold in before["folds"]:
        assert set(fold["threshold_training_indices"]).isdisjoint(fold["validation_indices"])
    truth = train.segments_to_frame(segs).assign(status=labels)
    truth["end_time"] = [s["t"].iloc[0] + pd.Timedelta(seconds=4) for s in segs]
    partial = train.run_cv(segs, labels, 3, "fake", nested=True, answer=truth)
    assert partial["oof_iou_f1"] == pytest.approx(0.5)
    assert all(fold["iou_f1"] == pytest.approx(0.5) for fold in partial["folds"])


def test_nested_file_folds_keep_outer_validation_out_of_calibration():
    from common.research import nested_folds

    y = np.array(["Normal"] * 9 + ["Side I"] * 6 + ["Side II"] * 6)
    folds = nested_folds(y, folds=3, repeats=2, inner_folds=3)
    for fold in folds:
        tr, te = set(fold["train"]), set(fold["validation"])
        assert tr.isdisjoint(te)
        seen = []
        for inner_tr, inner_te in fold["inner"]:
            assert set(inner_tr).isdisjoint(inner_te)
            assert set(inner_tr) | set(inner_te) == tr
            assert te.isdisjoint(inner_tr) and te.isdisjoint(inner_te)
            seen.extend(inner_te)
        assert sorted(seen) == sorted(tr)
    for repeat in range(2):
        assert sorted(i for f in folds if f["repeat"] == repeat for i in f["validation"]) == list(range(len(y)))


def test_rail_engineered_features_preserve_legacy_features():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "Rail Corrugation" / "code" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("rail_pipeline_test", path)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    side = np.where(np.arange(64) % 2 == 0, 1, 2)
    rms = np.where(side == 1, 2.0, 1.0)
    ct = pd.DataFrame(dict(car=np.arange(64) // 8, side=side, rms=rms, log_rms=np.log10(rms),
                           shock_rms=rms / 2, shock_log_rms=np.log10(rms / 2), dom_lambda=0.1))
    legacy = pipeline.aggregate(ct, 12.0)
    enhanced = pipeline.aggregate(ct, 12.0, paired=True)
    assert set(legacy) <= set(enhanced)
    assert np.allclose(list(legacy.values()), [enhanced[k] for k in legacy], equal_nan=True)
    old_rows = pipeline.side_relative_rows(legacy)
    rows = pipeline.side_relative_rows(enhanced, engineered=True)
    for old, new in zip(old_rows, rows):
        assert np.allclose(list(old.values()), [new[k] for k in old], equal_nan=True)
    assert rows[0]["side_id"] == 1 and rows[1]["side_id"] == 2
    assert rows[0]["ratio_rms_mean"] == pytest.approx(1 / 3)
    assert rows[1]["ratio_rms_mean"] == pytest.approx(-1 / 3)
    assert rows[0]["own_pair_log_rms_mean"] == pytest.approx(np.log10(2))
    assert rows[0]["own_pair_car_diff_mean"] == pytest.approx(1.0)
    assert rows[0]["own_pair_car_dom_count"] == pytest.approx(8.0)
    assert rows[0]["own_pair_car_ratio_mean"] == pytest.approx(2.0)
    assert rows[1]["own_pair_car_diff_mean"] == pytest.approx(-1.0)
    assert rows[1]["own_pair_car_dom_count"] == pytest.approx(0.0)
    assert rows[1]["own_pair_car_diff_pos_frac"] == pytest.approx(0.0)
    assert rows[0]["own_pair_car_shock_diff_mean"] == pytest.approx(0.5)
    assert rows[0]["own_pair_speednorm_rms"] == pytest.approx(2.0 / 13.0)
    assert rows[0]["rel_pair_car_diff_mean"] == pytest.approx(2.0)


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
