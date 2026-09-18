"""
pipelines.py - Unified Pipeline Definitions for C151 App
Contains all pipeline classes required for unpickling and inference across:
- Saloon Door (DoorPipeline)
- Structural Health Monitoring (SHMPipeline)
- Rail Corrugation (RailPipeline)
- Air Conditioning / Ventilation (ACVPipeline)
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd


# ==========================================
# 1. SHM PIPELINE
# ==========================================
CORE_FEATURES = [
    "log_rf_energy_range_m_4.25",
    "log_rf_energy_range_m_3.5",
    "log_rf_energy_range_m_3.0",
    "log_rf_goodman_su_600_m_3.5",
    "p95_p05",
    "p99_p01",
    "std",
    "ptp",
    "rf_p95_range",
    "p01",
    "crest_factor",
    "spec_alpha2",
    "spec_zero_crossing",
]


class SHMPipeline:
    """End-to-end inference pipeline for SHM cumulative fatigue damage."""
    def __init__(self, features: List[str], scaler: Any, models: Dict[str, Any], weights: Dict[str, float]):
        self.features = features
        self.scaler = scaler
        self.models = models
        self.weights = weights

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[self.features].values
        X_scaled = self.scaler.transform(X)

        preds = np.zeros(len(df_features), dtype=np.float64)
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w > 0:
                pred_log = model.predict(X_scaled)
                pred_val = np.exp(pred_log)
                preds += w * pred_val

        return np.clip(preds, 0.001, 1.5)


# ==========================================
# 2. DOOR PIPELINE
# ==========================================
class DoorPipeline:
    """End-to-end inference pipeline for Door abnormal resistance detection."""
    def __init__(self, feature_columns: List[str], model: Any, threshold: float = 0.35):
        self.feature_columns = feature_columns
        self.model = model
        self.threshold = threshold

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[self.feature_columns].values
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df_features: pd.DataFrame) -> List[str]:
        probs = self.predict_proba(df_features)
        preds = []
        for p in probs:
            if p >= self.threshold:
                preds.append("Abnormal resistance")
            else:
                preds.append("Normal")
        return preds


# ==========================================
# 3. RAIL CORRUGATION PIPELINE
# ==========================================
class RailPipeline:
    """End-to-end inference pipeline for Rail Corrugation 3-Class Detection & Localization."""
    def __init__(
        self,
        feature_columns: List[str],
        models: List[Any],
        class_names: List[str] = None,
        alpha: float = 1.05,
        beta: float = 0.80
    ):
        self.feature_columns = feature_columns
        self.models = models
        self.class_names = class_names if class_names is not None else ["Normal", "Side I", "Side II"]
        self.alpha = alpha
        self.beta = beta

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[self.feature_columns].values
        n_samples = len(df_features)
        total_probs = np.zeros((n_samples, len(self.class_names)), dtype=np.float64)

        for model in self.models:
            probs = model.predict_proba(X)
            total_probs += probs

        avg_probs = total_probs / len(self.models)
        return avg_probs

    def predict(self, df_features: pd.DataFrame) -> List[str]:
        probs = self.predict_proba(df_features).copy()
        probs[:, 1] *= self.alpha
        probs[:, 2] *= self.beta

        best_idx = np.argmax(probs, axis=1)
        return [self.class_names[idx] for idx in best_idx]


# ==========================================
# 4. ACV PIPELINE
# ==========================================
class ACVPipeline:
    """Thermodynamic Condition Monitoring & Fault Localisation Pipeline for ACV."""
    def __init__(self, w_cool_err=1.0, w_fleet_rel=0.5, w_pressure=10.0, w_persistence=0.2):
        self.w_cool_err = float(w_cool_err)
        self.w_fleet_rel = float(w_fleet_rel)
        self.w_pressure = float(w_pressure)
        self.w_persistence = float(w_persistence)
        self.is_fitted = True

    def compute_anomaly_score(self, feat_dict: dict) -> float:
        if feat_dict.get("is_active", 1) == 0:
            return -999.0

        cool_err = feat_dict.get("tracking_error_cool", 0.0)
        rel_err = feat_dict.get("fleet_rel_err", 0.0)
        p_drop = feat_dict.get("pressure_asymmetry", 0.0)
        persistence = feat_dict.get("persistence_over_setpoint", 0.0)

        score = (
            self.w_cool_err * cool_err
            + self.w_fleet_rel * rel_err
            + self.w_pressure * p_drop
            + self.w_persistence * persistence
        )
        return float(score)

    def predict_file(self, file_path_or_df) -> dict:
        try:
            from ACV.code.features import extract_acv_features
        except ImportError:
            from features import extract_acv_features

        res = extract_acv_features(file_path_or_df)
        active_cars = res["active_cars"]
        all_cars = res["all_cars"]
        car_features = res["car_features"]
        time_series = res["time_series"]

        for c in all_cars:
            if c in active_cars:
                car_features[c]["final_score"] = self.compute_anomaly_score(car_features[c])
            else:
                car_features[c]["final_score"] = -999.0

        ranked_active = sorted(active_cars, key=lambda c: car_features[c]["final_score"], reverse=True)
        inactive_cars = [c for c in all_cars if c not in active_cars]
        final_ranked = ranked_active + inactive_cars

        ranked_str = "|".join(final_ranked)
        top_car = final_ranked[0] if final_ranked else "01"

        return {
            "ranked_cars": ranked_str,
            "top_faulty_car": top_car,
            "active_cars": active_cars,
            "all_cars": all_cars,
            "car_features": car_features,
            "time_series": time_series
        }

    def predict_batch(self, input_path: str) -> pd.DataFrame:
        input_path = str(input_path)
        if os.path.isfile(input_path):
            files = [input_path]
        elif os.path.isdir(input_path):
            import glob
            files = sorted(glob.glob(os.path.join(input_path, "*.xlsx")))
        else:
            raise FileNotFoundError(f"Path does not exist: {input_path}")

        records = []
        for f in files:
            file_id = os.path.basename(f)
            diag = self.predict_file(f)
            records.append({
                "file_id": file_id,
                "ranked_cars": diag["ranked_cars"]
            })

        return pd.DataFrame(records)


def register_pipeline_shims():
    """Binds all pipeline classes to sys.modules['pipeline'] to ensure unpickling works seamlessly."""
    import types
    if "pipeline" not in sys.modules:
        p_mod = types.ModuleType("pipeline")
        sys.modules["pipeline"] = p_mod
    else:
        p_mod = sys.modules["pipeline"]

    p_mod.SHMPipeline = SHMPipeline
    p_mod.DoorPipeline = DoorPipeline
    p_mod.RailPipeline = RailPipeline
    p_mod.ACVPipeline = ACVPipeline
    p_mod.CORE_FEATURES = CORE_FEATURES
