"""
features.py - Feature Extraction Engine for ACV Refrigerant Leakage Localisation

Extracts thermodynamic tracking errors, fleet-relative temperature departures,
cooling mode performance metrics, and refrigeration circuit asymmetry indicators
from multivariate trainborne ACV telemetry.
"""

import os
import re
import numpy as np
import pandas as pd


def load_telemetry(file_path_or_df):
    """
    Loads telemetry DataFrame from .xlsx path using calamine or openpyxl,
    or passes through existing DataFrame.
    """
    if isinstance(file_path_or_df, pd.DataFrame):
        return file_path_or_df
    
    file_path = str(file_path_or_df)
    try:
        return pd.read_excel(file_path, engine="calamine")
    except Exception:
        return pd.read_excel(file_path, engine="openpyxl")


def extract_acv_features(file_path_or_df):
    """
    Extracts comprehensive thermodynamic and diagnostic features per car from ACV telemetry.

    Parameters
    ----------
    file_path_or_df : str, Path, or pd.DataFrame
        Input Excel file path or loaded DataFrame.

    Returns
    -------
    dict
        {
            'active_cars': list of active car IDs,
            'all_cars': list of all car IDs found in headers,
            'car_features': dict of {car_id: {feature_name: value}},
            'time_series': dict of {car_id: {'indoor': series, 'target': series, 'error': series}}
        }
    """
    df = load_telemetry(file_path_or_df)
    cols = df.columns.tolist()
    
    # 1. Discover all car identifiers (e.g. ['01', '02', ..., '08'])
    all_cars = sorted(list(set(re.findall(r"Car\s+(\d+)", " ".join(cols)))))
    if not all_cars:
        # Fallback if no 'Car <NN>' pattern
        all_cars = [f"{i:02d}" for i in range(1, 9)]
        
    # 2. Check active status (presence of non-null telemetry)
    active_cars = []
    car_col_map = {}
    for c in all_cars:
        c_cols = [col for col in cols if f"Car {c}" in col]
        non_null_count = df[c_cols].notnull().sum().sum() if c_cols else 0
        if non_null_count > 0:
            active_cars.append(c)
        car_col_map[c] = c_cols

    if not active_cars:
        active_cars = list(all_cars)

    # 3. Resolve key column mappings for each car
    car_series = {}
    for c in active_cars:
        c_cols = car_col_map[c]
        
        # Indoor Cabin Temperature
        tin_col = None
        for col in c_cols:
            col_l = col.lower()
            if any(k in col_l for k in ["indoor average temperature", "passenger cabin temperature"]):
                tin_col = col
                break
        
        # Target / Control Setpoint Temperature
        tctrl_col = None
        for col in c_cols:
            col_l = col.lower()
            if any(k in col_l for k in ["control temperature (cooling)", "target temperature value", "target temperature 0"]):
                tctrl_col = col
                break
                
        # Outdoor / Ambient Temperature
        tout_col = None
        for col in c_cols:
            col_l = col.lower()
            if any(k in col_l for k in ["outdoor average temperature", "outside temperature sensor reading", "fresh air temperature"]):
                tout_col = col
                break
                
        # ACV Running / Operating Mode
        mode_col = None
        for col in c_cols:
            col_l = col.lower()
            if any(k in col_l for k in ["acv running mode", "acv operating mode"]):
                mode_col = col
                break
                
        # High-side pressures
        p_high_cols = [col for col in c_cols if "high pressure" in col.lower()]
        p_low_cols = [col for col in c_cols if "low pressure" in col.lower()]
        
        tin = df[tin_col] if tin_col else pd.Series(dtype=float)
        tctrl = df[tctrl_col] if tctrl_col else pd.Series(dtype=float)
        tout = df[tout_col] if tout_col else pd.Series(dtype=float)
        mode = df[mode_col] if mode_col else pd.Series(dtype=object)
        
        # Ensure numeric
        tin = pd.to_numeric(tin, errors="coerce")
        tctrl = pd.to_numeric(tctrl, errors="coerce")
        tout = pd.to_numeric(tout, errors="coerce")
        
        car_series[c] = {
            "indoor": tin,
            "target": tctrl,
            "outdoor": tout,
            "mode": mode,
            "p_high": [pd.to_numeric(df[p], errors="coerce") for p in p_high_cols],
            "p_low": [pd.to_numeric(df[p], errors="coerce") for p in p_low_cols]
        }

    # 4. Fleet-wide baselines across time (median cabin temp & median tracking error)
    all_indoor = pd.DataFrame({c: car_series[c]["indoor"] for c in active_cars})
    fleet_median_indoor = all_indoor.median(axis=1)
    
    all_err = pd.DataFrame({c: (car_series[c]["indoor"] - car_series[c]["target"]) for c in active_cars})
    fleet_median_err = all_err.median(axis=1)

    # 5. Extract car-level features
    car_features = {}
    time_series_data = {}
    
    for c in all_cars:
        if c not in active_cars:
            # Inactive car (e.g. disconnected or absent)
            car_features[c] = {
                "is_active": 0,
                "anomaly_score": -999.0,
                "tracking_error_mean": -999.0,
                "tracking_error_median": -999.0,
                "tracking_error_p90": -999.0,
                "tracking_error_max": -999.0,
                "tracking_error_cool": -999.0,
                "indoor_temp_mean": -999.0,
                "indoor_temp_max": -999.0,
                "fleet_rel_temp": -999.0,
                "fleet_rel_err": -999.0,
                "persistence_over_setpoint": 0.0,
                "pressure_asymmetry": 0.0
            }
            time_series_data[c] = {"indoor": [], "target": [], "error": []}
            continue

        s = car_series[c]
        tin = s["indoor"]
        tctrl = s["target"]
        mode = s["mode"]
        
        err = tin - tctrl
        valid_mask = err.notnull()
        err_valid = err[valid_mask]
        
        # Cooling mode filter
        if not mode.empty:
            cool_mask = mode.astype(str).str.lower().str.contains("cool") & valid_mask
            err_cool = err[cool_mask] if cool_mask.sum() > 0 else err_valid
        else:
            err_cool = err_valid

        # Statistical aggregates
        mean_err = float(err_valid.mean()) if len(err_valid) > 0 else 0.0
        median_err = float(err_valid.median()) if len(err_valid) > 0 else 0.0
        p90_err = float(err_valid.quantile(0.90)) if len(err_valid) > 0 else 0.0
        max_err = float(err_valid.max()) if len(err_valid) > 0 else 0.0
        cool_mean_err = float(err_cool.mean()) if len(err_cool) > 0 else mean_err

        mean_tin = float(tin.mean()) if tin.notnull().sum() > 0 else 0.0
        max_tin = float(tin.max()) if tin.notnull().sum() > 0 else 0.0

        # Fleet-relative departure
        rel_temp = float((tin - fleet_median_indoor).mean()) if len(tin) > 0 else 0.0
        rel_err = float((err - fleet_median_err).mean()) if len(err_valid) > 0 else 0.0

        # Persistence exceeding setpoint
        persistence = float((err_valid > 0.5).mean()) if len(err_valid) > 0 else 0.0
        severe_rate = float((err_valid > 2.0).mean()) if len(err_valid) > 0 else 0.0

        # Pressure asymmetry between dual circuits
        p_drop = 0.0
        if len(s["p_high"]) >= 2:
            p1 = float(s["p_high"][0].mean())
            p2 = float(s["p_high"][1].mean())
            if not (np.isnan(p1) or np.isnan(p2)):
                p_drop = abs(p1 - p2) / max(p1, p2, 1.0)

        # Composite thermodynamic anomaly score
        # When refrigerant leaks, cooling tracking error is elevated and/or circuit pressure drops
        anomaly_score = cool_mean_err + 0.5 * rel_err + p_drop * 10.0

        car_features[c] = {
            "is_active": 1,
            "anomaly_score": anomaly_score,
            "tracking_error_mean": mean_err,
            "tracking_error_median": median_err,
            "tracking_error_p90": p90_err,
            "tracking_error_max": max_err,
            "tracking_error_cool": cool_mean_err,
            "indoor_temp_mean": mean_tin,
            "indoor_temp_max": max_tin,
            "fleet_rel_temp": rel_temp,
            "fleet_rel_err": rel_err,
            "persistence_over_setpoint": persistence,
            "severe_excursion_rate": severe_rate,
            "pressure_asymmetry": p_drop
        }

        # Downsample time series for UI visualization
        step = max(1, len(tin) // 400)
        time_series_data[c] = {
            "indoor": tin.iloc[::step].ffill().tolist(),
            "target": tctrl.iloc[::step].ffill().tolist(),
            "error": err.iloc[::step].fillna(0).tolist()
        }

    return {
        "active_cars": active_cars,
        "all_cars": all_cars,
        "car_features": car_features,
        "time_series": time_series_data
    }
