"""
predict.py
Real-time congestion prediction pipeline.
Takes road + time as input, engineers features, and returns congestion level.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from train_model import load_model, load_feature_columns

CONGESTION_MAP_REVERSE = {0: "Low", 1: "Medium", 2: "High"}
CONGESTION_COLOR       = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
ALTERNATE_ROUTES = {
    "R101": "R103 via Outer Ring Road",
    "R102": "R104 via Express Highway",
    "R103": "R101 via City Boulevard",
    "R104": "R105 via Bypass Road",
    "R105": "R102 via Northern Link",
}


# ─────────────────────────────────────────────
# 1. Feature Builder for Single Prediction
# ─────────────────────────────────────────────

def build_single_record(road_id: str,
                         timestamp: pd.Timestamp,
                         vehicle_count: int = 400,
                         average_speed: float = 35.0,
                         weather: str = "Clear",
                         holiday_flag: int = 0,
                         special_event: str = "None",
                         recent_counts: list = None) -> pd.DataFrame:
    """
    Construct a single-row feature dataframe for inference.
    `recent_counts` is a list of the last 12 vehicle counts on that road
    (most recent last) — used for lag/rolling features.
    """
    if recent_counts is None:
        recent_counts = [vehicle_count] * 12

    hour  = timestamp.hour
    dow   = timestamp.dayofweek

    row = {
        "hour_of_day":          hour,
        "minute_of_hour":       timestamp.minute,
        "day_of_week":          dow,
        "day_of_month":         timestamp.day,
        "month":                timestamp.month,
        "week_of_year":         timestamp.isocalendar()[1],
        "weekend_flag":         int(dow >= 5),
        "hour_sin":             np.sin(2 * np.pi * hour / 24),
        "hour_cos":             np.cos(2 * np.pi * hour / 24),
        "dow_sin":              np.sin(2 * np.pi * dow / 7),
        "dow_cos":              np.cos(2 * np.pi * dow / 7),
        "vehicle_count":        vehicle_count,
        "average_speed":        average_speed,
        "holiday_flag":         holiday_flag,
        "road_id_enc":          _encode_road(road_id),
        # Lag features (from recent history)
        "vehicle_count_lag_1":  recent_counts[-1],
        "vehicle_count_lag_2":  recent_counts[-2],
        "vehicle_count_lag_3":  recent_counts[-3],
        "vehicle_count_lag_6":  recent_counts[-6],
        "vehicle_count_lag_12": recent_counts[-12],
        # Rolling means
        "rolling_avg_3":        np.mean(recent_counts[-3:]),
        "rolling_avg_6":        np.mean(recent_counts[-6:]),
        "rolling_avg_12":       np.mean(recent_counts[-12:]),
        "rolling_std_3":        np.std(recent_counts[-3:]),
        "rolling_std_6":        np.std(recent_counts[-6:]),
        "rolling_std_12":       np.std(recent_counts[-12:]),
        # Derived
        "traffic_growth_rate":  (vehicle_count - recent_counts[-1]) / max(recent_counts[-1], 1),
        "speed_drop_rate":      (100 - average_speed) / 100,
        "volume_capacity_ratio": vehicle_count / 700,
        # Weather dummies
        "weather_Clear":  int(weather == "Clear"),
        "weather_Rainy":  int(weather == "Rainy"),
        "weather_Foggy":  int(weather == "Foggy"),
        "weather_Cloudy": int(weather == "Cloudy"),
        # Event dummies
        "event_None":     int(special_event == "None"),
        "event_Festival": int(special_event == "Festival"),
        "event_Match":    int(special_event == "Match"),
        "event_Concert":  int(special_event == "Concert"),
    }

    return pd.DataFrame([row])


def _encode_road(road_id: str) -> int:
    """Simple deterministic encoding for road IDs."""
    road_order = {"R101": 0, "R102": 1, "R103": 2, "R104": 3, "R105": 4}
    return road_order.get(road_id, hash(road_id) % 100)


# ─────────────────────────────────────────────
# 2. Align Columns with Training Schema
# ─────────────────────────────────────────────

def align_features(df: pd.DataFrame,
                   feature_columns: list) -> pd.DataFrame:
    """
    Add any missing columns (filled with 0) and reorder to match training.
    """
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


# ─────────────────────────────────────────────
# 3. Core Prediction Function
# ─────────────────────────────────────────────

def predict_congestion(road_id: str,
                        timestamp: pd.Timestamp,
                        vehicle_count: int = 400,
                        average_speed: float = 35.0,
                        weather: str = "Clear",
                        holiday_flag: int = 0,
                        special_event: str = "None",
                        recent_counts: list = None,
                        model=None,
                        feature_columns: list = None) -> dict:
    """
    Predict congestion level and return a structured result dict.
    """
    # Load model & columns if not provided
    if model is None:
        model = load_model("congestion_model")
    if feature_columns is None:
        feature_columns = load_feature_columns()

    # Build feature row
    X = build_single_record(road_id, timestamp, vehicle_count,
                             average_speed, weather, holiday_flag,
                             special_event, recent_counts)

    # Align to training schema
    X = align_features(X, feature_columns)

    # Predict
    pred_class = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    congestion_label = CONGESTION_MAP_REVERSE[pred_class]
    icon             = CONGESTION_COLOR[congestion_label]

    # Estimated vehicle count (simple rule if not provided directly)
    est_count = vehicle_count

    result = {
        "road_id":           road_id,
        "timestamp":         timestamp.strftime("%Y-%m-%d %H:%M"),
        "predicted_class":   pred_class,
        "congestion_level":  congestion_label,
        "icon":              icon,
        "estimated_vehicles": est_count,
        "suggested_action":  _get_action(congestion_label, road_id),
    }

    if proba is not None:
        result["probabilities"] = {
            "Low":    round(float(proba[0]), 3),
            "Medium": round(float(proba[1]), 3),
            "High":   round(float(proba[2]), 3),
        }

    return result


def _get_action(level: str, road_id: str) -> str:
    if level == "High":
        alt = ALTERNATE_ROUTES.get(road_id, "available side roads")
        return f"Use alternate route via {alt}"
    elif level == "Medium":
        return "Expect moderate delays — consider travelling 30 min earlier/later."
    else:
        return "Road is clear — no action required."


# ─────────────────────────────────────────────
# 4. Batch Prediction
# ─────────────────────────────────────────────

def batch_predict(df: pd.DataFrame,
                  model=None,
                  feature_columns: list = None) -> pd.DataFrame:
    """
    Run prediction on a dataframe that already has features engineered.
    Returns the original dataframe with 'predicted_congestion' column added.
    """
    if model is None:
        model = load_model("congestion_model")
    if feature_columns is None:
        feature_columns = load_feature_columns()

    X = align_features(df.copy(), feature_columns)
    preds = model.predict(X)
    df = df.copy()
    df["predicted_congestion"] = [CONGESTION_MAP_REVERSE[p] for p in preds]
    return df


# ─────────────────────────────────────────────
# 5. Print Result
# ─────────────────────────────────────────────

def print_prediction(result: dict):
    print("\n" + "=" * 45)
    print("  TRAFFIC CONGESTION PREDICTION")
    print("=" * 45)
    print(f"  Road       : {result['road_id']}")
    print(f"  Time       : {result['timestamp']}")
    print(f"  Vehicles   : ~{result['estimated_vehicles']}")
    print(f"  Congestion : {result['icon']} {result['congestion_level']}")
    if "probabilities" in result:
        p = result["probabilities"]
        print(f"  Confidence : Low {p['Low']:.0%}  Med {p['Medium']:.0%}  High {p['High']:.0%}")
    print(f"  Action     : {result['suggested_action']}")
    print("=" * 45 + "\n")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ts = pd.Timestamp("2024-03-15 08:30:00")

    result = predict_congestion(
        road_id="R101",
        timestamp=ts,
        vehicle_count=620,
        average_speed=18.0,
        weather="Clear",
        holiday_flag=0,
        special_event="None",
    )
    print_prediction(result)

    # Another example — off-peak
    result2 = predict_congestion(
        road_id="R103",
        timestamp=pd.Timestamp("2024-03-15 14:00:00"),
        vehicle_count=210,
        average_speed=65.0,
    )
    print_prediction(result2)