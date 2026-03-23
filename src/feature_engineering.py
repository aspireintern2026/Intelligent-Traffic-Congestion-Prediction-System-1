"""
feature_engineering.py
Builds temporal, contextual, and derived features for traffic prediction.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. Temporal Features
# ─────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame,
                           ts_col: str = "timestamp") -> pd.DataFrame:
    """Extract date/time components from the timestamp column."""
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])

    df["hour_of_day"]    = ts.dt.hour
    df["minute_of_hour"] = ts.dt.minute
    df["day_of_week"]    = ts.dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"]   = ts.dt.day
    df["month"]          = ts.dt.month
    df["week_of_year"]   = ts.dt.isocalendar().week.astype(int)
    df["weekend_flag"]   = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for hour (helps ML models understand 23→0 wrap-around)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ─────────────────────────────────────────────
# 2. Lag / Rolling Features
# ─────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     group_col: str = "road_id",
                     target_col: str = "vehicle_count",
                     lags: list = None) -> pd.DataFrame:
    """Add lagged vehicle-count values grouped by road."""
    if lags is None:
        lags = [1, 2, 3, 6, 12]   # ×30 min lags

    df = df.copy()
    df.sort_values([group_col, "timestamp"], inplace=True)

    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame,
                          group_col: str = "road_id",
                          target_col: str = "vehicle_count",
                          windows: list = None) -> pd.DataFrame:
    """Add rolling mean/std of vehicle count grouped by road."""
    if windows is None:
        windows = [3, 6, 12]

    df = df.copy()
    df.sort_values([group_col, "timestamp"], inplace=True)

    for w in windows:
        roll = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_avg_{w}"] = roll

        roll_std = df.groupby(group_col)[target_col].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        )
        df[f"rolling_std_{w}"] = roll_std.fillna(0)

    return df


# ─────────────────────────────────────────────
# 3. Derived Traffic Features
# ─────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute traffic-specific derived signals."""
    df = df.copy()

    # Traffic growth rate (% change vs previous reading)
    if "vehicle_count_lag_1" in df.columns:
        df["traffic_growth_rate"] = (
            (df["vehicle_count"] - df["vehicle_count_lag_1"])
            / (df["vehicle_count_lag_1"].replace(0, np.nan))
        ).fillna(0)

    # Speed drop rate
    if "average_speed" in df.columns and "vehicle_count" in df.columns:
        df["speed_drop_rate"] = np.where(
            df["vehicle_count"] > 0,
            (100 - df["average_speed"]) / 100,
            0,
        )

    # Volume-to-capacity ratio proxy (assuming road capacity ~700 vehicles)
    ROAD_CAPACITY = 700
    if "vehicle_count" in df.columns:
        df["volume_capacity_ratio"] = df["vehicle_count"] / ROAD_CAPACITY

    return df


# ─────────────────────────────────────────────
# 4. Categorical Encoding
# ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode weather and special events; label-encode road_id."""
    df = df.copy()

    if "weather_condition" in df.columns:
        weather_dummies = pd.get_dummies(
            df["weather_condition"], prefix="weather", drop_first=False
        )
        df = pd.concat([df, weather_dummies], axis=1)
        df.drop(columns=["weather_condition"], inplace=True)

    if "special_events" in df.columns:
        event_dummies = pd.get_dummies(
            df["special_events"], prefix="event", drop_first=False
        )
        df = pd.concat([df, event_dummies], axis=1)
        df.drop(columns=["special_events"], inplace=True)

    if "road_id" in df.columns:
        df["road_id_enc"] = pd.Categorical(df["road_id"]).codes
        df.drop(columns=["road_id"], inplace=True)

    return df


# ─────────────────────────────────────────────
# 5. Full Feature Pipeline
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   drop_ts: bool = True) -> pd.DataFrame:
    """Run all feature-engineering steps in order."""
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_derived_features(df)
    df = encode_categoricals(df)

    # Drop rows that couldn't be filled (early lags)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if drop_ts and "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)

    print(f"[INFO] Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 6. Train / Test Split
# ─────────────────────────────────────────────

def split_features_target(df: pd.DataFrame,
                           target_col: str = "congestion_level",
                           test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test (time-ordered split)."""
    from sklearn.model_selection import train_test_split

    drop_cols = [target_col]
    # Keep vehicle_count as a feature but also use it optionally as reg target
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from data_pipeline import load_data, clean_data, encode_target, save_processed

    processed_path = "data/processed/processed_traffic_data.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError("Run data_pipeline.py first to generate processed data.")

    df = load_data(processed_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_feat = build_features(df)
    os.makedirs("data/processed", exist_ok=True)
    df_feat.to_csv("data/processed/features.csv", index=False)
    print("[INFO] Feature CSV saved to data/processed/features.csv")