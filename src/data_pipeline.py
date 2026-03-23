"""
data_pipeline.py
Handles loading, cleaning, and preprocessing of traffic datasets.
"""

import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw traffic CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} records from {filepath}")
    return df


# ─────────────────────────────────────────────
# 2. Data Cleaning
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw traffic dataframe:
      - Parse timestamps
      - Drop nulls
      - Remove duplicates
      - Fix negative / impossible values
    """
    df = df.copy()

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        before = len(df)
        df.dropna(subset=["timestamp"], inplace=True)
        print(f"[INFO] Dropped {before - len(df)} rows with invalid timestamps.")

    # Drop rows with any remaining nulls
    before = len(df)
    df.dropna(inplace=True)
    print(f"[INFO] Dropped {before - len(df)} rows with missing values.")

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Dropped {before - len(df)} duplicate rows.")

    # Sanity checks
    if "vehicle_count" in df.columns:
        df = df[df["vehicle_count"] >= 0]
    if "average_speed" in df.columns:
        df = df[df["average_speed"] >= 0]

    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Clean dataset has {len(df)} records.")
    return df


# ─────────────────────────────────────────────
# 3. Encode Target
# ─────────────────────────────────────────────

CONGESTION_MAP = {"Low": 0, "Medium": 1, "High": 2}
CONGESTION_REVERSE = {v: k for k, v in CONGESTION_MAP.items()}


def encode_target(df: pd.DataFrame, target_col: str = "congestion_level") -> pd.DataFrame:
    """Label-encode the congestion level column."""
    df = df.copy()
    if target_col in df.columns:
        df[target_col] = df[target_col].map(CONGESTION_MAP)
        df.dropna(subset=[target_col], inplace=True)
        df[target_col] = df[target_col].astype(int)
    return df


# ─────────────────────────────────────────────
# 4. Generate Synthetic Dataset (for demo)
# ─────────────────────────────────────────────

def generate_synthetic_dataset(n_days: int = 60,
                                roads: list = None,
                                output_path: str = "data/raw/traffic_data.csv") -> pd.DataFrame:
    """
    Generate a realistic synthetic traffic dataset when no real data is available.
    Covers multiple roads across n_days at 30-minute intervals.
    """
    if roads is None:
        roads = ["R101", "R102", "R103", "R104", "R105"]

    np.random.seed(42)
    records = []

    base_date = pd.Timestamp("2024-01-01")
    intervals = pd.date_range(base_date, periods=n_days * 48, freq="30min")

    weather_options = ["Clear", "Rainy", "Foggy", "Cloudy"]
    events = ["None", "None", "None", "None", "Festival", "Match", "Concert"]

    for road in roads:
        for ts in intervals:
            hour = ts.hour
            dow = ts.dayofweek          # 0=Mon … 6=Sun
            is_weekend = int(dow >= 5)
            is_holiday = int(ts.date() in _get_holidays())

            # --- Vehicle count heuristic ---
            if 7 <= hour <= 9 or 17 <= hour <= 19:          # peak
                base_count = np.random.randint(450, 700)
            elif 10 <= hour <= 16:                            # mid-day
                base_count = np.random.randint(200, 450)
            elif 20 <= hour <= 23:                            # evening
                base_count = np.random.randint(150, 300)
            else:                                             # night
                base_count = np.random.randint(20, 100)

            if is_weekend:
                base_count = int(base_count * 0.7)
            if is_holiday:
                base_count = int(base_count * 0.5)

            vehicle_count = max(0, base_count + np.random.randint(-30, 30))

            # --- Speed heuristic ---
            if vehicle_count > 500:
                avg_speed = max(5, np.random.uniform(10, 25))
                congestion = "High"
            elif vehicle_count > 300:
                avg_speed = np.random.uniform(25, 45)
                congestion = "Medium"
            else:
                avg_speed = np.random.uniform(45, 80)
                congestion = "Low"

            records.append({
                "timestamp": ts,
                "road_id": road,
                "vehicle_count": vehicle_count,
                "average_speed": round(avg_speed, 1),
                "congestion_level": congestion,
                "weather_condition": np.random.choice(weather_options),
                "day_of_week": dow,
                "holiday_flag": is_holiday,
                "special_events": np.random.choice(events),
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Synthetic dataset saved to {output_path}  ({len(df)} rows)")
    return df


def _get_holidays():
    """Return a small set of holiday dates."""
    return {
        pd.Timestamp("2024-01-26").date(),
        pd.Timestamp("2024-08-15").date(),
        pd.Timestamp("2024-10-02").date(),
    }


# ─────────────────────────────────────────────
# 5. Save Processed Data
# ─────────────────────────────────────────────

def save_processed(df: pd.DataFrame,
                   output_path: str = "data/processed/processed_traffic_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Processed data saved to {output_path}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 – generate (or load) raw data
    raw_path = "data/raw/traffic_data.csv"
    if not os.path.exists(raw_path):
        generate_synthetic_dataset(output_path=raw_path)

    df_raw = load_data(raw_path)

    # Step 2 – clean
    df_clean = clean_data(df_raw)

    # Step 3 – encode target
    df_encoded = encode_target(df_clean)

    # Step 4 – save
    save_processed(df_encoded)