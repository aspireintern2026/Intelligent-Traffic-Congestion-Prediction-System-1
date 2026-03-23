"""
eda.py  ←  Exploratory Data Analysis
Can be run as a plain Python script OR converted to Jupyter notebook.

Usage:
    python notebooks/eda.py
    # or: jupyter nbconvert --to notebook --execute notebooks/eda.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
data_path = "data/processed/processed_traffic_data.csv"
if not os.path.exists(data_path):
    print("[INFO] Processed data not found — generating synthetic dataset …")
    from data_pipeline import generate_synthetic_dataset, clean_data, encode_target, save_processed
    df_raw = generate_synthetic_dataset()
    df_clean = clean_data(df_raw)
    df = encode_target(df_clean)
    save_processed(df)
else:
    df = pd.read_csv(data_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]      = df["timestamp"].dt.hour
df["dow"]       = df["timestamp"].dt.dayofweek
df["dow_name"]  = df["timestamp"].dt.day_name()

CONGESTION_REVERSE = {0: "Low", 1: "Medium", 2: "High"}
if df["congestion_level"].dtype in [int, float]:
    df["congestion_label"] = df["congestion_level"].map(CONGESTION_REVERSE)
else:
    df["congestion_label"] = df["congestion_level"]

print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.describe())
print("\nCongestion distribution:\n", df["congestion_label"].value_counts())

# ─────────────────────────────────────────────
# Plot 1 — Hourly Average Vehicle Count
# ─────────────────────────────────────────────
hourly_avg = df.groupby("hour")["vehicle_count"].mean()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hourly_avg.index, hourly_avg.values, marker="o", color="#2196F3", linewidth=2)
ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.2, color="#2196F3")
ax.axvspan(7, 9,   alpha=0.15, color="red",    label="Morning Peak")
ax.axvspan(17, 19, alpha=0.15, color="orange", label="Evening Peak")
ax.set_title("Average Vehicle Count by Hour of Day")
ax.set_xlabel("Hour")
ax.set_ylabel("Avg Vehicle Count")
ax.set_xticks(range(24))
ax.legend()
plt.tight_layout()
fig.savefig(f"{REPORTS_DIR}/hourly_traffic.png", dpi=120)
plt.show(); plt.close()

# ─────────────────────────────────────────────
# Plot 2 — Congestion Level Distribution
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["congestion_label"].value_counts()
ax.bar(counts.index, counts.values,
       color=["#28a745", "#ffc107", "#dc3545"], edgecolor="white")
ax.set_title("Congestion Level Distribution")
ax.set_ylabel("Record Count")
plt.tight_layout()
fig.savefig(f"{REPORTS_DIR}/congestion_distribution.png", dpi=120)
plt.show(); plt.close()

# ─────────────────────────────────────────────
# Plot 3 — Speed vs Vehicle Count
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
color_map = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
for level, grp in df.groupby("congestion_label"):
    ax.scatter(grp["vehicle_count"], grp["average_speed"],
               alpha=0.3, s=8, label=level, color=color_map.get(level, "grey"))
ax.set_title("Average Speed vs Vehicle Count")
ax.set_xlabel("Vehicle Count")
ax.set_ylabel("Average Speed (km/h)")
ax.legend(title="Congestion")
plt.tight_layout()
fig.savefig(f"{REPORTS_DIR}/speed_vs_count.png", dpi=120)
plt.show(); plt.close()

# ─────────────────────────────────────────────
# Plot 4 — Congestion by Day of Week
# ─────────────────────────────────────────────
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_cong = (
    df.groupby(["dow_name", "congestion_label"])
    .size()
    .unstack(fill_value=0)
    .reindex(dow_order)
)
ax = dow_cong.plot(kind="bar", figsize=(10, 4),
                   color=["#28a745", "#ffc107", "#dc3545"],
                   edgecolor="white")
ax.set_title("Congestion Level by Day of Week")
ax.set_xlabel("")
ax.set_ylabel("Record Count")
ax.set_xticklabels(dow_order, rotation=30, ha="right")
plt.tight_layout()
ax.figure.savefig(f"{REPORTS_DIR}/congestion_by_dow.png", dpi=120)
plt.show(); plt.close()

# ─────────────────────────────────────────────
# Plot 5 — Heatmap: Road × Hour Congestion
# ─────────────────────────────────────────────
if "road_id" in df.columns:
    pivot = df.pivot_table(values="vehicle_count",
                           index="road_id", columns="hour", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, cmap="RdYlGn_r", ax=ax,
                linewidths=0.3, cbar_kws={"label": "Avg Vehicles"})
    ax.set_title("Avg Vehicle Count — Road × Hour")
    plt.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/heatmap_road_hour.png", dpi=120)
    plt.show(); plt.close()

# ─────────────────────────────────────────────
# Plot 6 — Correlation Matrix
# ─────────────────────────────────────────────
num_cols = ["vehicle_count", "average_speed", "congestion_level",
            "holiday_flag", "day_of_week"]
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
fig.savefig(f"{REPORTS_DIR}/correlation_matrix.png", dpi=120)
plt.show(); plt.close()

print(f"\n[EDA] All plots saved to {REPORTS_DIR}/")