"""
streamlit_dashboard.py
Interactive Streamlit dashboard for Traffic Congestion Prediction.

Run with:
    streamlit run app/streamlit_dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, time

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Load Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    """Load trained model and feature columns once."""
    try:
        from train_model import load_model, load_feature_columns
        model    = load_model("congestion_model")
        feat_cols = load_feature_columns()
        return model, feat_cols, True
    except Exception as e:
        return None, None, False

model, feature_columns, model_loaded = load_resources()

# ─────────────────────────────────────────────
# Sidebar — Inputs
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=80)
st.sidebar.title("🚦 Traffic Predictor")
st.sidebar.markdown("---")

ROAD_OPTIONS = {
    "R101 — City Center Road":   "R101",
    "R102 — Highway 21":         "R102",
    "R103 — Outer Ring Road":    "R103",
    "R104 — Express Highway":    "R104",
    "R105 — Northern Link":      "R105",
}

road_label   = st.sidebar.selectbox("Select Road", list(ROAD_OPTIONS.keys()))
road_id      = ROAD_OPTIONS[road_label]

selected_date = st.sidebar.date_input("Select Date", value=datetime.today())
selected_time = st.sidebar.time_input("Select Time", value=time(8, 30))

vehicle_count = st.sidebar.slider("Current Vehicle Count", 0, 800, 400, step=10)
average_speed = st.sidebar.slider("Average Speed (km/h)", 0, 120, 35, step=1)
weather       = st.sidebar.selectbox("Weather Condition",
                                      ["Clear", "Cloudy", "Rainy", "Foggy"])
holiday_flag  = int(st.sidebar.checkbox("Public Holiday / Weekend"))
special_event = st.sidebar.selectbox("Special Event",
                                      ["None", "Festival", "Match", "Concert"])

predict_btn = st.sidebar.button("🔍 Predict Congestion", use_container_width=True)

# ─────────────────────────────────────────────
# Main Panel — Header
# ─────────────────────────────────────────────
st.title("🚦 Intelligent Traffic Congestion Prediction System")
st.markdown(
    "Analyzes historical traffic patterns and predicts congestion levels "
    "to help city planners and drivers make informed decisions."
)
st.markdown("---")

# ─────────────────────────────────────────────
# KPI Cards — Current Road Status
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🚗 Vehicles on Road", f"{vehicle_count}")
col2.metric("⚡ Avg Speed", f"{average_speed} km/h")
col3.metric("🌤 Weather", weather)
col4.metric("📅 Holiday", "Yes" if holiday_flag else "No")

st.markdown("---")

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if predict_btn:
    if not model_loaded:
        st.error("⚠️ Model not found. Run `train_model.py` first to train and save models.")
    else:
        from predict import predict_congestion, print_prediction

        ts = pd.Timestamp(
            year=selected_date.year,
            month=selected_date.month,
            day=selected_date.day,
            hour=selected_time.hour,
            minute=selected_time.minute,
        )

        with st.spinner("Predicting…"):
            result = predict_congestion(
                road_id=road_id,
                timestamp=ts,
                vehicle_count=vehicle_count,
                average_speed=average_speed,
                weather=weather,
                holiday_flag=holiday_flag,
                special_event=special_event,
                model=model,
                feature_columns=feature_columns,
            )

        # ── Result Card ──────────────────────────────
        level  = result["congestion_level"]
        icon   = result["icon"]
        color  = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}[level]

        st.markdown(
            f"""
            <div style="background:{color}22; border-left:6px solid {color};
                        padding:20px; border-radius:8px; margin:10px 0;">
                <h2 style="color:{color};">{icon} Congestion Level: {level}</h2>
                <p><b>Road:</b> {road_label}</p>
                <p><b>Time:</b> {result['timestamp']}</p>
                <p><b>Estimated Vehicles:</b> ~{result['estimated_vehicles']}</p>
                <p><b>Suggested Action:</b> {result['suggested_action']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Probability Bar ──────────────────────────
        if "probabilities" in result:
            st.subheader("Prediction Confidence")
            proba = result["probabilities"]
            prob_df = pd.DataFrame(
                {"Congestion Level": list(proba.keys()),
                 "Probability": list(proba.values())}
            )
            fig, ax = plt.subplots(figsize=(6, 2))
            bars = ax.barh(prob_df["Congestion Level"],
                           prob_df["Probability"],
                           color=["#28a745", "#ffc107", "#dc3545"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            for bar, val in zip(bars, prob_df["Probability"]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1%}", va="center")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────
# Hourly Forecast Chart
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Hourly Traffic Forecast — Today")

hours = list(range(0, 24))
# Simulate hourly vehicle counts (heuristic)
base = []
for h in hours:
    if 7 <= h <= 9 or 17 <= h <= 19:
        base.append(np.random.randint(450, 680))
    elif 10 <= h <= 16:
        base.append(np.random.randint(200, 400))
    elif 20 <= h <= 23:
        base.append(np.random.randint(100, 250))
    else:
        base.append(np.random.randint(20, 80))

def count_to_level(c):
    if c > 500: return "High"
    if c > 280: return "Medium"
    return "Low"

levels  = [count_to_level(c) for c in base]
colors  = [{"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}[l] for l in levels]

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.bar(hours, base, color=colors, edgecolor="white", width=0.8)
ax2.set_xticks(hours)
ax2.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Estimated Vehicles")
ax2.set_title(f"Hourly Vehicle Count Forecast — {road_label}")
ax2.grid(axis="y", alpha=0.3)

patches = [
    mpatches.Patch(color="#28a745", label="Low"),
    mpatches.Patch(color="#ffc107", label="Medium"),
    mpatches.Patch(color="#dc3545", label="High"),
]
ax2.legend(handles=patches, loc="upper right")
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ─────────────────────────────────────────────
# Congestion Heatmap (Roads × Hours)
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("🗺️ Congestion Heatmap — All Roads × Hour of Day")

np.random.seed(0)
road_names = list(ROAD_OPTIONS.keys())
heatmap_data = []
for r in road_names:
    row = []
    for h in range(24):
        if 7 <= h <= 9 or 17 <= h <= 19:
            row.append(np.random.uniform(0.6, 1.0))
        elif 10 <= h <= 16:
            row.append(np.random.uniform(0.3, 0.7))
        else:
            row.append(np.random.uniform(0.0, 0.3))
    heatmap_data.append(row)

hm_df = pd.DataFrame(heatmap_data,
                     index=road_names,
                     columns=[f"{h:02d}:00" for h in range(24)])

fig3, ax3 = plt.subplots(figsize=(14, 4))
sns.heatmap(hm_df, cmap="RdYlGn_r", ax=ax3,
            linewidths=0.3, cbar_kws={"label": "Congestion Index"})
ax3.set_title("Congestion Intensity — Roads × Hours")
ax3.set_xlabel("Hour of Day")
plt.tight_layout()
st.pyplot(fig3)
plt.close()

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("🚦 Intelligent Traffic Congestion Prediction System | Built with Streamlit & scikit-learn")