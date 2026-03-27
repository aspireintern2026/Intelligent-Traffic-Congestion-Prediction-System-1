"""
streamlit_dashboard.py  (with SQLite CRUD)
Run with:
    streamlit run src/streamlit_dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, time, date

# ── DB import ───────────────────────────────────────────────────────────────
from database import (
    init_db, seed_roads,
    # Roads
    create_road, read_all_roads, read_road, update_road, delete_road,
    # Traffic Records
    create_traffic_record, read_traffic_records,
    read_traffic_record_by_id, update_traffic_record, delete_traffic_record,
    # Predictions
    save_prediction, read_predictions, delete_prediction, clear_old_predictions,
    # Analytics
    get_congestion_summary, get_hourly_avg, get_db_stats,
)

# ── Bootstrap DB ─────────────────────────────────────────────────────────────
init_db()
seed_roads()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load ML model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    try:
        from train_model import load_model, load_feature_columns
        m  = load_model("congestion_model")
        fc = load_feature_columns()
        return m, fc, True
    except Exception:
        return None, None, False

model, feature_columns, model_loaded = load_resources()

# ── Helpers ───────────────────────────────────────────────────────────────────
LEVEL_COLOR = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
LEVEL_ICON  = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

def badge(level: str) -> str:
    c = LEVEL_COLOR.get(level, "#999")
    return f'<span style="background:{c};color:white;padding:2px 10px;border-radius:12px;font-size:12px;">{level}</span>'

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🚦 Traffic System")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "🔮 Predict Congestion",
    "🛣️ Manage Roads",
    "📋 Traffic Records",
    "📊 Prediction History",
    "📈 Analytics",
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🚦 Intelligent Traffic Congestion Prediction System")
    st.markdown("Real-time insights powered by **SQLite** + **Machine Learning**")
    st.markdown("---")

    stats = get_db_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("🛣️ Roads Monitored",    stats["roads"])
    c2.metric("📋 Traffic Records",    stats["traffic_records"])
    c3.metric("🔮 Predictions Saved",  stats["predictions"])

    st.markdown("---")

    # Congestion summary chart
    summary = get_congestion_summary()
    if not summary.empty:
        st.subheader("Congestion Distribution by Road")
        pivot = summary.pivot_table(
            index="road_id", columns="congestion_level", values="count", fill_value=0
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        pivot.plot(kind="bar", ax=ax,
                   color=[LEVEL_COLOR.get(c, "#999") for c in pivot.columns],
                   edgecolor="white")
        ax.set_xlabel("Road"); ax.set_ylabel("Records"); ax.set_xticklabels(
            pivot.index, rotation=0)
        ax.legend(title="Level"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    else:
        st.info("No traffic records yet — add some in **Traffic Records** tab.")

    # Hourly heatmap
    hourly = get_hourly_avg()
    if not hourly.empty:
        st.subheader("Hourly Average Vehicle Count — All Roads")
        pivot_h = hourly.pivot_table(index="road_id", columns="hour",
                                      values="avg_vehicles", fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(14, 3))
        sns.heatmap(pivot_h, cmap="RdYlGn_r", ax=ax2,
                    linewidths=0.3, cbar_kws={"label": "Avg Vehicles"})
        ax2.set_title("Road × Hour Heatmap")
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT CONGESTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Congestion":
    st.title("🔮 Predict Congestion")
    st.markdown("Fill in road details and get an AI-powered prediction.")
    st.markdown("---")

    roads_df = read_all_roads()
    road_options = roads_df["road_id"].tolist() if not roads_df.empty else ["R101"]

    col1, col2 = st.columns(2)
    with col1:
        road_id       = st.selectbox("Road", road_options)
        pred_date     = st.date_input("Date", value=date.today())
        pred_time     = st.time_input("Time", value=time(8, 30))
        vehicle_count = st.number_input("Vehicle Count", 0, 1200, 400, 10)
    with col2:
        average_speed = st.number_input("Average Speed (km/h)", 0.0, 140.0, 35.0, 1.0)
        weather       = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy", "Foggy"])
        holiday_flag  = int(st.checkbox("Public Holiday"))
        special_event = st.selectbox("Special Event", ["None","Festival","Match","Concert"])

    if st.button("🔍 Predict", use_container_width=True):
        if not model_loaded:
            st.error("Model not found. Run `run_pipeline.py` first.")
        else:
            from predict import predict_congestion
            ts = pd.Timestamp(year=pred_date.year, month=pred_date.month,
                              day=pred_date.day, hour=pred_time.hour,
                              minute=pred_time.minute)
            result = predict_congestion(
                road_id=road_id, timestamp=ts,
                vehicle_count=vehicle_count, average_speed=average_speed,
                weather=weather, holiday_flag=holiday_flag,
                special_event=special_event,
                model=model, feature_columns=feature_columns,
            )

            level = result["congestion_level"]
            color = LEVEL_COLOR[level]
            st.markdown(f"""
            <div style="background:{color}22;border-left:6px solid {color};
                        padding:20px;border-radius:8px;margin:12px 0">
                <h2 style="color:{color};">{LEVEL_ICON[level]} {level} Congestion</h2>
                <b>Road:</b> {road_id} &nbsp;|&nbsp;
                <b>Time:</b> {result['timestamp']} &nbsp;|&nbsp;
                <b>Vehicles:</b> ~{vehicle_count}<br><br>
                <b>Action:</b> {result['suggested_action']}
            </div>""", unsafe_allow_html=True)

            # Confidence bars
            if "probabilities" in result:
                p = result["probabilities"]
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.barh(list(p.keys()), list(p.values()),
                        color=[LEVEL_COLOR[k] for k in p])
                ax.set_xlim(0, 1); ax.set_xlabel("Probability")
                for i, (k, v) in enumerate(p.items()):
                    ax.text(v + 0.01, i, f"{v:.0%}", va="center")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            # Save to DB
            proba = result.get("probabilities", {})
            save_prediction(
                road_id=road_id,
                timestamp=result["timestamp"],
                vehicle_count=vehicle_count,
                average_speed=average_speed,
                predicted_level=level,
                prob_low=proba.get("Low"),
                prob_medium=proba.get("Medium"),
                prob_high=proba.get("High"),
                suggested_action=result["suggested_action"],
            )
            st.success("✅ Prediction saved to database!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MANAGE ROADS  (CRUD)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛣️ Manage Roads":
    st.title("🛣️ Manage Roads")
    tab_view, tab_add, tab_edit, tab_del = st.tabs(["📋 View All", "➕ Add", "✏️ Edit", "🗑️ Delete"])

    # ── View ────────────────────────────────
    with tab_view:
        df = read_all_roads()
        if df.empty:
            st.info("No roads found.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Add ─────────────────────────────────
    with tab_add:
        st.subheader("Add New Road")
        with st.form("add_road_form"):
            new_id   = st.text_input("Road ID (e.g. R106)")
            new_name = st.text_input("Road Name")
            new_type = st.selectbox("Road Type", ["Urban","Highway","Ring","Expressway"])
            new_cap  = st.number_input("Capacity (vehicles)", 100, 2000, 700, 50)
            submitted = st.form_submit_button("Add Road")
        if submitted:
            if not new_id or not new_name:
                st.warning("Road ID and Name are required.")
            elif create_road(new_id.strip(), new_name.strip(), new_type, new_cap):
                st.success(f"✅ Road **{new_id}** added!")
                st.rerun()
            else:
                st.error(f"Road ID **{new_id}** already exists.")

    # ── Edit ────────────────────────────────
    with tab_edit:
        st.subheader("Edit Road")
        df = read_all_roads()
        if df.empty:
            st.info("No roads to edit.")
        else:
            road_to_edit = st.selectbox("Select Road", df["road_id"].tolist(), key="edit_road_sel")
            road_data = read_road(road_to_edit)
            if road_data:
                with st.form("edit_road_form"):
                    e_name = st.text_input("Road Name", road_data["road_name"])
                    e_type = st.selectbox("Road Type",
                                          ["Urban","Highway","Ring","Expressway"],
                                          index=["Urban","Highway","Ring","Expressway"].index(
                                              road_data.get("road_type","Urban")))
                    e_cap  = st.number_input("Capacity", 100, 2000,
                                              int(road_data.get("capacity", 700)), 50)
                    save   = st.form_submit_button("💾 Save Changes")
                if save:
                    if update_road(road_to_edit, e_name, e_type, e_cap):
                        st.success("✅ Road updated!")
                        st.rerun()
                    else:
                        st.error("Update failed.")

    # ── Delete ──────────────────────────────
    with tab_del:
        st.subheader("Delete Road")
        df = read_all_roads()
        if df.empty:
            st.info("No roads to delete.")
        else:
            road_to_del = st.selectbox("Select Road to Delete",
                                        df["road_id"].tolist(), key="del_road_sel")
            st.warning(f"⚠️ This will delete road **{road_to_del}**. This cannot be undone.")
            if st.button("🗑️ Delete Road", type="primary"):
                if delete_road(road_to_del):
                    st.success(f"✅ Road **{road_to_del}** deleted.")
                    st.rerun()
                else:
                    st.error("Deletion failed (may have linked records).")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TRAFFIC RECORDS  (CRUD)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Traffic Records":
    st.title("📋 Traffic Records")
    tab_view, tab_add, tab_edit, tab_del = st.tabs(["📋 View & Filter","➕ Add","✏️ Edit","🗑️ Delete"])

    roads_df = read_all_roads()
    road_list = roads_df["road_id"].tolist() if not roads_df.empty else []

    # ── View & Filter ────────────────────────
    with tab_view:
        st.subheader("Filter Records")
        fc1, fc2, fc3 = st.columns(3)
        f_road  = fc1.selectbox("Road", ["All"] + road_list, key="f_road")
        f_level = fc2.selectbox("Congestion", ["All","Low","Medium","High"], key="f_level")
        f_limit = fc3.number_input("Max rows", 10, 1000, 100, 10)

        df = read_traffic_records(
            road_id=None if f_road == "All" else f_road,
            congestion_level=None if f_level == "All" else f_level,
            limit=int(f_limit),
        )
        if df.empty:
            st.info("No records found.")
        else:
            st.write(f"**{len(df)} records**")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Export CSV", csv, "traffic_records.csv", "text/csv")

    # ── Add ─────────────────────────────────
    with tab_add:
        st.subheader("Add Traffic Record")
        with st.form("add_record_form"):
            ar1, ar2 = st.columns(2)
            a_road    = ar1.selectbox("Road ID", road_list)
            a_ts      = ar1.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)",
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            a_vc      = ar1.number_input("Vehicle Count", 0, 1200, 300, 10)
            a_speed   = ar2.number_input("Average Speed (km/h)", 0.0, 140.0, 45.0, 1.0)
            a_level   = ar2.selectbox("Congestion Level", ["Low","Medium","High"])
            a_weather = ar2.selectbox("Weather", ["Clear","Cloudy","Rainy","Foggy"])
            a_holiday = int(st.checkbox("Holiday Flag"))
            a_event   = st.selectbox("Special Event", ["None","Festival","Match","Concert"])
            submitted = st.form_submit_button("➕ Add Record")
        if submitted:
            new_id = create_traffic_record(
                a_road, a_ts, int(a_vc), float(a_speed),
                a_level, a_weather, a_holiday, a_event,
            )
            st.success(f"✅ Record added (ID: {new_id})")
            st.rerun()

    # ── Edit ────────────────────────────────
    with tab_edit:
        st.subheader("Edit Traffic Record")
        record_id = st.number_input("Enter Record ID to Edit", min_value=1, step=1)
        if st.button("🔍 Load Record"):
            st.session_state["edit_rec"] = read_traffic_record_by_id(int(record_id))

        rec = st.session_state.get("edit_rec")
        if rec:
            st.write(f"Editing record **ID {rec['id']}** — Road: {rec['road_id']}")
            with st.form("edit_record_form"):
                er1, er2 = st.columns(2)
                e_vc      = er1.number_input("Vehicle Count", 0, 1200, int(rec["vehicle_count"]), 10)
                e_speed   = er1.number_input("Average Speed", 0.0, 140.0, float(rec["average_speed"]), 1.0)
                e_level   = er2.selectbox("Congestion Level", ["Low","Medium","High"],
                                           index=["Low","Medium","High"].index(rec["congestion_level"]))
                e_weather = er2.selectbox("Weather", ["Clear","Cloudy","Rainy","Foggy"],
                                           index=["Clear","Cloudy","Rainy","Foggy"].index(
                                               rec.get("weather_condition","Clear")))
                save = st.form_submit_button("💾 Save Changes")
            if save:
                if update_traffic_record(rec["id"], vehicle_count=int(e_vc),
                                          average_speed=float(e_speed),
                                          congestion_level=e_level,
                                          weather_condition=e_weather):
                    st.success("✅ Record updated!")
                    st.session_state.pop("edit_rec", None)
                    st.rerun()
                else:
                    st.error("Update failed.")
        elif st.session_state.get("edit_rec") is None and record_id:
            pass  # waiting for load

    # ── Delete ──────────────────────────────
    with tab_del:
        st.subheader("Delete Traffic Record")
        del_id = st.number_input("Record ID to Delete", min_value=1, step=1, key="del_rec_id")
        rec_preview = read_traffic_record_by_id(int(del_id)) if del_id else None
        if rec_preview:
            st.write(rec_preview)
            st.warning("⚠️ This will permanently delete the record.")
            if st.button("🗑️ Delete Record", type="primary"):
                if delete_traffic_record(int(del_id)):
                    st.success(f"✅ Record {del_id} deleted.")
                    st.rerun()
                else:
                    st.error("Deletion failed.")
        else:
            st.info("Enter a valid Record ID above.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Prediction History":
    st.title("📊 Prediction History")
    roads_df = read_all_roads()
    road_list = roads_df["road_id"].tolist() if not roads_df.empty else []

    tab_view, tab_del, tab_purge = st.tabs(["📋 View","🗑️ Delete One","🧹 Purge Old"])

    with tab_view:
        ph_road  = st.selectbox("Filter by Road", ["All"] + road_list, key="ph_road")
        ph_limit = st.number_input("Max rows", 10, 500, 50, 10)
        df = read_predictions(
            road_id=None if ph_road == "All" else ph_road,
            limit=int(ph_limit),
        )
        if df.empty:
            st.info("No predictions saved yet.")
        else:
            # Colour-code predicted_level
            st.write(f"**{len(df)} predictions**")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Pie chart
            pie = df["predicted_level"].value_counts()
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(pie.values,
                   labels=pie.index,
                   colors=[LEVEL_COLOR.get(l,"#ccc") for l in pie.index],
                   autopct="%1.0f%%", startangle=90)
            ax.set_title("Predicted Level Distribution")
            st.pyplot(fig); plt.close()

    with tab_del:
        del_pid = st.number_input("Prediction ID to Delete", min_value=1, step=1)
        if st.button("🗑️ Delete Prediction", type="primary"):
            if delete_prediction(int(del_pid)):
                st.success(f"✅ Prediction {del_pid} deleted.")
                st.rerun()
            else:
                st.error("Prediction not found.")

    with tab_purge:
        days = st.number_input("Delete predictions older than (days)", 1, 365, 30)
        if st.button("🧹 Purge Old Predictions", type="primary"):
            n = clear_old_predictions(int(days))
            st.success(f"✅ Deleted {n} old prediction(s).")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.title("📈 Analytics")
    roads_df = read_all_roads()
    road_list = roads_df["road_id"].tolist() if not roads_df.empty else []

    an_road = st.selectbox("Select Road for Hourly Analysis",
                            ["All"] + road_list, key="an_road")
    road_filter = None if an_road == "All" else an_road

    hourly = get_hourly_avg(road_filter)
    if not hourly.empty:
        # Line chart
        st.subheader("Hourly Average Vehicle Count")
        fig, ax = plt.subplots(figsize=(12, 4))
        for rid, grp in hourly.groupby("road_id"):
            ax.plot(grp["hour"], grp["avg_vehicles"], marker="o", label=rid)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Hour"); ax.set_ylabel("Avg Vehicles")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Speed chart
        st.subheader("Hourly Average Speed")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        for rid, grp in hourly.groupby("road_id"):
            ax2.plot(grp["hour"], grp["avg_speed"], marker="s", label=rid)
        ax2.set_xticks(range(24))
        ax2.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=7)
        ax2.set_xlabel("Hour"); ax2.set_ylabel("Avg Speed (km/h)")
        ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()
    else:
        st.info("No traffic records yet — add records in the **Traffic Records** tab.")

    # Summary table
    st.subheader("Congestion Summary Table")
    summary = get_congestion_summary()
    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("No data available.")

    # DB stats
    st.subheader("Database Statistics")
    stats = get_db_stats()
    s1, s2, s3 = st.columns(3)
    s1.metric("Roads",            stats["roads"])
    s2.metric("Traffic Records",  stats["traffic_records"])
    s3.metric("Predictions",      stats["predictions"])

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("🚦 Traffic Prediction System\nSQLite + scikit-learn")