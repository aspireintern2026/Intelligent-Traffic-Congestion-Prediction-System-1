# 🚦 Intelligent Traffic Congestion Prediction System

> Predicts traffic congestion patterns using historical traffic data to help city planners, drivers, and traffic control authorities manage traffic flow efficiently.

---

## 📁 Project Structure

```
traffic_congestion_prediction_system/
├── data/
│   ├── raw/                     # Raw traffic CSV
│   └── processed/               # Cleaned & feature-engineered data
├── notebooks/
│   └── eda.py                   # Exploratory Data Analysis
├── src/
│   ├── data_pipeline.py         # Load, clean, encode data
│   ├── feature_engineering.py   # Temporal, lag, rolling features
│   ├── train_model.py           # Train & save models
│   ├── evaluate_model.py        # Metrics, plots, reports
│   └── predict.py               # Inference pipeline
├── models/
│   ├── congestion_model.pkl     # Best model alias
│   ├── RandomForest.pkl
│   ├── XGBoost.pkl
│   └── feature_columns.pkl
├── app/
│   └── streamlit_dashboard.py   # Interactive web UI
├── reports/                     # Charts & HTML reports
├── run_pipeline.py              # Master script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```

This will:
- Generate a synthetic 90-day traffic dataset
- Clean and engineer features
- Train Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- Evaluate all models and save comparison charts
- Run three demo predictions

### 3. Launch the dashboard
```bash
streamlit run app/streamlit_dashboard.py
```

### 4. Exploratory Data Analysis
```bash
python notebooks/eda.py
```

---

## 🧠 ML Pipeline

```
Traffic Dataset
    ↓
Data Cleaning          (data_pipeline.py)
    ↓
Feature Engineering    (feature_engineering.py)
    ↓
Traffic Prediction     (train_model.py)
    ↓
Evaluation             (evaluate_model.py)
    ↓
Congestion Dashboard   (streamlit_dashboard.py)
```

---

## 📊 Features Used

| Feature               | Description                           |
|-----------------------|---------------------------------------|
| `hour_of_day`         | Hour extracted from timestamp         |
| `day_of_week`         | 0=Monday … 6=Sunday                   |
| `weekend_flag`        | 1 if Saturday/Sunday                  |
| `vehicle_count`       | Current vehicle count on road         |
| `average_speed`       | Average vehicle speed (km/h)          |
| `vehicle_count_lag_N` | Vehicle count N intervals ago         |
| `rolling_avg_N`       | Rolling mean over last N intervals    |
| `traffic_growth_rate` | % change vs previous interval        |
| `speed_drop_rate`     | Proxy for congestion intensity        |
| `volume_capacity_ratio` | Vehicles / Road capacity (700)      |
| `weather_*`           | One-hot encoded weather condition     |
| `event_*`             | One-hot encoded special event         |

---

## 🎯 Models & Performance (typical)

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~78%     |
| Random Forest       | ~86%     |
| Gradient Boosting   | ~88%     |
| **XGBoost**         | **~92%** |

---

## 📈 Evaluation Metrics

**Classification:** Accuracy, Precision, Recall, F1-Score  
**Regression proxy:** MAE, RMSE, MAPE

---

## 🔮 Example Output

```
=============================================
  TRAFFIC CONGESTION PREDICTION
=============================================
  Road       : R101
  Time       : 2024-06-10 08:30
  Vehicles   : ~620
  Congestion : 🔴 High
  Confidence : Low 2%  Med 8%  High 90%
  Action     : Use alternate route via R103 via Outer Ring Road
=============================================
```

---

## 🔧 Advanced Enhancements

- **Traffic Heatmap** — `folium` / `geopandas` geospatial visualization
- **Route Optimization** — Suggest alternate routes for high congestion
- **Event-based Prediction** — Spikes during festivals/matches
- **Explainability** — SHAP values to identify key congestion factors
- **LSTM / GRU** — Deep learning temporal models (see `train_model.py`)

---

## 📦 Dataset

The project auto-generates a synthetic dataset if no real data is present.  
For real-world use, recommended datasets:
- **METR-LA** Traffic Dataset
- **Urban Traffic Dataset** (UCI / Kaggle)

Place your CSV at `data/raw/traffic_data.csv` with columns:

```
timestamp, road_id, vehicle_count, average_speed, congestion_level,
weather_condition, day_of_week, holiday_flag, special_events
```

---

*Allotted to: Thiruvarun.M | Aspire Code AI*