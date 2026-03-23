"""
run_pipeline.py
Master script — runs the full ML pipeline end to end.

Steps:
  1. Generate / load raw data
  2. Clean & encode
  3. Feature engineering
  4. Train all models
  5. Evaluate & compare
  6. Demo prediction

Usage:
    python run_pipeline.py
"""

import os
import sys
import pandas as pd

# Ensure src/ is on the path
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)


def banner(step: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {step}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# STEP 1 — Data Pipeline
# ─────────────────────────────────────────────
banner("1 — Data Pipeline")
from data_pipeline import (
    generate_synthetic_dataset,
    load_data, clean_data, encode_target, save_processed,
)

RAW_PATH       = "data/raw/traffic_data.csv"
PROCESSED_PATH = "data/processed/processed_traffic_data.csv"

if not os.path.exists(RAW_PATH):
    generate_synthetic_dataset(n_days=90, output_path=RAW_PATH)

df_raw   = load_data(RAW_PATH)
df_clean = clean_data(df_raw)
df_enc   = encode_target(df_clean)
save_processed(df_enc, PROCESSED_PATH)


# ─────────────────────────────────────────────
# STEP 2 — Feature Engineering
# ─────────────────────────────────────────────
banner("2 — Feature Engineering")
from feature_engineering import build_features, split_features_target

df_enc["timestamp"] = pd.to_datetime(df_enc["timestamp"])
df_feat = build_features(df_enc)

FEATURES_PATH = "data/processed/features.csv"
os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
df_feat.to_csv(FEATURES_PATH, index=False)
print(f"[INFO] Features saved → {FEATURES_PATH}")

X_train, X_test, y_train, y_test = split_features_target(df_feat)


# ─────────────────────────────────────────────
# STEP 3 — Training
# ─────────────────────────────────────────────
banner("3 — Model Training")
from train_model import train_all, save_model, save_feature_columns

trained_models = train_all(X_train, y_train)

for name, mdl in trained_models.items():
    save_model(mdl, name)

save_feature_columns(list(X_train.columns))

# Save best-model alias
best_name = "XGBoost" if "XGBoost" in trained_models else "RandomForest"
save_model(trained_models[best_name], "congestion_model")
print(f"[INFO] Best model alias: congestion_model → {best_name}")


# ─────────────────────────────────────────────
# STEP 4 — Evaluation
# ─────────────────────────────────────────────
banner("4 — Model Evaluation")
from evaluate_model import (
    evaluate_classifier,
    plot_confusion_matrix,
    plot_model_comparison,
    save_evaluation_report,
    plot_feature_importance,
)
from train_model import load_feature_columns

all_results = []
for mname, mdl in trained_models.items():
    metrics, y_pred = evaluate_classifier(mdl, X_test, y_test, mname)
    all_results.append(metrics)
    plot_confusion_matrix(
        y_test, y_pred,
        model_name=mname,
        save_path=f"reports/confusion_{mname}.png",
    )

plot_model_comparison(all_results, save_path="reports/model_comparison.png")
save_evaluation_report(all_results)

feat_cols = load_feature_columns()
plot_feature_importance(
    trained_models[best_name],
    feat_cols,
    save_path=f"reports/feature_importance_{best_name}.png",
)


# ─────────────────────────────────────────────
# STEP 5 — Demo Predictions
# ─────────────────────────────────────────────
banner("5 — Demo Predictions")
from predict import predict_congestion, print_prediction
from train_model import load_model

model = load_model("congestion_model")

scenarios = [
    dict(road_id="R101", timestamp=pd.Timestamp("2024-06-10 08:30"),
         vehicle_count=620, average_speed=18.0, weather="Clear"),
    dict(road_id="R102", timestamp=pd.Timestamp("2024-06-10 14:00"),
         vehicle_count=210, average_speed=65.0, weather="Cloudy"),
    dict(road_id="R103", timestamp=pd.Timestamp("2024-06-10 17:45"),
         vehicle_count=510, average_speed=22.0, weather="Rainy"),
]

for sc in scenarios:
    result = predict_congestion(**sc, model=model, feature_columns=feat_cols)
    print_prediction(result)


print("\n✅  Full pipeline complete.")
print("👉  Launch dashboard with:  streamlit run app/streamlit_dashboard.py")