"""
train_model.py
Trains and saves traffic congestion classification models.
Supports: Logistic Regression, Random Forest, XGBoost, LightGBM.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not installed – XGBoost model will be skipped.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARN] lightgbm not installed – LightGBM model will be skipped.")


# ─────────────────────────────────────────────
# 1. Model Registry
# ─────────────────────────────────────────────

def get_models() -> dict:
    """Return a dictionary of model name → sklearn estimator / pipeline."""
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    return models


# ─────────────────────────────────────────────
# 2. Training
# ─────────────────────────────────────────────

def train_all(X_train: pd.DataFrame,
              y_train: pd.Series,
              model_names: list = None) -> dict:
    """Train all (or selected) models and return fitted objects."""
    all_models = get_models()
    if model_names:
        all_models = {k: v for k, v in all_models.items() if k in model_names}

    trained = {}
    for name, model in all_models.items():
        print(f"[TRAIN] Fitting {name} …")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"[TRAIN] {name} done.")

    return trained


# ─────────────────────────────────────────────
# 3. Saving & Loading
# ─────────────────────────────────────────────

def save_model(model, name: str, model_dir: str = "models"):
    """Pickle a trained model to disk."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[SAVE] Model saved → {path}")


def load_model(name: str, model_dir: str = "models"):
    """Load a pickled model from disk."""
    path = os.path.join(model_dir, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[LOAD] Model loaded ← {path}")
    return model


def save_feature_columns(columns: list, path: str = "models/feature_columns.pkl"):
    """Persist the list of training columns so predict.py can reindex at inference."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(columns, f)
    print(f"[SAVE] Feature columns saved → {path}")


def load_feature_columns(path: str = "models/feature_columns.pkl") -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# 4. LSTM / GRU (Deep Learning)
# ─────────────────────────────────────────────

def build_lstm_model(input_shape: tuple, n_classes: int = 3):
    """Build a simple LSTM classifier."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except ImportError:
        print("[WARN] TensorFlow not installed – LSTM model unavailable.")
        return None

    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model


def prepare_sequences(X: np.ndarray, y: np.ndarray,
                       time_steps: int = 6):
    """Reshape flat feature array into (samples, time_steps, features) for LSTM."""
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_engineering import split_features_target

    features_path = "data/processed/features.csv"
    if not os.path.exists(features_path):
        raise FileNotFoundError("Run feature_engineering.py first.")

    df = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test = split_features_target(df)

    # Train classical models
    trained_models = train_all(X_train, y_train)

    # Save all
    for model_name, model_obj in trained_models.items():
        save_model(model_obj, model_name)

    # Save feature column names for inference
    save_feature_columns(list(X_train.columns))

    # Save best model alias (XGBoost preferred, else Random Forest)
    best_name = "XGBoost" if "XGBoost" in trained_models else "RandomForest"
    save_model(trained_models[best_name], "congestion_model")
    print(f"\n[TRAIN] Best model alias → congestion_model ({best_name})")