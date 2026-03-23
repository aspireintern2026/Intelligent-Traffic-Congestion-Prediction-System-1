"""
evaluate_model.py
Evaluates trained models using classification and regression metrics.
Generates comparison report and confusion matrix visualization.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

sys.path.insert(0, os.path.dirname(__file__))
from train_model import load_model, load_feature_columns
from feature_engineering import split_features_target

CONGESTION_LABELS = ["Low", "Medium", "High"]
REPORTS_DIR = "reports"


# ─────────────────────────────────────────────
# 1. Classification Metrics
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: pd.DataFrame,
                         y_test: pd.Series,
                         model_name: str = "Model") -> dict:
    """Return dict of classification metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }

    print(f"\n{'='*50}")
    print(f" {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:12s}: {v}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=CONGESTION_LABELS,
                                 zero_division=0))

    return metrics, y_pred


# ─────────────────────────────────────────────
# 2. Regression Metrics (vehicle count)
# ─────────────────────────────────────────────

def evaluate_regressor_proxy(y_true: np.ndarray,
                              y_pred: np.ndarray) -> dict:
    """MAE / RMSE / MAPE on numeric predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAPE : {mape:.2f}%")

    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


# ─────────────────────────────────────────────
# 3. Confusion Matrix Plot
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred,
                           model_name: str = "Model",
                           save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CONGESTION_LABELS,
                yticklabels=CONGESTION_LABELS,
                ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"[PLOT] Confusion matrix saved → {save_path}")

    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 4. Model Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_model_comparison(results: list, save_path: str = None):
    """
    results: list of dicts from evaluate_classifier
    """
    df_res = pd.DataFrame(results)
    df_plot = df_res.set_index("model")[["accuracy", "precision", "recall", "f1_score"]]

    ax = df_plot.plot(kind="bar", figsize=(10, 5), colormap="tab10", edgecolor="black")
    ax.set_title("Model Comparison — Congestion Prediction")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"[PLOT] Model comparison chart saved → {save_path}")

    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 5. Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list,
                             top_n: int = 20,
                             save_path: str = None):
    """Works for tree-based models (RF, XGBoost, LightGBM, GBM)."""
    if not hasattr(model, "feature_importances_"):
        # Try to extract from sklearn Pipeline
        if hasattr(model, "named_steps"):
            inner = model.named_steps.get("clf", None)
            if inner and hasattr(inner, "feature_importances_"):
                importances = inner.feature_importances_
            else:
                print("[INFO] Feature importances not available for this model.")
                return
        else:
            print("[INFO] Feature importances not available for this model.")
            return
    else:
        importances = model.feature_importances_

    indices = np.argsort(importances)[::-1][:top_n]
    names   = [feature_names[i] for i in indices]
    values  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=values, y=names, palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"[PLOT] Feature importance chart saved → {save_path}")

    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 6. Save HTML Summary
# ─────────────────────────────────────────────

def save_evaluation_report(results: list,
                            path: str = "reports/evaluation_report.html"):
    df_res = pd.DataFrame(results)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    html = df_res.to_html(index=False, border=1, justify="center")
    with open(path, "w") as f:
        f.write(f"<html><body><h2>Traffic Model Evaluation</h2>{html}</body></html>")
    print(f"[SAVE] Evaluation report → {path}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    features_path = "data/processed/features.csv"
    if not os.path.exists(features_path):
        raise FileNotFoundError("Run feature_engineering.py first.")

    df = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test = split_features_target(df)

    model_names = ["LogisticRegression", "RandomForest", "GradientBoosting"]
    try:
        from xgboost import XGBClassifier
        model_names.append("XGBoost")
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        model_names.append("LightGBM")
    except ImportError:
        pass

    all_results = []
    for mname in model_names:
        try:
            model = load_model(mname)
        except FileNotFoundError:
            print(f"[SKIP] {mname} not found in models/ — run train_model.py first.")
            continue

        metrics, y_pred = evaluate_classifier(model, X_test, y_test, mname)
        all_results.append(metrics)

        plot_confusion_matrix(
            y_test, y_pred,
            model_name=mname,
            save_path=f"{REPORTS_DIR}/confusion_{mname}.png",
        )

    if all_results:
        plot_model_comparison(
            all_results,
            save_path=f"{REPORTS_DIR}/model_comparison.png",
        )
        save_evaluation_report(all_results)

        # Feature importance for best model
        best = "XGBoost" if "XGBoost" in model_names else "RandomForest"
        try:
            best_model = load_model(best)
            feat_cols   = load_feature_columns()
            plot_feature_importance(
                best_model, feat_cols,
                save_path=f"{REPORTS_DIR}/feature_importance_{best}.png",
            )
        except Exception as e:
            print(f"[WARN] Feature importance failed: {e}")