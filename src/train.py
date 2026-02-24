"""
train.py — Train RF and SVM models with MLflow experiment tracking.
Reads hyperparameters from params.yaml, saves model bundle to water_model.pkl.
"""

import json
import os
import pickle

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data_preprocessing import load_params, load_data, preprocess, split_data, fit_scaler

# ── Try to import MLflow (optional dependency) ──
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not installed. Training will proceed without experiment tracking.")


def train_models(params):
    """Train Random Forest and SVM models, log to MLflow, and save bundle."""
    # ── Load & preprocess ──
    data_path = params["data"]["processed_path"]
    features = params["features"]["names"]
    target = params["features"]["target"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    df = load_data(data_path)
    X, y, imputer = preprocess(df, features, target)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    scaler, X_train_scaled = fit_scaler(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Train Random Forest ──
    rf_params = params["model"]["rf"]
    rf_model = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        random_state=rf_params["random_state"],
    )
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f}")

    # ── Train SVM ──
    svm_params = params["model"]["svm"]
    svm_model = SVC(
        kernel=svm_params["kernel"],
        probability=svm_params["probability"],
        random_state=svm_params["random_state"],
    )
    svm_model.fit(X_train_scaled, y_train)
    svm_accuracy = svm_model.score(X_test_scaled, y_test)
    print(f"✅ SVM Accuracy: {svm_accuracy:.4f}")

    # ── MLflow Logging ──
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("water-contamination-detection")
        with mlflow.start_run(run_name="rf_svm_training"):
            # Log params
            mlflow.log_param("rf_n_estimators", rf_params["n_estimators"])
            mlflow.log_param("svm_kernel", svm_params["kernel"])
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("features", features)

            # Log metrics
            mlflow.log_metric("rf_accuracy", rf_accuracy)
            mlflow.log_metric("svm_accuracy", svm_accuracy)

            # Log models
            mlflow.sklearn.log_model(rf_model, "rf_model")
            mlflow.sklearn.log_model(svm_model, "svm_model")

            print("✅ Logged experiment to MLflow")

    # ── Save model bundle ──
    model_path = params["output"]["model_path"]
    bundle = {
        "rf_model": rf_model,
        "svm_model": svm_model,
        "scaler": scaler,
        "imputer": imputer,
        "features": features,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"✅ Model bundle saved to {model_path}")

    # ── Save training metrics ──
    metrics = {
        "rf_accuracy": round(rf_accuracy, 4),
        "svm_accuracy": round(svm_accuracy, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    os.makedirs("reports", exist_ok=True)
    metrics_path = params["output"]["metrics_path"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")

    return bundle, metrics


if __name__ == "__main__":
    params = load_params()
    train_models(params)
