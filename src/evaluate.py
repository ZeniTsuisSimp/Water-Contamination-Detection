"""
evaluate.py — Evaluate the trained model and generate reports.
Produces classification reports, confusion matrix plots, and metrics JSON.
"""

import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.data_preprocessing import load_params, load_data, preprocess, split_data, fit_scaler


def evaluate_models(params):
    """Load saved model bundle, evaluate on test set, save reports."""
    # ── Load data ──
    data_path = params["data"]["processed_path"]
    features = params["features"]["names"]
    target = params["features"]["target"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    df = load_data(data_path)
    X, y, _ = preprocess(df, features, target)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    # ── Load model bundle ──
    model_path = params["output"]["model_path"]
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    rf_model = bundle["rf_model"]
    svm_model = bundle["svm_model"]
    scaler = bundle["scaler"]

    # ── Evaluate Random Forest ──
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, output_dict=True)
    print(f"✅ Random Forest Test Accuracy: {rf_accuracy:.4f}")
    print(classification_report(y_test, rf_pred))

    # ── Evaluate SVM ──
    X_test_scaled = scaler.transform(X_test)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_report = classification_report(y_test, svm_pred, output_dict=True)
    print(f"✅ SVM Test Accuracy: {svm_accuracy:.4f}")
    print(classification_report(y_test, svm_pred))

    # ── Save evaluation metrics ──
    os.makedirs("reports", exist_ok=True)

    eval_metrics = {
        "rf_test_accuracy": round(rf_accuracy, 4),
        "svm_test_accuracy": round(svm_accuracy, 4),
        "rf_classification_report": rf_report,
        "svm_classification_report": svm_report,
    }
    eval_path = params["output"]["eval_metrics_path"]
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"✅ Evaluation metrics saved to {eval_path}")

    # ── Save confusion matrix plots ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, rf_pred, display_labels=["Unsafe", "Safe"], ax=axes[0], cmap="Blues"
    )
    axes[0].set_title("Random Forest — Confusion Matrix")

    ConfusionMatrixDisplay.from_predictions(
        y_test, svm_pred, display_labels=["Unsafe", "Safe"], ax=axes[1], cmap="Oranges"
    )
    axes[1].set_title("SVM — Confusion Matrix")

    plt.tight_layout()
    plot_path = "reports/confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix plot saved to {plot_path}")

    return eval_metrics


if __name__ == "__main__":
    params = load_params()
    evaluate_models(params)
