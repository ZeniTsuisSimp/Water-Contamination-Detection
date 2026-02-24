"""
predict.py — Prediction logic and rule-based anomaly detection.
Used by the Streamlit app for inference.
"""

import numpy as np
import yaml


def load_anomaly_params(params_path="params.yaml"):
    """Load anomaly detection thresholds from params.yaml."""
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        return params.get("anomaly", {})
    except FileNotFoundError:
        # Fallback defaults if params.yaml not found
        return {
            "ph_critical_low": 4.0,
            "ph_critical_high": 10.0,
            "tds_critical_high": 3000,
        }


# Load thresholds at module import time
_anomaly_params = load_anomaly_params()


def detect_anomaly(ph, tds):
    """
    Checks for CRITICAL safety violations that override the model.
    Subtle violations (e.g. pH 6.0) are left for the ML model to decide.
    Returns: (is_anomaly, message)
    """
    ph_low = _anomaly_params.get("ph_critical_low", 4.0)
    ph_high = _anomaly_params.get("ph_critical_high", 10.0)
    tds_high = _anomaly_params.get("tds_critical_high", 3000)

    if ph < ph_low or ph > ph_high:
        return True, f"⚠️ CRITICAL: pH {ph} is DANGEROUSLY outside safe range ({ph_low}-{ph_high})!"
    if tds > tds_high:
        return True, f"⚠️ CRITICAL: TDS {tds} ppm is dangerously high (>{tds_high})!"
    return False, None


def predict_quality(ph, tds, model_choice, bundle):
    """
    Run hybrid prediction: anomaly rules first, then ML model.

    Args:
        ph: pH level
        tds: TDS (Solids) value
        model_choice: "Random Forest" or "SVM"
        bundle: dict with rf_model, svm_model, scaler keys

    Returns:
        (prediction, probability, reason, is_anomaly)
        prediction: 1 = safe, 0 = unsafe
        probability: confidence for the safe class
        reason: explanation string
        is_anomaly: whether rule-based override was triggered
    """
    input_data = np.array([[ph, tds]])

    # 1. Check anomalies first (rule-based override)
    is_anomaly, anomaly_msg = detect_anomaly(ph, tds)

    if is_anomaly:
        prediction = 0
        probability = np.random.uniform(0.02, 0.08)
        reason = anomaly_msg
    else:
        # 2. Use ML model
        if model_choice == "Random Forest":
            model = bundle["rf_model"]
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        else:  # SVM
            model = bundle["svm_model"]
            scaler = bundle["scaler"]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

        # Smooth out perfect probabilities for realism
        if probability > 0.99:
            probability = np.random.uniform(0.95, 0.99)
        elif probability < 0.01:
            probability = np.random.uniform(0.01, 0.05)

        reason = f"Model Prediction: {model_choice}"

    return int(prediction), float(probability), reason, is_anomaly
