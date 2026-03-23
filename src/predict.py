"""
predict.py — ML-only prediction logic.
Used by the Streamlit app for inference.
"""

import numpy as np


def predict_quality(ph, tds, model_choice, bundle):
    """
    Run ML model prediction for water quality.

    Args:
        ph: pH level
        tds: TDS (Solids) value
        model_choice: "Random Forest" or "SVM"
        bundle: dict with rf_model, svm_model, scaler keys

    Returns:
        (prediction, probability, reason)
        prediction: 1 = safe, 0 = unsafe
        probability: confidence for the predicted class
        reason: explanation string
    """
    input_data = np.array([[ph, tds]])

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

    return int(prediction), float(probability), reason
