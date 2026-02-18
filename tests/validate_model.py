"""
validate_model.py — CI step to verify the trained model bundle loads correctly.
The model is saved as a dict with keys: rf_model, svm_model, scaler, imputer, features.
"""
import pickle
import os
import sys

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'water_model.pkl')

def validate():
    # 1. Check model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at: {MODEL_PATH}")
        sys.exit(1)
    print(f"✅ Model file found: {MODEL_PATH}")

    # 2. Check file is not empty
    size = os.path.getsize(MODEL_PATH)
    if size == 0:
        print("❌ Model file is empty (0 bytes)")
        sys.exit(1)
    print(f"✅ Model file size: {size / 1024:.1f} KB")

    # 3. Try to load with pickle
    try:
        with open(MODEL_PATH, 'rb') as f:
            bundle = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    print(f"✅ Model bundle loaded successfully: {type(bundle).__name__}")

    # 4. Verify it's a dict with expected keys
    if not isinstance(bundle, dict):
        print(f"❌ Expected a dict, got {type(bundle).__name__}")
        sys.exit(1)

    required_keys = ['rf_model', 'svm_model', 'scaler']
    for key in required_keys:
        if key not in bundle:
            print(f"❌ Missing required key: '{key}'")
            sys.exit(1)
        print(f"✅ Found key '{key}': {type(bundle[key]).__name__}")

    # 5. Verify models have predict methods
    for model_key in ['rf_model', 'svm_model']:
        model = bundle[model_key]
        if not hasattr(model, 'predict'):
            print(f"❌ {model_key} does not have a 'predict' method")
            sys.exit(1)
        print(f"✅ {model_key} has 'predict' method")

    # 6. Quick prediction sanity check
    try:
        import numpy as np
        test_input = np.array([[7.0, 500]])
        rf_pred = bundle['rf_model'].predict(test_input)
        print(f"✅ RF test prediction (pH=7, TDS=500): {rf_pred[0]}")

        scaled_input = bundle['scaler'].transform(test_input)
        svm_pred = bundle['svm_model'].predict(scaled_input)
        print(f"✅ SVM test prediction (pH=7, TDS=500): {svm_pred[0]}")
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        sys.exit(1)

    print("\n🎉 All model validations passed!")

if __name__ == '__main__':
    validate()
