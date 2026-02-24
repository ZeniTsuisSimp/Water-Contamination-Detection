"""
data_preprocessing.py — Data loading, imputation, and train/test splitting.
Reads configuration from params.yaml.
"""

import pandas as pd
import numpy as np
import json
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_params(params_path="params.yaml"):
    """Load parameters from params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path):
    """Load dataset from CSV."""
    df = pd.read_csv(data_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def preprocess(df, features, target):
    """
    Impute missing values and return clean X, y arrays.
    Handles text labels (Safe/Unsafe) by encoding to binary (1/0).
    Returns: X (DataFrame), y (Series), imputer (fitted SimpleImputer)
    """
    # Encode text labels to binary if needed
    if df[target].dtype == "object":
        # Drop Unknown/ambiguous labels
        df = df[df[target].isin(["Safe", "Unsafe"])].copy()
        df[target] = df[target].map({"Safe": 1, "Unsafe": 0})
        print(f"Encoded '{target}': Safe→1, Unsafe→0 ({len(df)} samples after filtering)")

    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df[features]), columns=features
    )
    df_imputed[target] = df[target].values

    X = df_imputed[features]
    y = df_imputed[target]
    return X, y, imputer


def split_data(X, y, test_size=0.2, random_state=42):
    """Split into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_scaler(X_train):
    """Fit StandardScaler on training data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled


def add_noise(df, ph_std=1.5, tds_std=80.0, seed=42):
    """
    Inject Gaussian noise into pH and TDS columns to combat overfitting.
    Returns: modified DataFrame (copy).
    """
    df = df.copy()
    np.random.seed(seed)

    df["pH"] = pd.to_numeric(df["pH"], errors="coerce").fillna(df["pH"].mean())
    df["TDS"] = pd.to_numeric(df["TDS"], errors="coerce").fillna(df["TDS"].mean())

    df["pH"] += np.random.normal(0, ph_std, df.shape[0])
    df["TDS"] += np.random.normal(0, tds_std, df.shape[0])

    df["pH"] = df["pH"].clip(0, 14)
    df["TDS"] = df["TDS"].clip(0, None)
    return df


if __name__ == "__main__":
    # ── Standalone: preprocess data and save summary ──
    params = load_params()

    data_path = params["data"]["processed_path"]
    features = params["features"]["names"]
    target = params["features"]["target"]

    df = load_data(data_path)
    X, y, imputer = preprocess(df, features, target)

    summary = {
        "total_samples": int(len(X)),
        "features": features,
        "target": target,
        "class_distribution": y.value_counts().to_dict(),
        "feature_stats": {
            col: {
                "mean": round(float(X[col].mean()), 4),
                "std": round(float(X[col].std()), 4),
                "min": round(float(X[col].min()), 4),
                "max": round(float(X[col].max()), 4),
            }
            for col in features
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Data summary saved to reports/data_summary.json")
    print(json.dumps(summary, indent=2))
