import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading dataset...")
try:
    df = pd.read_csv('Dataset/water_potability.csv')
    print("Dataset loaded.")
except FileNotFoundError:
    print("Dataset not found!")
    exit(1)

# Features
features = ['ph', 'Solids']
target = 'Potability'

if set(features).issubset(df.columns) and target in df.columns:
    df_selected = df[features + [target]].copy()
    
    # Impute missing values
    df_selected['ph'] = df_selected['ph'].fillna(df_selected['ph'].mean())
    df_selected['Solids'] = df_selected['Solids'].fillna(df_selected['Solids'].mean())
    
    X = df_selected[features]
    y = df_selected[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Save
    with open('water_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to 'water_model.pkl'")
else:
    print("Required columns missing.")
