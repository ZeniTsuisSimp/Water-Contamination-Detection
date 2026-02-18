import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load Data
try:
    df = pd.read_csv('Dataset/water_potability.csv')
    print("Dataset loaded.")
except FileNotFoundError:
    print("Dataset not found!")
    exit()

# Handle Missing Values (Mean Imputation for all)
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Scenario 1: pH and Solids ONLY
X_2 = df_filled[['ph', 'Solids']]
y = df_filled['Potability']
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y, test_size=0.2, random_state=42)

rf_2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_2.fit(X_train_2, y_train_2)
acc_2 = accuracy_score(y_test_2, rf_2.predict(X_test_2))

# Scenario 2: ALL Features
X_all = df_filled.drop('Potability', axis=1)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y, test_size=0.2, random_state=42)

rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train_all, y_train_all)
acc_all = accuracy_score(y_test_all, rf_all.predict(X_test_all))

print(f"\nAccuracy with pH + Solids: {acc_2:.4f}")
print(f"Accuracy with ALL Features: {acc_all:.4f}")
print(f"Improvement: {(acc_all - acc_2)*100:.2f}%")
