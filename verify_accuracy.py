import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load dataset
file_path = 'Dataset/water_classified_noisy.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Handle missing values in Target
df = df.dropna(subset=['Classification'])

# Clean Classification column
df['Classification'] = df['Classification'].astype(str).str.strip()
print(f"Unique Classification values: {df['Classification'].unique()}")

# Features and Target
X = df[['pH', 'TDS']]
y = df['Classification']

# Handle missing values in Features
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encode Target
y = y.map({'Safe': 1, 'Unsafe': 0})

# Check for NaNs in y after mapping
if y.isna().any():
    print("NaNs found in target after mapping. Checking unmapped values...")
    print(df.loc[y.isna(), 'Classification'].unique())
    # Drop rows where mapping failed
    valid_indices = y.dropna().index
    X = X.loc[y.index.isin(valid_indices)] #Align X with valid y
    y = y.dropna()


# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate Accuracies
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

output = []
output.append(f"Training Accuracy: {train_acc*100:.2f}%")
output.append(f"Testing Accuracy: {test_acc*100:.2f}%")

if train_acc > 0.99 and test_acc > 0.99:
    output.append("WARNING: Both accuracies are extremely high. This might indicate simple separation or data leakage.")
elif train_acc > 0.99 and test_acc < 0.9:
    output.append("WARNING: High training accuracy but lower testing accuracy indicates Overfitting.")

with open('verification_results.txt', 'w') as f:
    f.write('\n'.join(output))

print('\n'.join(output))
