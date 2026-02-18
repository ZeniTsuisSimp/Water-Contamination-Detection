import pandas as pd
import numpy as np

# Load dataset
input_file = 'Dataset/water_classified_fixed.csv'
output_file = 'Dataset/water_classified_noisy.csv'

try:
    df = pd.read_csv(input_file)
    print("Dataset loaded.")
except FileNotFoundError:
    print(f"File not found: {input_file}")
    exit()

# Handle missing values first (if any leftovers)
df = df.dropna(subset=['Classification'])
# Ensure numeric types
df['pH'] = pd.to_numeric(df['pH'], errors='coerce')
df['TDS'] = pd.to_numeric(df['TDS'], errors='coerce')

# Fill NaN features with mean if any
df['pH'] = df['pH'].fillna(df['pH'].mean())
df['TDS'] = df['TDS'].fillna(df['TDS'].mean())


# Inject Noise
np.random.seed(42) # For reproducibility

# Noise parameters
ph_std = 1.5   # Standard deviation for pH noise
tds_std = 80.0 # Standard deviation for TDS noise

print(f"Injecting noise: pH_std={ph_std}, TDS_std={tds_std}")

# Add noise
df['pH'] += np.random.normal(0, ph_std, df.shape[0])
df['TDS'] += np.random.normal(0, tds_std, df.shape[0])

# Clip values to realistic ranges
df['pH'] = df['pH'].clip(0, 14) 
df['TDS'] = df['TDS'].clip(0, None) # TDS can't be negative

# Save new dataset
df.to_csv(output_file, index=False)
print(f"Noisy dataset saved to: {output_file}")
