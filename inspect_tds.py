import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('Dataset/water_classified_fixed.csv')
    print("Dataset loaded.")
except FileNotFoundError:
    print("Dataset not found!")
    exit()

# Rename for consistency
df.rename(columns={'TDS': 'Solids'}, inplace=True)

# Sort and print top 20
sorted_tds = df['Solids'].sort_values(ascending=False)
print("\nTop 20 Highest TDS Values:")
print(sorted_tds.head(20).to_string())

# Check for values > 50000 (likely errors or extreme outliers)
extreme_count = df[df['Solids'] > 50000].shape[0]
print(f"\nCount of TDS > 50,000: {extreme_count}")

# Visual Inspection
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['Solids'], color='orange')
plt.title('TDS Values Boxplot (Checking for Outliers)')
plt.show() # This won't show in terminal, but the stats will be useful.
