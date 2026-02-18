import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
try:
    df = pd.read_csv('Dataset/water_classified_fixed.csv')
    df.rename(columns={'pH': 'ph', 'TDS': 'Solids', 'Classification': 'Potability'}, inplace=True)

    # Encoding
    target_mapping = {'Safe': 1, 'Unsafe': 0}
    df['Potability'] = df['Potability'].str.strip().map(target_mapping)
    df = df.dropna(subset=['Potability'])
    df['Potability'] = df['Potability'].astype(int)

    # Apply the same cleaning (TDS <= 10000)
    df_clean = df[df['Solids'] <= 10000].copy()

    # Write results to file
    with open('investigation_results.txt', 'w') as f:
        f.write(f"Original Count: {len(df)}\n")
        f.write(f"Cleaned Count: {len(df_clean)}\n\n")
        
        f.write("--- Class Distribution (Cleaned) ---\n")
        f.write(str(df_clean['Potability'].value_counts(normalize=True)) + "\n")
        f.write("Counts:\n")
        f.write(str(df_clean['Potability'].value_counts()) + "\n\n")

        f.write("--- Correlation Matrix ---\n")
        corr = df_clean[['ph', 'Solids', 'Potability']].corr()
        f.write(str(corr) + "\n")

    # 3. Visual Check for Separation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='ph', y='Solids', hue='Potability', data=df_clean, palette='coolwarm', alpha=0.6)
    plt.title('pH vs TDS (Cleaned Data) - Check for Separation')
    plt.savefig('separation_check.png')
    print("Investigation complete. Results saved to 'investigation_results.txt'.")

except Exception as e:
    print(f"Error: {e}")
