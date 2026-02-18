import pandas as pd
import os

# Simple thresholds
PH_MIN = 6.5
PH_MAX = 8.5
TDS_MAX = 500  # mg/L

def fix_ph_value(ph):
    """Fix invalid pH values by adding decimal point"""
    if pd.isna(ph) or ph < 0:
        return None
    
    # Valid pH range is 0-14
    if ph <= 14:
        return ph
    
    # If pH is too high, add decimal point
    ph_str = str(int(ph)) if ph == int(ph) else str(ph)
    
    # Try different decimal positions to get value between 0-14
    # For 2500 -> 2.500, for 67115 -> 6.7115, etc
    if len(ph_str) > 2:
        # Move decimal point after first digit
        fixed = float(ph_str[0] + '.' + ph_str[1:])
        if 0 <= fixed <= 14:
            return fixed
    
    return None

def classify_simple(ph, tds):
    """Classify water as Safe or Unsafe based on pH and TDS"""
    if pd.isna(ph) and pd.isna(tds):
        return "Unknown"
    
    if pd.notna(ph) and (ph < PH_MIN or ph > PH_MAX):
        return "Unsafe"
    
    if pd.notna(tds) and tds > TDS_MAX:
        return "Unsafe"
    
    if pd.notna(ph) or pd.notna(tds):
        return "Safe"
    
    return "Unknown"

dataset_dir = "d:\\6th sem\\Water Contamination Detection\\Dataset\\Data"
all_data = []

# Process water_potability.csv
print("Processing water_potability.csv...")
try:
    df = pd.read_csv(os.path.join(dataset_dir, "water_potability.csv"))
    df['Source'] = 'water_potability'
    df['pH'] = pd.to_numeric(df['ph'], errors='coerce')
    if 'Conductivity' in df.columns:
        # Convert conductivity to approximate TDS (TDS ≈ Conductivity × 0.65)
        df['TDS'] = pd.to_numeric(df['Conductivity'], errors='coerce') * 0.65
    else:
        df['TDS'] = None
    # Fix invalid pH values by adding decimal point
    df['pH'] = df['pH'].apply(fix_ph_value)
    df['Classification'] = df.apply(lambda row: classify_simple(row.get('pH'), row.get('TDS')), axis=1)
    all_data.append(df)
    print(f"  Added {len(df)} records")
except Exception as e:
    print(f"  Error: {e}")

# Process Packaged_drinking_water_data.csv
print("Processing Packaged_drinking_water_data.csv...")
try:
    df = pd.read_csv(os.path.join(dataset_dir, "Packaged_drinking_water_data.csv"))
    df['Source'] = 'Packaged_drinking_water'
    if 'pH (pouvoir hydrogene)' in df.columns:
        df['pH'] = pd.to_numeric(df['pH (pouvoir hydrogene)'], errors='coerce')
    if 'TDS (Total Dissolved Solids)' in df.columns:
        df['TDS'] = pd.to_numeric(df['TDS (Total Dissolved Solids)'], errors='coerce')
    else:
        df['TDS'] = None
    # Fix invalid pH values by adding decimal point
    df['pH'] = df['pH'].apply(fix_ph_value)
    df['Classification'] = df.apply(lambda row: classify_simple(row.get('pH'), row.get('TDS')), axis=1)
    all_data.append(df)
    print(f"  Added {len(df)} records")
except Exception as e:
    print(f"  Error: {e}")

# Process Pipeline_drinking_water_data.csv
print("Processing Pipeline_drinking_water_data.csv...")
try:
    df = pd.read_csv(os.path.join(dataset_dir, "Pipeline_drinking_water_data.csv"))
    df['Source'] = 'Pipeline_drinking_water'
    if 'pH (pouvoir hydrogene)' in df.columns:
        df['pH'] = pd.to_numeric(df['pH (pouvoir hydrogene)'], errors='coerce')
    if 'TDS (Total Dissolved Solids)' in df.columns:
        df['TDS'] = pd.to_numeric(df['TDS (Total Dissolved Solids)'], errors='coerce')
    else:
        df['TDS'] = None
    # Fix invalid pH values by adding decimal point
    df['pH'] = df['pH'].apply(fix_ph_value)
    df['Classification'] = df.apply(lambda row: classify_simple(row.get('pH'), row.get('TDS')), axis=1)
    all_data.append(df)
    print(f"  Added {len(df)} records")
except Exception as e:
    print(f"  Error: {e}")

# Process Pond_water_Data.csv
print("Processing Pond_water_Data.csv...")
try:
    df = pd.read_csv(os.path.join(dataset_dir, "Pond_water_Data.csv"))
    df['Source'] = 'Pond_water'
    if 'pH (pouvoir hydrogene)' in df.columns:
        df['pH'] = pd.to_numeric(df['pH (pouvoir hydrogene)'], errors='coerce')
    if 'TDS (Total Dissolved Solids)' in df.columns:
        df['TDS'] = pd.to_numeric(df['TDS (Total Dissolved Solids)'], errors='coerce')
    else:
        df['TDS'] = None
    # Fix invalid pH values by adding decimal point
    df['pH'] = df['pH'].apply(fix_ph_value)
    df['Classification'] = df.apply(lambda row: classify_simple(row.get('pH'), row.get('TDS')), axis=1)
    all_data.append(df)
    print(f"  Added {len(df)} records")
except Exception as e:
    print(f"  Error: {e}")

# Process water_dataX.csv
print("Processing water_dataX.csv...")
try:
    df = pd.read_csv(os.path.join(dataset_dir, "water_dataX.csv"), encoding='latin1')
    df['Source'] = 'water_dataX'
    if 'PH' in df.columns:
        df['pH'] = pd.to_numeric(df['PH'], errors='coerce')
    if 'CONDUCTIVITY (µmhos/cm)' in df.columns:
        df['TDS'] = pd.to_numeric(df['CONDUCTIVITY (µmhos/cm)'], errors='coerce') * 0.65
    else:
        df['TDS'] = None
    # Fix invalid pH values by adding decimal point
    df['pH'] = df['pH'].apply(fix_ph_value)
    df['Classification'] = df.apply(lambda row: classify_simple(row.get('pH'), row.get('TDS')), axis=1)
    all_data.append(df)
    print(f"  Added {len(df)} records")
except Exception as e:
    print(f"  Error: {e}")

# Process ground_water_quality_in_tripura-2014.csv
print("Processing ground_water_quality_in_tripura-2014.csv...")
try:
    df = pd.read_csv(os.path.join(dataset_dir, "ground_water_quality_in_tripura-2014.csv"), encoding='latin1')
    df['Source'] = 'groundwater_tripura'
    ph_col = [col for col in df.columns if 'pH' in col and 'Mean' in col]
    if ph_col:
        df['pH'] = pd.to_numeric(df[ph_col[0]], errors='coerce')
    cond_col = [col for col in df.columns if 'CONDUCTIVITY' in col and 'Mean' in col]
    if cond_col:
        df['TDS'] = pd.to_numeric(df[cond_col[0]], errors='coerce') * 0.65
    else:
        df['TDS'] = None
    # Fix invalid pH values by adding decimal point
    df['pH'] = df['pH'].apply(fix_ph_value)
    df['Classification'] = df.apply(lambda row: classify_simple(row.get('pH'), row.get('TDS')), axis=1)
    all_data.append(df)
    print(f"  Added {len(df)} records")
except Exception as e:
    print(f"  Error: {e}")

# Combine all data
print("\nCombining datasets...")
combined_df = pd.concat(all_data, ignore_index=True, sort=False)

# Keep only relevant columns
output_df = combined_df[['Source', 'pH', 'TDS', 'Classification']].copy()

# Save to CSV
output_file = "d:\\6th sem\\Water Contamination Detection\\Dataset\\water_data_classified.csv"
output_df.to_csv(output_file, index=False)

print(f"\n✓ Complete!")
print(f"Total records: {len(output_df)}")
print(f"Safe: {len(output_df[output_df['Classification'] == 'Safe'])}")
print(f"Unsafe: {len(output_df[output_df['Classification'] == 'Unsafe'])}")
print(f"Unknown: {len(output_df[output_df['Classification'] == 'Unknown'])}")
print(f"\nFile saved to: {output_file}")
