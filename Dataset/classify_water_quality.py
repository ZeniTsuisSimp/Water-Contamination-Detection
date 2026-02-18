import pandas as pd
import numpy as np
import os

# Define quality thresholds for 6-tier classification
# pH thresholds
PH_VERY_PURE_MIN = 7.0
PH_VERY_PURE_MAX = 7.5
PH_PURE_MIN = 6.8
PH_PURE_MAX = 8.0
PH_SAFE_MIN = 6.5
PH_SAFE_MAX = 8.5
PH_MODERATE_MIN = 6.0
PH_MODERATE_MAX = 9.0

# TDS thresholds (mg/L)
TDS_VERY_PURE_MAX = 100
TDS_PURE_MAX = 250
TDS_SAFE_MAX = 500
TDS_MODERATE_MAX = 1000
TDS_HIGH_MAX = 2000

# Conductivity thresholds (µmhos/cm)
CONDUCTIVITY_VERY_PURE_MAX = 150
CONDUCTIVITY_PURE_MAX = 400
CONDUCTIVITY_SAFE_MAX = 800
CONDUCTIVITY_MODERATE_MAX = 1500
CONDUCTIVITY_HIGH_MAX = 3000

# Nitrate thresholds (mg/L)
NITRATE_VERY_PURE_MAX = 10
NITRATE_PURE_MAX = 20
NITRATE_SAFE_MAX = 45
NITRATE_MODERATE_MAX = 100
NITRATE_HIGH_MAX = 200

# Function to classify water quality into 6 categories
def classify_water_quality(ph=None, tds=None, conductivity=None, nitrate=None):
    """
    Classify water into 6 categories: Very Pure, Pure, Safe, Moderately Contaminated, Highly Contaminated, Extremely Contaminated
    Returns: (classification, reason)
    """
    contamination_score = 0
    issues = []
    
    # Check pH
    if pd.notna(ph):
        if ph < PH_VERY_PURE_MIN or ph > PH_VERY_PURE_MAX:
            if ph < PH_PURE_MIN or ph > PH_PURE_MAX:
                if ph < PH_SAFE_MIN or ph > PH_SAFE_MAX:
                    if ph < PH_MODERATE_MIN or ph > PH_MODERATE_MAX:
                        contamination_score += 5
                        issues.append(f"pH {ph} (Critical)")
                    else:
                        contamination_score += 3
                        issues.append(f"pH {ph} (Out of range)")
                else:
                    contamination_score += 2
                    issues.append(f"pH {ph} (Acceptable range)")
            else:
                contamination_score += 1
                issues.append(f"pH {ph} (Good)")
    
    # Check TDS
    if pd.notna(tds):
        if tds > TDS_VERY_PURE_MAX:
            if tds > TDS_PURE_MAX:
                if tds > TDS_SAFE_MAX:
                    if tds > TDS_MODERATE_MAX:
                        if tds > TDS_HIGH_MAX:
                            contamination_score += 5
                            issues.append(f"TDS {tds} (Extremely High)")
                        else:
                            contamination_score += 4
                            issues.append(f"TDS {tds} (Highly contaminated)")
                    else:
                        contamination_score += 3
                        issues.append(f"TDS {tds} (Moderately contaminated)")
                else:
                    contamination_score += 2
                    issues.append(f"TDS {tds} (Slightly above safe limit)")
            else:
                contamination_score += 1
                issues.append(f"TDS {tds} (Pure)")
    
    # Check Conductivity (proxy for TDS)
    if pd.notna(conductivity) and pd.isna(tds):
        if conductivity > CONDUCTIVITY_VERY_PURE_MAX:
            if conductivity > CONDUCTIVITY_PURE_MAX:
                if conductivity > CONDUCTIVITY_SAFE_MAX:
                    if conductivity > CONDUCTIVITY_MODERATE_MAX:
                        if conductivity > CONDUCTIVITY_HIGH_MAX:
                            contamination_score += 5
                            issues.append(f"Conductivity {conductivity} (Extremely High)")
                        else:
                            contamination_score += 4
                            issues.append(f"Conductivity {conductivity} (Highly contaminated)")
                    else:
                        contamination_score += 3
                        issues.append(f"Conductivity {conductivity} (Moderately contaminated)")
                else:
                    contamination_score += 2
                    issues.append(f"Conductivity {conductivity} (Slightly elevated)")
            else:
                contamination_score += 1
                issues.append(f"Conductivity {conductivity} (Pure)")
    
    # Check Nitrates
    if pd.notna(nitrate):
        if nitrate > NITRATE_VERY_PURE_MAX:
            if nitrate > NITRATE_PURE_MAX:
                if nitrate > NITRATE_SAFE_MAX:
                    if nitrate > NITRATE_MODERATE_MAX:
                        if nitrate > NITRATE_HIGH_MAX:
                            contamination_score += 5
                            issues.append(f"Nitrate {nitrate} (Extremely High)")
                        else:
                            contamination_score += 4
                            issues.append(f"Nitrate {nitrate} (Highly contaminated)")
                    else:
                        contamination_score += 3
                        issues.append(f"Nitrate {nitrate} (Moderately above limit)")
                else:
                    contamination_score += 2
                    issues.append(f"Nitrate {nitrate} (Slightly above limit)")
            else:
                contamination_score += 1
                issues.append(f"Nitrate {nitrate} (Pure)")
    
    # Determine final classification based on score
    if contamination_score == 0:
        category = "Excellent"
    elif contamination_score <= 3:
        category = "Very Good"
    elif contamination_score <= 6:
        category = "Good"
    elif contamination_score <= 10:
        category = "Fair"
    elif contamination_score <= 14:
        category = "Poor"
    else:
        category = "Critical"
    
    reason = "; ".join(issues) if issues else "Excellent quality - All parameters within very pure range"
    
    return category, reason

# Collect all classified data
all_classified_data = []

dataset_dir = "d:\\6th sem\\Water Contamination Detection\\Dataset\\Data"

# 1. Process water_potability.csv
print("Processing water_potability.csv...")
try:
    df1 = pd.read_csv(os.path.join(dataset_dir, "water_potability.csv"))
    df1['Source'] = 'water_potability'
    
    for idx, row in df1.iterrows():
        classification, reason = classify_water_quality(
            ph=row['ph'],
            conductivity=row['Conductivity'] if 'Conductivity' in df1.columns else None
        )
        df1.at[idx, 'Classification'] = classification
        df1.at[idx, 'Reason'] = reason
    
    all_classified_data.append(df1)
    print(f"  Processed {len(df1)} records")
except Exception as e:
    print(f"  Error: {e}")

# 2. Process Packaged_drinking_water_data.csv
print("Processing Packaged_drinking_water_data.csv...")
try:
    df2 = pd.read_csv(os.path.join(dataset_dir, "Packaged_drinking_water_data.csv"))
    df2['Source'] = 'Packaged_drinking_water'
    
    # Rename columns for consistency
    if 'pH (pouvoir hydrogene)' in df2.columns:
        df2['pH'] = df2['pH (pouvoir hydrogene)']
    if 'TDS (Total Dissolved Solids)' in df2.columns:
        df2['TDS'] = df2['TDS (Total Dissolved Solids)']
    
    for idx, row in df2.iterrows():
        classification, reason = classify_water_quality(
            ph=row.get('pH'),
            tds=row.get('TDS')
        )
        df2.at[idx, 'Classification'] = classification
        df2.at[idx, 'Reason'] = reason
    
    all_classified_data.append(df2)
    print(f"  Processed {len(df2)} records")
except Exception as e:
    print(f"  Error: {e}")

# 3. Process Pipeline_drinking_water_data.csv
print("Processing Pipeline_drinking_water_data.csv...")
try:
    df3 = pd.read_csv(os.path.join(dataset_dir, "Pipeline_drinking_water_data.csv"))
    df3['Source'] = 'Pipeline_drinking_water'
    
    if 'pH (pouvoir hydrogene)' in df3.columns:
        df3['pH'] = df3['pH (pouvoir hydrogene)']
    if 'TDS (Total Dissolved Solids)' in df3.columns:
        df3['TDS'] = df3['TDS (Total Dissolved Solids)']
    
    for idx, row in df3.iterrows():
        classification, reason = classify_water_quality(
            ph=row.get('pH'),
            tds=row.get('TDS')
        )
        df3.at[idx, 'Classification'] = classification
        df3.at[idx, 'Reason'] = reason
    
    all_classified_data.append(df3)
    print(f"  Processed {len(df3)} records")
except Exception as e:
    print(f"  Error: {e}")

# 4. Process Pond_water_Data.csv
print("Processing Pond_water_Data.csv...")
try:
    df4 = pd.read_csv(os.path.join(dataset_dir, "Pond_water_Data.csv"))
    df4['Source'] = 'Pond_water'
    
    if 'pH (pouvoir hydrogene)' in df4.columns:
        df4['pH'] = df4['pH (pouvoir hydrogene)']
    if 'TDS (Total Dissolved Solids)' in df4.columns:
        df4['TDS'] = df4['TDS (Total Dissolved Solids)']
    
    for idx, row in df4.iterrows():
        classification, reason = classify_water_quality(
            ph=row.get('pH'),
            tds=row.get('TDS')
        )
        df4.at[idx, 'Classification'] = classification
        df4.at[idx, 'Reason'] = reason
    
    all_classified_data.append(df4)
    print(f"  Processed {len(df4)} records")
except Exception as e:
    print(f"  Error: {e}")

# 5. Process water_dataX.csv
print("Processing water_dataX.csv...")
try:
    df5 = pd.read_csv(os.path.join(dataset_dir, "water_dataX.csv"), encoding='latin1')
    df5['Source'] = 'water_dataX'
    
    for idx, row in df5.iterrows():
        try:
            # Extract nitrate value - handle column name variations
            nitrate = None
            for col in df5.columns:
                if 'NITRATE' in col.upper() or 'NITRITE' in col.upper():
                    val = row[col]
                    if isinstance(val, str) and val.upper() != 'NAN':
                        # Try to extract numeric value
                        try:
                            nitrate = float(val)
                        except:
                            pass
                    elif pd.notna(val) and not isinstance(val, str):
                        nitrate = float(val)
            
            # Get PH and Conductivity
            ph_val = None
            cond_val = None
            
            if 'PH' in row and pd.notna(row['PH']):
                try:
                    ph_val = float(row['PH'])
                except:
                    pass
            
            if 'CONDUCTIVITY (µmhos/cm)' in row and pd.notna(row['CONDUCTIVITY (µmhos/cm)']):
                try:
                    cond_val = float(row['CONDUCTIVITY (µmhos/cm)'])
                except:
                    pass
            
            classification, reason = classify_water_quality(
                ph=ph_val,
                conductivity=cond_val,
                nitrate=nitrate
            )
            df5.at[idx, 'Classification'] = classification
            df5.at[idx, 'Reason'] = reason
        except Exception as e:
            df5.at[idx, 'Classification'] = 'UNKNOWN'
            df5.at[idx, 'Reason'] = f'Error processing: {str(e)}'
    
    all_classified_data.append(df5)
    print(f"  Processed {len(df5)} records")
except Exception as e:
    print(f"  Error: {e}")

# 6. Process ground_water_quality_in_tripura-2014.csv
print("Processing ground_water_quality_in_tripura-2014.csv...")
try:
    df6 = pd.read_csv(os.path.join(dataset_dir, "ground_water_quality_in_tripura-2014.csv"), encoding='latin1')
    df6['Source'] = 'groundwater_tripura'
    
    for idx, row in df6.iterrows():
        try:
            # Extract mean pH value
            ph_mean = None
            for col in df6.columns:
                if 'pH : Mean' in col:
                    val = row[col]
                    if pd.notna(val):
                        try:
                            ph_mean = float(val)
                        except:
                            pass
            
            # Extract mean conductivity
            conductivity_mean = None
            for col in df6.columns:
                if 'CONDUCTIVITY' in col and 'Mean' in col:
                    val = row[col]
                    if pd.notna(val):
                        try:
                            conductivity_mean = float(val)
                        except:
                            pass
            
            classification, reason = classify_water_quality(
                ph=ph_mean,
                conductivity=conductivity_mean
            )
            df6.at[idx, 'Classification'] = classification
            df6.at[idx, 'Reason'] = reason
        except Exception as e:
            df6.at[idx, 'Classification'] = 'UNKNOWN'
            df6.at[idx, 'Reason'] = f'Error processing: {str(e)}'
    
    all_classified_data.append(df6)
    print(f"  Processed {len(df6)} records")
except Exception as e:
    print(f"  Error: {e}")

# Combine all data
print("\nCombining all datasets...")
combined_df = pd.concat(all_classified_data, ignore_index=True, sort=False)

# Save to new CSV file in parent Dataset directory
output_file = "d:\\6th sem\\Water Contamination Detection\\Dataset\\water_quality_classified.csv"
combined_df.to_csv(output_file, index=False)

print(f"\n✓ Classification complete!")
print(f"Total records processed: {len(combined_df)}")
print(f"\nCLASSIFICATION DISTRIBUTION:")
print("-" * 60)
for category in ["Very Pure", "Pure", "Safe", "Moderately Contaminated", "Highly Contaminated", "Extremely Contaminated"]:
    count = len(combined_df[combined_df['Classification'] == category])
    percentage = (count / len(combined_df)) * 100 if len(combined_df) > 0 else 0
    print(f"{category:30} : {count:5} records ({percentage:6.2f}%)")
print(f"\nClassified data saved to: {output_file}")

# Display summary statistics
print("\n" + "="*80)
print("SUMMARY BY SOURCE AND CATEGORY:")
print("="*80)
for source in sorted(combined_df['Source'].unique()):
    source_data = combined_df[combined_df['Source'] == source]
    print(f"\n{source}:")
    for category in ["Very Pure", "Pure", "Safe", "Moderately Contaminated", "Highly Contaminated", "Extremely Contaminated"]:
        count = len(source_data[source_data['Classification'] == category])
        if count > 0:
            percentage = (count / len(source_data)) * 100
            print(f"  {category:30} : {count:5} ({percentage:6.2f}%)")
