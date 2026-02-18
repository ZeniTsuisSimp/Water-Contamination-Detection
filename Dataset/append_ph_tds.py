import pandas as pd
import numpy as np


def fix_ph_value(ph):
    if pd.isna(ph) or ph < 0:
        return np.nan
    if ph <= 14:
        return ph
    ph_str = str(int(ph)) if ph == int(ph) else str(ph)
    if len(ph_str) > 1:
        fixed = float(ph_str[0] + "." + ph_str[1:])
        if 0 <= fixed <= 14:
            return fixed
    return np.nan


# Load target (only pH/TDS)
target = pd.read_csv("water_data_classified.csv")
if not {"pH", "TDS"}.issubset(target.columns):
    raise ValueError("water_data_classified.csv must have pH and TDS columns")

target = target[["pH", "TDS"]].copy()

# Load source data to append
src = pd.read_csv("water_quality_classified.csv")

# Build pH column by coalescing known fields
ph_cols = ["pH", "ph", "PH", "pH (pouvoir hydrogene)"]
ph_series = None
for col in ph_cols:
    if col in src.columns:
        col_vals = pd.to_numeric(src[col], errors="coerce")
        ph_series = col_vals if ph_series is None else ph_series.combine_first(col_vals)

if ph_series is None:
    ph_series = pd.Series([np.nan] * len(src))

# Build TDS column by coalescing known fields
if "TDS" in src.columns:
    tds_series = pd.to_numeric(src["TDS"], errors="coerce")
elif "TDS (Total Dissolved Solids)" in src.columns:
    tds_series = pd.to_numeric(src["TDS (Total Dissolved Solids)"], errors="coerce")
else:
    tds_series = pd.Series([np.nan] * len(src))

if "Solids" in src.columns:
    tds_series = tds_series.combine_first(pd.to_numeric(src["Solids"], errors="coerce"))

if "Conductivity" in src.columns:
    tds_series = tds_series.combine_first(pd.to_numeric(src["Conductivity"], errors="coerce") * 0.65)

if "CONDUCTIVITY (µmhos/cm)" in src.columns:
    tds_series = tds_series.combine_first(
        pd.to_numeric(src["CONDUCTIVITY (µmhos/cm)"], errors="coerce") * 0.65
    )

appended = pd.DataFrame({"pH": ph_series, "TDS": tds_series})
appended["pH"] = appended["pH"].apply(fix_ph_value)

# Keep rows with at least one value
appended = appended[(appended["pH"].notna()) | (appended["TDS"].notna())]

# Combine and save
combined = pd.concat([target, appended], ignore_index=True)
combined.to_csv("water_data_classified.csv", index=False)

print(f"Appended rows: {len(appended)}")
print(f"Total rows now: {len(combined)}")
