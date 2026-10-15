import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

input_file = "image_outputs.csv"
output_file = "variation_by_name.csv"

# Define input file delimiter
delimiter = ";"

metrics = ["cx", "cy", "sx", "sy"]

# --------------------------------------------------
# Read CSV
# --------------------------------------------------

df = pd.read_csv(input_file, sep=delimiter)

# Normalize column names to handle whitespace.
df.columns = df.columns.str.strip()

required_columns = ["File name", "Capture number", *metrics]
missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Convert measurements to numeric values.
for col in metrics:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------------
# Calculate variation across captures for each name
# --------------------------------------------------

grouped = df.groupby("File name", sort=False)

summary = grouped[metrics].agg(
    ["count", "mean", "std", "min", "max"]
)

# Flatten multi-level column names.
summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

# Calculate range for each measurement.
for metric in metrics:
    summary[f"{metric}_range"] = (
        summary[f"{metric}_max"] - summary[f"{metric}_min"]
    )

# Calculate relative variation as a percentage of the mean.
# Use absolute mean to avoid negative percentages.
for metric in metrics:
    mean = summary[f"{metric}_mean"].abs()
    summary[f"{metric}_variation_pct"] = (
        summary[f"{metric}_std"] / mean * 100
    ).where(mean != 0)

# Save results.
summary.reset_index().to_csv(output_file, sep=";", index=False)

print(f"Processed {len(df)} rows.")
print(f"Found {len(summary)} unique names.")
print(f"Results saved to: {output_file}")

# Maximum and minimum allowed variation percentage
variation_max_threshold = 30
variation_min_threshold = 0.1

# Variation percentage columns
variation_columns = [
    f"{metric}_variation_pct"
    for metric in ["cx", "cy", "sx", "sy"]
]

# Keep names where any metric is below the threshold.
mask = (
        (summary[variation_columns] >= variation_min_threshold)
        & (summary[variation_columns] <= variation_max_threshold)
).all(axis=1)

filtered = summary.loc[mask].reset_index()

# Save filtered results
filtered.to_csv(
    "filtered_variation_by_name.csv",
    sep=";",
    index=False,
)
