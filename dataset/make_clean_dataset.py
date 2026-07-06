import pandas as pd
import numpy as np

# Read converted dataset
df = pd.read_csv("dataset/converted_dataset.csv")

# Convert power column to numeric
df["power"] = pd.to_numeric(df["power"], errors="coerce")

# Create clean dataframe
clean_df = pd.DataFrame()

# Generate Date column
clean_df["Date"] = pd.date_range(
    start="2024-01-01",
    periods=len(df),
    freq="min"
).date.astype(str)

# Generate Time column
clean_df["Time"] = pd.date_range(
    start="2024-01-01",
    periods=len(df),
    freq="min"
).time.astype(str)

# Energy column from power values
clean_df["Energy Consumption (kWh)"] = df["power"].fillna(0)

# Random temperature values
clean_df["Outdoor Temperature (°C)"] = np.random.uniform(
    20,
    35,
    len(df)
)

# Appliance labels
appliances = [
    "Fan",
    "AC",
    "TV",
    "Fridge",
    "Light",
    "Washing Machine"
]

clean_df["Appliance Type"] = np.random.choice(
    appliances,
    len(df)
)

# Save final dataset
clean_df.to_csv(
    "dataset/clean_nilm_drift_dataset.csv",
    index=False
)

print("✅ clean_nilm_drift_dataset.csv created successfully!")