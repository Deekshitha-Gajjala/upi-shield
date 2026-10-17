from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the original dataset
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("Original PaySim CSV not found.")

input_path = csv_files[0]
output_path = OUTPUT_DIR / "paysim_features.csv"

print("Loading:", input_path.name)

# Read only the columns needed for initial features
columns = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
]

df = pd.read_csv(input_path, usecols=columns)

print("Original shape:", df.shape)

# Feature 1: Log-transformed transaction amount
df["log_amount"] = np.log1p(df["amount"])

# Feature 2: Hour within the simulated day
df["hour_of_day"] = (df["step"] - 1) % 24

# Feature 3: Day within the simulated period
df["day_number"] = ((df["step"] - 1) // 24) + 1

# Keep account IDs for later network investigation.
# They will NOT be used as raw ML features.
df.to_csv(output_path, index=False)

print("\nEngineered features created:")
print(df.columns.tolist())

print("\nSample rows:")
print(df.head())

print("\nFraud distribution:")
print(df["isFraud"].value_counts())

print("\nSaved engineered dataset to:")
print(output_path)

print("\nFeature preparation completed!")