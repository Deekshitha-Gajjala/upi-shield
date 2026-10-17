from pathlib import Path
import pandas as pd

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found inside the data folder."
    )

dataset_path = csv_files[0]

print("Dataset:", dataset_path.name)

# Load the dataset
df = pd.read_csv(dataset_path)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

if "isFraud" in df.columns:
    print("\nFraud distribution:")
    print(df["isFraud"].value_counts())

    print("\nFraud percentage:")
    print(df["isFraud"].value_counts(normalize=True) * 100)