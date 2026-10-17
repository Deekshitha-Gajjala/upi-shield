from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Locate and load the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

csv_file = next(DATA_DIR.glob("*.csv"))
df = pd.read_csv(csv_file)

print("Dataset loaded:", df.shape)

# 1. Fraud by transaction type
print("\nFraud counts by transaction type:")

fraud_counts = pd.crosstab(
    df["type"],
    df["isFraud"]
)

print(fraud_counts)

# 2. Fraud rate by transaction type
print("\nFraud rate by transaction type (%):")

fraud_rate = (
    df.groupby("type")["isFraud"].mean() * 100
).sort_values(ascending=False)

print(fraud_rate.round(4))

# 3. Compare transaction amounts
print("\nTransaction amounts by fraud label:")

print(
    df.groupby("isFraud")["amount"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)

# 4. Visualize transaction types
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="type",
    hue="isFraud"
)

plt.title("Transaction Types: Fraud vs Legitimate")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 5. Visualize fraud rates by type
plt.figure(figsize=(9, 5))

fraud_rate.sort_values().plot(
    kind="barh"
)

plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Fraud Rate (%)")
plt.ylabel("Transaction Type")
plt.tight_layout()
plt.show()