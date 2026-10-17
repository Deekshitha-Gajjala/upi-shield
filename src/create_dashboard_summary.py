
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "paysim_features.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "dashboard_summary.csv"

print("Reading transaction type and fraud labels...")

df = pd.read_csv(
    DATA_PATH,
    usecols=["type", "isFraud"],
)

summary = (
    df.groupby("type")
    .agg(
        transaction_count=("isFraud", "size"),
        fraud_count=("isFraud", "sum"),
    )
    .reset_index()
)

summary["legitimate_count"] = (
    summary["transaction_count"] - summary["fraud_count"]
)

summary["fraud_rate_pct"] = (
    summary["fraud_count"] / summary["transaction_count"] * 100
)

summary.to_csv(OUTPUT_PATH, index=False)

print("\nDashboard summary:")
print(summary.to_string(index=False))
print(f"\nSaved summary to: {OUTPUT_PATH}")
