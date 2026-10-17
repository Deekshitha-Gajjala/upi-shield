
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_PATH = DATA_DIR / "processed" / "paysim_features.csv"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_PATH = OUTPUT_DIR / "paysim_behavior_features.csv"

print("Loading transaction data...")

df = pd.read_csv(
    INPUT_PATH,
    usecols=[
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud",
        "log_amount",
        "hour_of_day",
        "day_number",
    ],
)

# Preserve transaction order.
# Within the same step, original file order breaks ties.
df["_row_order"] = range(len(df))
df = df.sort_values(["step", "_row_order"]).reset_index(drop=True)

print("Building sender behavior features...")

sender = df.groupby("nameOrig", sort=False)

# Number of earlier transactions made by this sender.
df["sender_txn_count_so_far"] = sender.cumcount()

# Cumulative amount from earlier transactions only.
df["_sender_amount_sum"] = sender["amount"].cumsum()
df["_sender_amount_sum_before"] = (
    df.groupby("nameOrig", sort=False)["_sender_amount_sum"].shift(1)
)

# Average amount from earlier transactions only.
df["sender_avg_amount_before"] = (
    df["_sender_amount_sum_before"]
    / df["sender_txn_count_so_far"].replace(0, float("nan"))
).fillna(0)

df = df.drop(
    columns=["_row_order", "_sender_amount_sum", "_sender_amount_sum_before"]
)

print("\nNew features:")
print(df[[
    "nameOrig",
    "step",
    "amount",
    "sender_txn_count_so_far",
    "sender_avg_amount_before",
]].head(10))

print("\nMissing values in new features:")
print(df[[
    "sender_txn_count_so_far",
    "sender_avg_amount_before",
]].isna().sum())

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved to:", OUTPUT_PATH)
print("Behavior feature preparation completed!")
