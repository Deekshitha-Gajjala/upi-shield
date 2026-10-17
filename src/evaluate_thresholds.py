
from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "paysim_features.csv"
MODEL_PATH = PROJECT_DIR / "models" / "baseline_model.joblib"

print("Loading data and model...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values("step").reset_index(drop=True)

# Reproduce the same development sample and chronological split.
df = df.sample(frac=0.20, random_state=42).sort_values("step")
df = df.reset_index(drop=True)

split_index = int(len(df) * 0.80)
test = df.iloc[split_index:]

features = ["type", "log_amount", "hour_of_day", "day_number"]
X_test = test[features]
y_test = test["isFraud"]

model = joblib.load(MODEL_PATH)
scores = model.predict_proba(X_test)[:, 1]

print("\nAverage precision:", round(
    average_precision_score(y_test, scores), 5
))

print("\nThreshold comparison")
print("-" * 78)
print(
    f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} "
    f"{'F1':>10} {'Alerts':>10} {'Missed':>10}"
)

for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(
        f"{threshold:10.2f} {precision:10.4f} {recall:10.4f} "
        f"{f1:10.4f} {tp + fp:10d} {fn:10d}"
    )

print("\nThreshold evaluation completed!")
