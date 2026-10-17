
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "processed" / "paysim_features.csv"
MODEL_PATH = PROJECT_DIR / "models" / "xgboost_model.json"

print("Loading dataset and XGBoost model...")
df = pd.read_csv(DATA_PATH)

# Reproduce the same development sample and test split.
df = df.sample(frac=0.20, random_state=42)
df = df.sort_values("step").reset_index(drop=True)

features = ["type", "log_amount", "hour_of_day", "day_number"]
X = pd.get_dummies(df[features], columns=["type"], dtype=int)
y = df["isFraud"]

split_index = int(len(df) * 0.80)
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

model = XGBClassifier()
model.load_model(MODEL_PATH)

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

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
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
