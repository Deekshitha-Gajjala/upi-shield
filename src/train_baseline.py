
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "paysim_features.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading prepared dataset...")
df = pd.read_csv(DATA_PATH)

# Keep a chronological split: earlier transactions for training,
# later transactions for testing.
df = df.sort_values("step").reset_index(drop=True)

# Use a 20% sample for the first development run.
# Sampling is reproducible and retains the natural class distribution.
df = df.sample(frac=0.20, random_state=42).sort_values("step")
df = df.reset_index(drop=True)

print("Development sample:", df.shape)
print("Fraud cases:", int(df["isFraud"].sum()))

# Features only: do not use account IDs or balance fields.
feature_columns = [
    "type",
    "log_amount",
    "hour_of_day",
    "day_number",
]

X = df[feature_columns]
y = df["isFraud"]

# Chronological 80/20 split within the sampled data
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

if y_train.nunique() < 2 or y_test.sum() == 0:
    raise ValueError(
        "Training or test split lacks fraud examples. "
        "Inspect the time split before training."
    )

# Encode transaction type and scale numeric features.
preprocessor = ColumnTransformer(
    transformers=[
        ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        (
            "numeric",
            StandardScaler(),
            ["log_amount", "hour_of_day", "day_number"],
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=300,
                solver="lbfgs",
            ),
        ),
    ]
)

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

print("Generating predictions...")
probabilities = model.predict_proba(X_test)[:, 1]

# Initial threshold only; we'll tune it later.
predictions = (probabilities >= 0.5).astype(int)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print(
    "Average precision:",
    round(average_precision_score(y_test, probabilities), 5),
)

model_path = MODEL_DIR / "baseline_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("Baseline training completed!")
