
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "paysim_features.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Reproducible 20% sample, then chronological split
df = df.sample(frac=0.20, random_state=42)
df = df.sort_values("step").reset_index(drop=True)

features = ["type", "log_amount", "hour_of_day", "day_number"]
X = df[features]
y = df["isFraud"]

# Encode transaction type
X = pd.get_dummies(X, columns=["type"], dtype=int)

split_index = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

if y_train.nunique() < 2 or y_test.sum() == 0:
    raise ValueError("The split does not contain enough fraud examples.")

# Give fraud cases more weight because they are rare.
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / max(positive_count, 1)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    n_jobs=4,
    random_state=42,
)

print("\nTraining XGBoost...")
model.fit(X_train, y_train)

print("Predicting...")
probabilities = model.predict_proba(X_test)[:, 1]
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

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print(
    "\nAverage precision:",
    round(average_precision_score(y_test, probabilities), 5),
)

model_path = MODEL_DIR / "xgboost_model.json"
model.save_model(model_path)

print("\nModel saved to:", model_path)
print("XGBoost training completed!")
