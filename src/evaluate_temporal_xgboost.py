
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "paysim_features.csv"

RANDOM_STATE = 42
TRAIN_SAMPLE_FRACTION = 0.20

# --------------------------------------------------
# 2. Load only the columns we need
# --------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "step",
        "type",
        "amount",
        "isFraud",
        "log_amount",
        "hour_of_day",
        "day_number",
    ],
)

df = df.sort_values("step").reset_index(drop=True)

# --------------------------------------------------
# 3. Split by time, not randomly
# --------------------------------------------------
max_step = int(df["step"].max())

train_cutoff = int(max_step * 0.70)
validation_cutoff = int(max_step * 0.85)

train_df = df[df["step"] <= train_cutoff].copy()

validation_df = df[
    (df["step"] > train_cutoff)
    & (df["step"] <= validation_cutoff)
].copy()

test_df = df[df["step"] > validation_cutoff].copy()

# Sample only the training period to reduce computation.
train_df = train_df.sample(
    frac=TRAIN_SAMPLE_FRACTION,
    random_state=RANDOM_STATE,
)

print("\nTemporal split:")
for name, part in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:
    print(
        f"{name}: {len(part):,} rows | "
        f"fraud cases: {int(part['isFraud'].sum()):,} | "
        f"steps: {part['step'].min()}–{part['step'].max()}"
    )

# Stop if a period has no fraud examples.
for name, part in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
]:
    if part.empty or part["isFraud"].nunique() < 2:
        raise ValueError(
            f"{name} period needs both fraud and legitimate examples. "
            "Review the time cutoffs."
        )

# --------------------------------------------------
# 4. Prepare features consistently
# --------------------------------------------------
FEATURES = [
    "type",
    "log_amount",
    "hour_of_day",
    "day_number",
]

def make_features(part, feature_columns=None):
    X = pd.get_dummies(
        part[FEATURES],
        columns=["type"],
        dtype=int,
    )

    if feature_columns is not None:
        X = X.reindex(
            columns=feature_columns,
            fill_value=0,
        )

    return X

y_train = train_df["isFraud"].astype(int)
y_validation = validation_df["isFraud"].astype(int)
y_test = test_df["isFraud"].astype(int)

X_train = make_features(train_df)
feature_columns = X_train.columns.tolist()

X_validation = make_features(
    validation_df,
    feature_columns,
)

X_test = make_features(
    test_df,
    feature_columns,
)

# --------------------------------------------------
# 5. Train XGBoost on earlier transactions
# --------------------------------------------------
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=negative_count / positive_count,
    tree_method="hist",
    n_jobs=4,
    random_state=RANDOM_STATE,
)

print("\nTraining XGBoost...")
model.fit(X_train, y_train)
print("Training complete.")

# --------------------------------------------------
# 6. Choose a threshold using validation data only
# --------------------------------------------------
validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

thresholds = np.arange(0.10, 0.96, 0.05)

results = []

for threshold in thresholds:
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation, predictions, zero_division=0
    )
    recall = recall_score(
        y_validation, predictions, zero_division=0
    )
    f1 = f1_score(
        y_validation, predictions, zero_division=0
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alerts": int(predictions.sum()),
    })

results_df = pd.DataFrame(results)

print("\nValidation threshold comparison:")
print(results_df.round(4).to_string(index=False))

# Select the threshold with the highest validation F1.
best_row = results_df.loc[results_df["f1"].idxmax()]
best_threshold = float(best_row["threshold"])

print(f"\nSelected threshold: {best_threshold:.2f}")
print("This threshold was selected using validation data.")

# --------------------------------------------------
# 7. Final evaluation on the later test period
# --------------------------------------------------
# Do not use these test results to tune the threshold.
test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFINAL TEMPORAL TEST RESULTS")
print(f"Threshold: {best_threshold:.2f}")
print(
    "Average precision:",
    round(average_precision_score(y_test, test_probabilities), 5),
)
print(
    "Precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4),
)
print(
    "Recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4),
)
print(
    "F1 score:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4),
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

# Save the newly trained model.
model_path = ROOT / "models" / "xgboost_temporal_model.json"
model.save_model(str(model_path))

print(f"\nTemporal model saved to: {model_path}")
