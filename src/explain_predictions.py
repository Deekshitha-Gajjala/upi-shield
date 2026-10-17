
from pathlib import Path

import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "paysim_features.csv"
MODEL_PATH = ROOT / "models" / "xgboost_temporal_model.json"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load model and data
# --------------------------------------------------
print("Loading model...")

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

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

# --------------------------------------------------
# 3. Encode features exactly as the model expects
# --------------------------------------------------
FEATURES = [
    "type",
    "log_amount",
    "hour_of_day",
    "day_number",
]

X = pd.get_dummies(
    df[FEATURES],
    columns=["type"],
    dtype=int,
)

expected_features = model.get_booster().feature_names
X = X.reindex(columns=expected_features, fill_value=0)

# --------------------------------------------------
# 4. Select one fraud and one legitimate transaction
# --------------------------------------------------
fraud_rows = df.index[df["isFraud"] == 1]
legitimate_rows = df.index[df["isFraud"] == 0]

if len(fraud_rows) == 0 or len(legitimate_rows) == 0:
    raise ValueError("Both fraud and legitimate examples are required.")

fraud_index = int(fraud_rows[0])
legitimate_index = int(legitimate_rows[0])

examples = [
    ("Fraud example", fraud_index),
    ("Legitimate example", legitimate_index),
]

# --------------------------------------------------
# 5. Create SHAP explainer
# --------------------------------------------------
print("Preparing SHAP explainer...")
explainer = shap.TreeExplainer(model)

# --------------------------------------------------
# 6. Explain both transactions
# --------------------------------------------------
for label, index in examples:
    transaction = X.iloc[[index]]
    actual_label = int(df.loc[index, "isFraud"])

    probability = float(model.predict_proba(transaction)[0, 1])

    print(f"\n{label}")
    print(f"Transaction type: {df.loc[index, 'type']}")
    print(f"Amount: {df.loc[index, 'amount']:.2f}")
    print(f"Actual dataset label: {actual_label}")
    print(f"Predicted fraud probability: {probability:.4f}")

    shap_values = explainer(transaction)

    safe_label = label.lower().replace(" ", "_")

    # Save the individual SHAP waterfall plot
    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False,
    )

    plt.title(f"UPI Shield - {label}")
    plt.tight_layout()

    output_path = REPORTS_DIR / f"shap_{safe_label}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Explanation saved to: {output_path}")

print("\nBoth transaction explanations are complete.")
