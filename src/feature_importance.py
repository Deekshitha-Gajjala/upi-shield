
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgboost_temporal_model.json"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the temporally trained model
model = XGBClassifier()
model.load_model(str(MODEL_PATH))

# Get importance scores
importance = pd.DataFrame({
    "feature": model.feature_names_in_,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(importance.to_string(index=False))

# Save results
importance.to_csv(
    REPORTS_DIR / "feature_importance.csv",
    index=False,
)

# Plot results
plt.figure(figsize=(9, 5))
plt.barh(importance["feature"], importance["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("UPI Shield - XGBoost Feature Importance")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "feature_importance.png",
    dpi=150,
)
plt.show()

print("\nSaved feature importance to the reports folder.")
