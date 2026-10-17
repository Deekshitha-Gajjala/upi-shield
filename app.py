
from pathlib import Path
from datetime import datetime, timezone
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "xgboost_temporal_model.json"
SUMMARY_PATH = ROOT / "data" / "processed" / "dashboard_summary.csv"

DEMO_THRESHOLD = 0.50

st.set_page_config(
    page_title="UPI Shield | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ UPI Shield")
st.subheader("AI-Powered Transaction Fraud Analysis")

st.warning(
    "Research prototype using the synthetic PaySim mobile-money dataset. "
    "This is not a live UPI monitoring system. Predictions are experimental "
    "and must not be treated as confirmation of fraud."
)


# ============================================================
# 2. LOAD MODEL AND DASHBOARD SUMMARY
# ============================================================

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_overview():
    return pd.read_csv(SUMMARY_PATH)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not SUMMARY_PATH.exists():
    st.error(f"Summary file not found: {SUMMARY_PATH}")
    st.info("Run: python src/create_dashboard_summary.py")
    st.stop()

try:
    model = load_model()
    overview_df = load_overview()
except Exception as exc:
    st.error(f"Unable to load the model or summary: {exc}")
    st.stop()


required_columns = {
    "type",
    "transaction_count",
    "fraud_count",
    "legitimate_count",
    "fraud_rate_pct",
}

if not required_columns.issubset(overview_df.columns):
    st.error(
        "dashboard_summary.csv is missing required columns. "
        "Run src/create_dashboard_summary.py again."
    )
    st.stop()

overview_df["type"] = overview_df["type"].astype(str)

total_transactions = int(overview_df["transaction_count"].sum())
fraud_count = int(overview_df["fraud_count"].sum())
legitimate_count = int(overview_df["legitimate_count"].sum())

fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions
    else 0.0
)


# ============================================================
# 3. DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Fraudulent Transactions", f"{fraud_count:,}")
col3.metric("Legitimate Transactions", f"{legitimate_count:,}")
col4.metric("Dataset Fraud Rate", f"{fraud_rate:.3f}%")

st.caption(
    "These metrics describe the labels in the dataset. They do not "
    "represent live payment activity or model accuracy."
)


# ============================================================
# 4. MODEL FEATURE VALIDATION
# ============================================================

feature_names = model.get_booster().feature_names

if not feature_names:
    st.error(
        "The model does not contain feature names. "
        "Check the feature order used during training."
    )
    st.stop()

# These are the features used by the temporal model.
# Fail safely if the saved model has a different feature schema.
expected_features = {
    "log_amount",
    "hour_of_day",
    "day_number",
    "type_PAYMENT",
    "type_CASH_IN",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_DEBIT",
}

if set(feature_names) != expected_features:
    st.error(
        "The saved model's feature schema differs from the expected "
        "temporal model schema. Prediction is stopped to avoid silently "
        "feeding incorrect features."
    )
    st.write("Features found in model:", feature_names)
    st.write("Expected features:", sorted(expected_features))
    st.stop()


# ============================================================
# 5. TRANSACTION INPUT
# ============================================================

st.header("🔎 Analyze a Transaction")

st.write(
    "Enter the transaction details below to calculate an experimental "
    "fraud-risk score."
)

with st.form("transaction_form"):
    left_col, right_col = st.columns(2)

    with left_col:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Enter a non-negative amount.",
        )

    with right_col:
        hour_of_day = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12,
        )

        day_number = st.number_input(
            "Day Number",
            min_value=1,
            max_value=744,
            value=17,
            help=(
                "Use the same day-number convention as the PaySim "
                "training dataset."
            ),
        )

    analyze_button = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
        width="stretch",
    )


# ============================================================
# 6. PREPARE MODEL INPUT
# ============================================================

def prepare_transaction(
    transaction_type,
    amount,
    hour_of_day,
    day_number,
    feature_names,
):
    """
    Build the input using the temporal model's training features.

    The transaction type is one-hot encoded. The numeric features
    match the temporal model's expected feature schema.
    """

    row = {
        "log_amount": float(np.log1p(amount)),
        "hour_of_day": int(hour_of_day),
        "day_number": int(day_number),
        "type_PAYMENT": int(transaction_type == "PAYMENT"),
        "type_CASH_IN": int(transaction_type == "CASH_IN"),
        "type_TRANSFER": int(transaction_type == "TRANSFER"),
        "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
        "type_DEBIT": int(transaction_type == "DEBIT"),
    }

    # Preserve the exact feature order stored in the model.
    return pd.DataFrame(
        [[row[name] for name in feature_names]],
        columns=feature_names,
    )


# ============================================================
# 7. PREDICTION
# ============================================================

if analyze_button:
    try:
        transaction_df = prepare_transaction(
            transaction_type=transaction_type,
            amount=amount,
            hour_of_day=hour_of_day,
            day_number=day_number,
            feature_names=feature_names,
        )

        probabilities = model.predict_proba(transaction_df)
        fraud_probability = float(probabilities[0, 1])

        flagged = fraud_probability >= DEMO_THRESHOLD

        st.divider()
        st.header("🧾 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Estimated Fraud Score",
            f"{fraud_probability:.2%}",
        )

        result_col2.metric(
            "Demonstration Threshold",
            f"{DEMO_THRESHOLD:.0%}",
        )

        st.progress(
            min(max(fraud_probability, 0.0), 1.0)
        )

        if flagged:
            st.error(
                "Flagged for review by the demonstration threshold. "
                "This is not proof that the transaction is fraudulent."
            )
        else:
            st.success(
                "Not flagged by the demonstration threshold. "
                "This does not guarantee that the transaction is legitimate."
            )

        st.caption(
            "The 50% threshold is for demonstration only. It is not the "
            "validation-optimized threshold from model evaluation. "
            "The score is not guaranteed to be a calibrated real-world probability."
        )

        # ====================================================
        # 8. SHAP EXPLANATION
        # ====================================================

        st.subheader("🧠 Explainable AI: SHAP")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transaction_df)

        shap_array = np.asarray(shap_values)

        if shap_array.ndim == 3:
            shap_row = shap_array[0, :, -1]
        elif shap_array.ndim == 2:
            shap_row = shap_array[0]
        else:
            shap_row = shap_array.reshape(-1)

        expected_value = np.asarray(
            explainer.expected_value
        ).reshape(-1)

        base_value = float(expected_value[-1])

        explanation = shap.Explanation(
            values=shap_row,
            base_values=base_value,
            data=transaction_df.iloc[0].to_numpy(),
            feature_names=transaction_df.columns.tolist(),
        )

        shap_table = pd.DataFrame(
            {
                "Feature": transaction_df.columns,
                "Input Value": transaction_df.iloc[0].to_numpy(),
                "SHAP Value": shap_row,
            }
        )

        shap_table["Absolute Impact"] = (
            shap_table["SHAP Value"].abs()
        )

        shap_table = shap_table.sort_values(
            "Absolute Impact",
            ascending=False,
        )

        st.write(
            "Positive SHAP values push the model output toward the fraud "
            "class; negative values push it away. SHAP values are on the "
            "model output scale and are not percentage-point changes."
        )

        st.dataframe(
            shap_table[
                ["Feature", "Input Value", "SHAP Value"]
            ].round(5),
            width="stretch",
            hide_index=True,
        )

        st.subheader("SHAP Waterfall Plot")

        plt.close("all")
        shap.plots.waterfall(
            explanation,
            max_display=12,
            show=False,
        )

        fig = plt.gcf()
        st.pyplot(fig, width="stretch")
        plt.close("all")

        # ====================================================
        # 9. DOWNLOADABLE CSV REPORT
        # ====================================================

        st.subheader("📥 Download Analysis Report")

        report_df = shap_table.copy()
        report_df["transaction_type"] = transaction_type
        report_df["transaction_amount"] = amount
        report_df["hour_of_day"] = hour_of_day
        report_df["day_number"] = int(day_number)
        report_df["fraud_score"] = fraud_probability
        report_df["demonstration_threshold"] = DEMO_THRESHOLD
        report_df["flagged_for_review"] = flagged
        report_df["model"] = "XGBoost temporal model"
        report_df["generated_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        csv_buffer = io.StringIO()
        report_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="Download CSV Report",
            data=csv_buffer.getvalue(),
            file_name="upi_shield_analysis_report.csv",
            mime="text/csv",
            width="stretch",
        )

    except Exception as exc:
        st.error(f"Transaction analysis failed: {exc}")
        st.exception(exc)


# ============================================================
# 10. DATASET INSIGHTS
# ============================================================

st.divider()
st.header("📈 Dataset Insights by Transaction Type")

type_summary = overview_df.copy()

left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Fraud Count by Transaction Type")

    fraud_chart = (
        type_summary.set_index("type")["fraud_count"]
        .sort_values(ascending=False)
    )

    st.bar_chart(fraud_chart)

with right_chart:
    st.subheader("Fraud Rate by Transaction Type")

    rate_chart = (
        type_summary.set_index("type")["fraud_rate_pct"]
        .sort_values(ascending=False)
    )

    st.bar_chart(rate_chart)

st.subheader("Transaction Type Summary")

display_summary = type_summary.rename(
    columns={
        "type": "Transaction Type",
        "transaction_count": "Total Transactions",
        "fraud_count": "Fraudulent",
        "legitimate_count": "Legitimate",
        "fraud_rate_pct": "Fraud Rate (%)",
    }
)

st.dataframe(
    display_summary[
        [
            "Transaction Type",
            "Total Transactions",
            "Fraudulent",
            "Legitimate",
            "Fraud Rate (%)",
        ]
    ].round(4),
    width="stretch",
    hide_index=True,
)


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "UPI Shield | Experimental fraud-analysis prototype | "
    "Synthetic PaySim dataset | Not a production payment-security system"
)
