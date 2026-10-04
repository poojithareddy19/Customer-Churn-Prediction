from __future__ import annotations

import pandas as pd
import streamlit as st

from src.features import FORM_FIELDS, apply_internet_guard, build_input_frame
from src.train import MODEL_PATH, load_or_train_bundle

COST_RULE = "Cost threshold"
CAPACITY_RULE = "Capacity (top N%)"

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Telco Customer Churn Prediction")
st.caption("A raw-feature pipeline trained from the Telco churn dataset and selected by ROC-AUC.")


@st.cache_resource
def load_bundle():
    return load_or_train_bundle()


if not MODEL_PATH.exists():
    st.warning(
        "No trained model found in artifacts/. Training will run now and can take several minutes. "
        "Run `python -m src.train` once beforehand to avoid this on every fresh deployment."
    )

with st.spinner("Loading model..."):
    bundle = load_bundle()
model = bundle.model
threshold = bundle.threshold
report = bundle.report

st.info(
    f"Selection metric: ROC-AUC. Validation threshold from expected cost: {threshold:.2f}. "
    f"Cost assumptions: wasted offer = {report['false_positive_cost_rupees']:,.0f}, missed churner = {report['false_negative_cost_rupees']:,.0f}."
)

with st.expander("Model summary", expanded=False):
    st.write("Best model:", report["best_model_name"])
    st.write("Cross-validated ROC-AUC (training folds):", round(report["tuned_models"][0]["cv_roc_auc"], 4))
    if "validation_metrics" in report:
        st.write("Validation ROC-AUC:", round(report["validation_metrics"]["roc_auc"], 4))
    st.write("Test ROC-AUC:", round(report["test_metrics"]["roc_auc"], 4))
    st.write("Test Brier score:", round(report["test_metrics"]["brier_score"], 4))

capacity_rows = report.get("test_capacity_table", [])
if capacity_rows:
    with st.expander("Lift and capacity tables (test split)", expanded=False):
        st.write("Lift by decile of predicted probability (decile 1 = highest risk):")
        st.dataframe(pd.DataFrame(report["test_lift_table"]), hide_index=True)
        st.write("Contacting the top share of customers by predicted probability:")
        st.dataframe(pd.DataFrame(capacity_rows), hide_index=True)

# Older artifacts have no capacity table, so only the cost rule is offered for them.
rule_options = [COST_RULE, CAPACITY_RULE] if capacity_rows else [COST_RULE]
decision_rule = st.radio("Decision rule", rule_options, horizontal=True)
capacity_threshold = None
if decision_rule == CAPACITY_RULE:
    capacity_lookup = {row["capacity"]: row for row in capacity_rows}
    selected_capacity = st.selectbox(
        "Contact the top share of customers",
        list(capacity_lookup),
        format_func=lambda capacity: f"Top {capacity:.0%}",
    )
    capacity_threshold = float(capacity_lookup[selected_capacity]["probability_cutoff"])
    st.caption(
        f"Probability cutoff for the top {selected_capacity:.0%} on the test split: {capacity_threshold:.3f}. "
        "Customers at or above it are flagged."
    )

with st.form("churn_prediction_form"):
    st.subheader("Customer Details")
    columns = st.columns(3)
    values = {}

    for index, field in enumerate(FORM_FIELDS):
        column = columns[index % 3]
        with column:
            if field.field_type == "select":
                values[field.name] = st.selectbox(field.label, field.options, index=field.options.index(field.default))
            else:
                values[field.name] = st.number_input(
                    field.label,
                    min_value=field.min_value,
                    max_value=field.max_value,
                    value=field.default,
                    step=field.step,
                    format=field.format,
                )

    if capacity_threshold is None:
        decision_threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=float(threshold), step=0.01)
    else:
        decision_threshold = capacity_threshold
    submit_button = st.form_submit_button("Predict churn risk")

if submit_button:
    guarded_values = apply_internet_guard(values)
    input_frame = build_input_frame(guarded_values, bundle.feature_columns)
    churn_probability = float(model.predict_proba(input_frame)[0, 1])
    churn_prediction = int(churn_probability >= decision_threshold)
    # Cost of each possible decision, so the two numbers can be compared directly.
    cost_if_not_contacted = report["false_negative_cost_rupees"] * churn_probability
    cost_if_contacted = report["false_positive_cost_rupees"] * (1 - churn_probability)

    st.subheader("Prediction Result")
    if churn_prediction:
        st.error(f"High churn risk: {churn_probability:.1%} probability")
    else:
        st.success(f"Lower churn risk: {churn_probability:.1%} probability")

    cost_columns = st.columns(2)
    cost_columns[0].metric("Expected cost if not contacted", f"{cost_if_not_contacted:,.0f}")
    cost_columns[1].metric("Expected cost if contacted", f"{cost_if_contacted:,.0f}")
    st.write("Decision rule:", decision_rule)
    st.write("Prediction threshold used:", f"{decision_threshold:.3f}")

    with st.expander("Applied input guard", expanded=False):
        st.write(guarded_values)
