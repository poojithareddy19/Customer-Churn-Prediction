from __future__ import annotations

import streamlit as st

from src.features import FORM_FIELDS, apply_internet_guard, build_input_frame
from src.train import MODEL_PATH, load_or_train_bundle

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

    slider_threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=float(threshold), step=0.01)
    submit_button = st.form_submit_button("Predict churn risk")

if submit_button:
    guarded_values = apply_internet_guard(values)
    input_frame = build_input_frame(guarded_values)
    churn_probability = float(model.predict_proba(input_frame)[0, 1])
    churn_prediction = int(churn_probability >= slider_threshold)
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
    st.write("Prediction threshold used:", f"{slider_threshold:.2f}")

    with st.expander("Applied input guard", expanded=False):
        st.write(guarded_values)
