import copy

import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    FORM_FIELDS,
    apply_internet_guard,
    build_input_frame,
    category_warning,
    charges_warning,
    load_dataset,
)


def test_internet_guard_forces_add_on_fields_to_no_internet_service():
    values = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "No",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 1,
        "MonthlyCharges": 20.0,
        "TotalCharges": 20.0,
    }

    guarded = apply_internet_guard(values)

    for field in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        assert guarded[field] == "No internet service"


def test_build_input_frame_keeps_training_column_order():
    values = {column: 0 if column in {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"} else "No" for column in FEATURE_COLUMNS}
    frame = build_input_frame(values)
    assert list(frame.columns) == FEATURE_COLUMNS


def test_build_input_frame_uses_given_column_list():
    values = {column: 0 if column in {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"} else "No" for column in FEATURE_COLUMNS}
    saved_columns = list(reversed(FEATURE_COLUMNS))
    frame = build_input_frame(values, saved_columns)
    assert list(frame.columns) == saved_columns


def test_phone_guard_forces_multiple_lines_to_no_phone_service():
    guarded = apply_internet_guard({"PhoneService": "No", "MultipleLines": "Yes", "InternetService": "DSL"})
    assert guarded["MultipleLines"] == "No phone service"


def test_load_dataset_fills_unbilled_total_charges_with_zero():
    frame = load_dataset()
    assert frame["TotalCharges"].isna().sum() == 0
    assert (frame.loc[frame["tenure"] == 0, "TotalCharges"] == 0.0).all()
    assert set(frame["Churn"].unique()) == {0, 1}


def test_charges_warning_silent_for_every_real_customer():
    frame = load_dataset()
    warnings = [charges_warning(row) for row in frame[FEATURE_COLUMNS].to_dict(orient="records")]
    assert len(warnings) == 7043
    assert all(warning is None for warning in warnings)


def test_charges_warning_flags_implausible_form_defaults():
    defaults = {field.name: field.default for field in FORM_FIELDS}
    # 1500 / (12 * 70) is about 1.79, above every customer in the data.
    assert charges_warning(defaults) is not None
    assert charges_warning({"tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0}) is None


def test_charges_warning_flags_positive_total_with_zero_tenure():
    assert charges_warning({"tenure": 0, "MonthlyCharges": 70.0, "TotalCharges": 70.0}) is not None
    assert charges_warning({"tenure": 0, "MonthlyCharges": 70.0, "TotalCharges": 0.0}) is None


def test_charges_warning_does_not_mutate_input():
    values = {field.name: field.default for field in FORM_FIELDS}
    original = copy.deepcopy(values)
    charges_warning(values)
    assert values == original


def test_category_warning_silent_for_every_real_customer_and_the_form():
    rows = load_dataset()[FEATURE_COLUMNS].to_dict(orient="records")
    assert all(category_warning(row) is None for row in rows)
    assert category_warning({field.name: field.default for field in FORM_FIELDS}) is None


def test_category_warning_names_unknown_and_blank_fields():
    values = {field.name: field.default for field in FORM_FIELDS}
    values["Contract"] = "Monthly"
    values["PaymentMethod"] = float("nan")

    message = category_warning(values)

    assert "Contract has unknown value 'Monthly'" in message
    assert "PaymentMethod is blank" in message
    # The input is left as it was.
    assert values["Contract"] == "Monthly"
    assert pd.isna(values["PaymentMethod"])
