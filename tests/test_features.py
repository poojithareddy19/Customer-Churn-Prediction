from src.features import FEATURE_COLUMNS, apply_internet_guard, build_input_frame, load_dataset


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


def test_phone_guard_forces_multiple_lines_to_no_phone_service():
    guarded = apply_internet_guard({"PhoneService": "No", "MultipleLines": "Yes", "InternetService": "DSL"})
    assert guarded["MultipleLines"] == "No phone service"


def test_load_dataset_fills_unbilled_total_charges_with_zero():
    frame = load_dataset()
    assert frame["TotalCharges"].isna().sum() == 0
    assert (frame.loc[frame["tenure"] == 0, "TotalCharges"] == 0.0).all()
    assert set(frame["Churn"].unique()) == {0, 1}
