from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = Path(__file__).resolve().parents[1] / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
CUSTOMER_ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Decision costs, in the dataset's currency units. Defaults are derived from the mean MonthlyCharges
# in the dataset (about 65): a retention offer is assumed to cost one month of revenue, and a lost
# customer is assumed to cost twelve months of revenue. Override with the environment variables
# CHURN_FALSE_POSITIVE_COST and CHURN_FALSE_NEGATIVE_COST, then retrain to move the threshold.
FALSE_POSITIVE_COST_RUPEES = float(os.environ.get("CHURN_FALSE_POSITIVE_COST", 65.0))
FALSE_NEGATIVE_COST_RUPEES = float(os.environ.get("CHURN_FALSE_NEGATIVE_COST", 780.0))

FIELD_OPTIONS: dict[str, list[Any]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@dataclass(frozen=True)
class FormField:
    name: str
    label: str
    field_type: str
    default: Any
    options: list[Any] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    format: str | None = None


FORM_FIELDS = [
    FormField("gender", "Gender", "select", "Female", FIELD_OPTIONS["gender"]),
    FormField("SeniorCitizen", "Senior Citizen", "select", 0, FIELD_OPTIONS["SeniorCitizen"]),
    FormField("Partner", "Partner", "select", "No", FIELD_OPTIONS["Partner"]),
    FormField("Dependents", "Dependents", "select", "No", FIELD_OPTIONS["Dependents"]),
    FormField("PhoneService", "Phone Service", "select", "Yes", FIELD_OPTIONS["PhoneService"]),
    FormField("MultipleLines", "Multiple Lines", "select", "No", FIELD_OPTIONS["MultipleLines"]),
    FormField("InternetService", "Internet Service", "select", "Fiber optic", FIELD_OPTIONS["InternetService"]),
    FormField("OnlineSecurity", "Online Security", "select", "No", FIELD_OPTIONS["OnlineSecurity"]),
    FormField("OnlineBackup", "Online Backup", "select", "No", FIELD_OPTIONS["OnlineBackup"]),
    FormField("DeviceProtection", "Device Protection", "select", "No", FIELD_OPTIONS["DeviceProtection"]),
    FormField("TechSupport", "Tech Support", "select", "No", FIELD_OPTIONS["TechSupport"]),
    FormField("StreamingTV", "Streaming TV", "select", "No", FIELD_OPTIONS["StreamingTV"]),
    FormField("StreamingMovies", "Streaming Movies", "select", "No", FIELD_OPTIONS["StreamingMovies"]),
    FormField("Contract", "Contract", "select", "Month-to-month", FIELD_OPTIONS["Contract"]),
    FormField("PaperlessBilling", "Paperless Billing", "select", "Yes", FIELD_OPTIONS["PaperlessBilling"]),
    FormField("PaymentMethod", "Payment Method", "select", "Electronic check", FIELD_OPTIONS["PaymentMethod"]),
    FormField("tenure", "Tenure (months)", "number", 12, None, 0, 72, 1, "%d"),
    FormField("MonthlyCharges", "Monthly Charges", "number", 70.0, None, 0.0, 200.0, 0.05, "%.2f"),
    FormField("TotalCharges", "Total Charges", "number", 1500.0, None, 0.0, 10000.0, 1.0, "%.2f"),
]


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path)
    frame = frame.drop(columns=[CUSTOMER_ID_COLUMN], errors="ignore")
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    # Blank TotalCharges only occurs for customers with tenure 0 (not billed yet), so 0.0 is the
    # semantically correct value. The median imputer in the pipeline stays as a safety net.
    not_yet_billed = frame["TotalCharges"].isna() & (frame["tenure"] == 0)
    frame.loc[not_yet_billed, "TotalCharges"] = 0.0
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = frame[FEATURE_COLUMNS].copy()
    target = frame[TARGET_COLUMN].astype(int).copy()
    return features, target


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def apply_internet_guard(values: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(values)
    if guarded.get("PhoneService") == "No":
        guarded["MultipleLines"] = "No phone service"
    if guarded.get("InternetService") == "No":
        for field in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            guarded[field] = "No internet service"
    return guarded


def build_input_frame(values: dict[str, Any]) -> pd.DataFrame:
    normalized = {column: values[column] for column in FEATURE_COLUMNS}
    return pd.DataFrame([normalized], columns=FEATURE_COLUMNS)
