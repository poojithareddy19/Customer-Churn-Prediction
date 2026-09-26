import joblib

from src.features import FEATURE_COLUMNS
from src.train import MODEL_PATH, load_or_train_bundle


def test_load_or_train_bundle_returns_saved_feature_columns():
    saved = joblib.load(MODEL_PATH)
    bundle = load_or_train_bundle()
    assert bundle.feature_columns == list(saved["feature_columns"])
    assert bundle.feature_columns == FEATURE_COLUMNS
