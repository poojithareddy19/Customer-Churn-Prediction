# Technical Documentation

This file describes how the code behaves. The README covers the findings and results; this file is the reference for what each part of the code does, how to run it, how it is tested, and what its limits are. Every statement below was checked against the code and `artifacts/training_report.json` on the `fix/doc-review-gaps` branch.

## Components

| Path | Role |
| --- | --- |
| `src/features.py` | Dataset loading and cleaning (`load_dataset`, `clean_frame`), preprocessing, form schema, cost and profit assumptions, input guards (`apply_internet_guard`, `charges_warning`, `category_warning`) |
| `src/train.py` | Splitting, candidate comparison, tuning, calibration, threshold selection, report and artifact export (`train_model`, `load_or_train_bundle`) |
| `src/evaluate.py` | Metrics, expected cost curve, lift and capacity tables, `outcomes_at_cutoffs`, bootstrap intervals, odds ratios, profit curve |
| `src/app_helpers.py` | Streamlit-free helpers for the app: report key fallbacks (`report_cost`, `profit_threshold`, `capacity_rows`), `expected_contact_value`, prediction logging, SHAP explanation |
| `src/score.py` | Batch scoring command, `python -m src.score` |
| `src/segments.py`, `src/experiment.py`, `src/monitoring.py` | DuckDB segment queries, experiment sizing and SRM check, population stability index |
| `app.py` | Streamlit interface |
| `scripts/` | Seven scripts that regenerate `reports/` and the README charts |
| `artifacts/` | Committed model bundle and training report |

## Data and cleaning

`clean_frame(frame, require_target=False)` applies the cleaning steps to a copy of a raw Telco-format frame:

- `TotalCharges` is converted to numeric. Blanks occur only for the 11 customers with tenure 0, who have not been billed yet, so blanks at tenure 0 become 0.0.
- `Churn` is mapped to 0/1 when present; `require_target=True` makes it mandatory.
- `SeniorCitizen` is cast to int.

`load_dataset()` reads the CSV, drops `customerID` and calls `clean_frame(..., require_target=True)`. Its output is identical to the version before `clean_frame` existed.

## Training

`python -m src.train` runs `train_model()`:

1. Stratified 60/20/20 train, validation and test split, `random_state=42`.
2. Five candidate pipelines (balanced logistic regression, Random Forest class-weighted and SMOTE, XGBoost class-weighted and SMOTE) scored by 5-fold stratified CV ROC-AUC on the training split, then tuned with `RandomizedSearchCV` (10 iterations each). The best tuned pipeline is `xgb_balanced_tuned`.
3. The winner is calibrated with `CalibratedClassifierCV(method="sigmoid", cv=3)` fitted on the training split, and scored on the validation split.
4. All operational decision values are chosen on the validation split:
   - the cost threshold (0.08), which minimises expected cost at 65 per wasted offer and 780 per missed churner;
   - the profit threshold (0.34), which maximises profit under the profit assumptions;
   - the capacity cutoffs (`validation_capacity_table`), the probability of the last customer inside the top 5%, 10%, 20%, 30%, 40% and 50%.
5. The final model is recalibrated on train plus validation and scored once on the test split.

The validation probabilities come from the model calibrated on the training split only, while the deployed model is recalibrated on train plus validation. All three kinds of operational value share this convention.

The test split is used only for evaluation. `test_capacity_table` ranks the test split by its own scores and is an evaluation result; the app does not use its cutoffs. `test_outcomes_at_validation_capacity_cutoffs` applies each validation cutoff to the test split and records the customers contacted, share contacted, recall, precision and expected cost.

The bundle saved to `artifacts/churn_model.joblib` holds `model`, `threshold`, `report` and `feature_columns`. The report is also written to `artifacts/training_report.json` with the run timestamp, git commit and a dirty flag.

## App behaviour

`python -m streamlit run app.py` loads the bundle (training first if the artifact is missing) and offers three decision rules:

| Rule | Threshold | Metrics shown after a prediction |
| --- | --- | --- |
| Profit (default) | Validation profit threshold, adjustable with a slider | Expected value of contacting, from `expected_contact_value()`, and 0 for not contacting |
| Cost | Validation cost threshold, adjustable with a slider | Expected cost if not contacted (780 x p) and if contacted (65 x (1 - p)) |
| Capacity | Fixed cutoff from `capacity_rows()`: the validation table, or the test table for artifacts trained before it existed | Same expected costs as the cost rule |

The expected value of contacting is `p x success_rate x MonthlyCharges x retention_months - offer_cost`, with the assumptions read from `report["profit_analysis"]["assumptions"]` (defaults 0.30, 12 months, 65). It is the per-customer expectation of the formula in `profit_curve()`. The profit threshold maximises total profit across all validation customers and does not depend on one customer's charges, so the two can disagree. For example, a one-year contract at 100 a month scores 0.27, below the threshold, with an expected value of +32. A customer paying 20 a month can score 0.42, above the threshold, with an expected value of -35.

Input checks:

- **Consistency guard.** `apply_internet_guard()` sets the add-on fields to "No internet service" when there is no internet service, and `MultipleLines` to "No phone service" when there is no phone service.
- **TotalCharges warning.** `charges_warning()` returns a message when `TotalCharges / (tenure x MonthlyCharges)` is outside 0.68 to 1.58, or when `TotalCharges` is positive at tenure 0. The bounds are the observed range in the dataset (0.689 to 1.574) rounded outward, so no real customer triggers it. The warning is informational: the model scores the values as entered.

Each prediction is appended to `logs/predictions.jsonl` (ignored by git) with `timestamp_utc`, `inputs`, `probability`, `decision_rule`, `threshold`, `verdict` and `warning` (null when there is none). A "Why this score?" panel shows the top five SHAP contributions of the uncalibrated model.

## Batch scoring

```bash
python -m src.score --input customers.csv --output scored.csv [--rule profit|cost|capacity] [--capacity 0.2] [--threshold 0.4]
```

- The input needs the 19 feature columns; `customerID` is copied to the output when present. Missing columns are listed in the error.
- Rows go through `clean_frame()`, then `apply_internet_guard()` row by row, and are scored with the saved model. The scores equal the app path exactly.
- `profit` uses the report's profit threshold, `cost` uses the threshold saved with the model, and `--threshold` overrides either; `decision_rule` then reads `profit (override)` or `cost (override)`. `capacity` flags the top share of the input file by score, ties keeping file order, with the same rounding as `capacity_table()` (0.2 of 7,043 rows flags 1,409). This differs from the app, which applies a fixed cutoff to one customer at a time.
- Output columns: `customerID` (if present), `churn_probability`, `risk_rank` (1 = highest), `decision_rule`, `threshold` (empty for capacity), `flagged`, `charges_warning`, `category_warning`.
- `category_warning()` names categorical fields that are blank or hold a value not seen in training (the values in `FIELD_OPTIONS`, which match the dataset exactly). Such rows are still scored: the one-hot encoder ignores unknown values and the imputer fills blanks. A summary line on stdout gives the number of affected rows.
- Expected errors print one line to stderr and exit 1 without a traceback: missing or empty file, no rows, missing columns, non-numeric `SeniorCitizen`, `tenure` or `MonthlyCharges`, non-numeric non-blank `TotalCharges`, `--capacity` outside (0, 1], `--threshold` with the capacity rule, and a missing artifact.
- The command never trains, and it does not write to `logs/predictions.jsonl`.

## Usage

```bash
pip install -r requirements.txt          # Python 3.11
python -m src.train                      # retrain; overwrites artifacts/
python -m streamlit run app.py           # app
python -m src.score --input WA_Fn-UseC_-Telco-Customer-Churn.csv --output scored.csv
```

Regenerate the reports after retraining with the seven scripts listed in the README (`seed_stability`, `segment_analysis`, `experiment_sizing`, `drift_check`, `logit_inference`, `error_analysis`, `readme_charts`). Cost and profit assumptions can be overridden with `CHURN_FALSE_POSITIVE_COST`, `CHURN_FALSE_NEGATIVE_COST`, `CHURN_OFFER_COST`, `CHURN_OFFER_SUCCESS_RATE` and `CHURN_RETENTION_MONTHS` before training.

`requirements.txt` pins every direct dependency, including SciPy (1.17.1), which `src/experiment.py` imports directly. The pin matches the version pip already resolved, so the environment is unchanged.

## Testing

```bash
pip install -r requirements-dev.txt      # adds pytest and MLflow
python -m pytest -q
```

66 tests across 8 modules:

| Module | Tests | Covers |
| --- | --- | --- |
| `tests/test_app_helpers.py` | 18 | Report key fallbacks, `capacity_rows`, `expected_contact_value` (including consistency with `profit_curve`), prediction logging with and without a warning, SHAP explanations |
| `tests/test_evaluate.py` | 11 | Cost curve, lift and capacity tables, `outcomes_at_cutoffs`, bootstrap intervals, odds ratios, profit curve |
| `tests/test_features.py` | 11 | Guards, input frame order, `load_dataset` cleaning, `charges_warning` and `category_warning` on all 7,043 rows and on bad inputs |
| `tests/test_score.py` | 9 | Batch output shape and IDs, equality with the app path, the guard on inconsistent rows, unknown and blank categories, the capacity rule, the profit threshold, override labelling, missing column and missing artifact errors |
| `tests/test_segments.py` | 7 | DuckDB segment queries and Wilson intervals |
| `tests/test_experiment.py` | 4 | Sample size and SRM check |
| `tests/test_monitoring.py` | 4 | Population stability index |
| `tests/test_train.py` | 2 | Saved feature columns, and an end-to-end training run on 800 rows that checks the report keys and the MLflow run |

CI (`.github/workflows/ci.yml`) runs the suite on every push and pull request with Python 3.11 and `requirements-dev.txt`.

## Limitations

### Observed from the code

- **Capacity cutoffs are a fixed probability, not a live ranking.** The app's capacity rule applies the validation cutoff to one customer at a time, so the share flagged in practice depends on the incoming population. On the test split the realised share was 4.8% to 21.9% for targets of 5% to 20%, and 48.4% for 50%.
- **Calibration mismatch for operational values.** Thresholds and capacity cutoffs come from the model calibrated on the training split, while the deployed model is recalibrated on train plus validation. The convention is consistent across all three, but the deployed probabilities are on a slightly different scale.
- **The default form triggers the TotalCharges warning.** The defaults (tenure 12, MonthlyCharges 70, TotalCharges 1,500) give a ratio of 1.79. They were kept on purpose so the default prediction (56.5%) stays the documented example.
- **Batch scoring flags bad categories but does not reject them.** A row with an unknown or blank category value is scored with that field encoded as all zeros or imputed, and is named in `category_warning`; its score is less reliable. A blank `TotalCharges` with tenure above 0 is filled by the pipeline's median imputer without a warning.
- **SHAP explains the uncalibrated model.** Sigmoid calibration is monotonic, so the direction and ranking hold, but the values are in log-odds of the base model, not changes in the displayed probability.

### Modelling and data

- Cost and profit assumptions are estimates. The offer success rate in particular is an assumption, not data.
- The randomised search is small (10 iterations per family), and the tuned models are within the fold standard deviation of each other. Tuned logistic regression is statistically tied with XGBoost.
- The dataset is a static snapshot, so real drift over time cannot be measured, and the model predicts whether a customer churns, not when.
- Scores are correlational. A high churn score does not mean a customer will respond to an offer.

## Verification Notes

- **Baseline.** Before the gap fixes, `main` at `8d476ae` had 43 passing tests on Python 3.11.16.
- **Retraining.** After the fixes the model was retrained on Python 3.11.16. Every value that existed in the report before is unchanged (float tolerance 1e-9, excluding `run_timestamp`, `git_commit` and `git_dirty`). The only additions are `validation_capacity_table` and `test_outcomes_at_validation_capacity_cutoffs`. The new model gives exactly the same probabilities as the previous one on all 7,043 rows. The report records commit `77a257e` with `git_dirty: false`. The later changes to `src/` only add `category_warning()` and change the batch scoring output; the training code path (`train.py`, `evaluate.py`, `load_dataset()` and the preprocessing) is unchanged since that commit, so the artifact still reflects it.
- **Reports.** All seven report scripts were rerun on the retrained model, and every file in `reports/` and `docs/images/` came out byte-identical. `docs/images/app_prediction.png` was then retaken separately to show the warning and the profit-mode metrics.
- **Environment.** Fresh Python 3.11 environments built from `main`'s and the branch's `requirements.txt` give identical `pip freeze` output.
- **Independent audit.** An audit of commit `68a3939` in a separate session found 46 PASS, 0 FAIL, 2 low-severity WARN and 1 N/A. The N/A was the CI run status, which could not be checked without GitHub access. The two WARNs:
  - Batch scoring accepted unknown category values silently. Later commits add the `category_warning` column and label `--threshold` overrides in `decision_rule`, which the audit also asked about.
  - No test caught a missing guard call in `src/score.py`. A later commit adds `test_inconsistent_rows_are_guarded_like_the_app`, which fails when the guard is removed.
