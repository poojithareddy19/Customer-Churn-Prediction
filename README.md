# Telco Customer Churn Prediction

Predicts whether a telecom customer is likely to churn, using a calibrated gradient-boosting pipeline trained on the IBM Telco Customer Churn dataset and served through a Streamlit interface that returns a churn probability, a risk verdict, and the expected cost of contacting or not contacting the customer.

Five candidate pipelines were compared under 5-fold stratified cross-validation, logistic regression, Random Forest and XGBoost variants were tuned with randomised search, and the winner was calibrated and evaluated on a held-out test set. **Test ROC-AUC 0.836**, against a majority-class baseline that catches zero churners.

---

## Problem

Acquiring a telecom subscriber costs substantially more than retaining one. A retention team can only act on customers it can identify in advance, so the useful output is not a churn label but a calibrated probability that lets limited retention budget go to the highest-risk accounts first, together with a decision threshold that reflects the relative cost of a missed churner versus an unnecessary retention offer.

## Dataset

- Source: IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, included in this repo)
- 7,043 customers, 19 features covering demographics, subscribed services, and account information
- Target: `Churn` (Yes/No), imbalanced at roughly 73/27

## Pipeline

All training and inference code lives in `src/`. The notebook is kept as the original exploratory analysis but is no longer the source of the deployed model.

1. **Loading and cleaning** (`src/features.py`)
   `customerID` is dropped, `Churn` is mapped to 0/1, and `TotalCharges` is converted to numeric. The 11 rows with a blank `TotalCharges` all have zero tenure, meaning the customer had not yet been billed, so they are filled with 0.0. A median imputer inside the pipeline remains as a safety net for unseen data.

2. **Preprocessing** (`src/features.py`)
   A `ColumnTransformer` scales the four numeric columns and one-hot encodes the fifteen categorical columns with `handle_unknown="ignore"`. Everything is fitted inside the model pipeline, so there are no separate encoder files and no leakage from the test set.

3. **Splitting** (`src/train.py`)
   Stratified 60/20/20 train, validation, test split with `random_state=42`. The validation split is used only to choose the decision threshold. The test split is touched once, at the end.

4. **Candidate comparison**
   Five pipelines are scored by ROC-AUC under 5-fold stratified cross-validation on the training split: balanced logistic regression, Random Forest (class-weighted and SMOTE), and XGBoost (class-weighted and SMOTE). SMOTE is applied through an `imblearn` pipeline, so resampling happens inside each fold on training data only.

5. **Tuning**
   The Random Forest and XGBoost variants and a class-weighted logistic regression (`C` on a log scale from 0.001 to 100, `l1` or `l2` penalty, `liblinear` solver) are tuned with `RandomizedSearchCV` (10 iterations, ROC-AUC scoring). The tuned pipeline with the highest mean CV ROC-AUC is selected, with no preference for any model family.

6. **Calibration and threshold**
   The winner is wrapped in `CalibratedClassifierCV` (sigmoid). Calibration is fitted on the training split, the resulting probabilities are scored on the validation split, and the decision threshold that minimises expected cost is recorded. The final model is then recalibrated on train plus validation.

7. **Evaluation** (`src/evaluate.py`)
   ROC-AUC, Brier score, confusion matrix at the chosen threshold, a calibration table, and permutation importance on the test split.

8. **Serialisation**
   The calibrated pipeline, the threshold, the feature column order, and the full report are saved together to `artifacts/churn_model.joblib`. A human-readable copy of the report is written to `artifacts/training_report.json`.

9. **Deployment** (`app.py`)
   Streamlit app that loads the artifact, exposes all 19 inputs, applies consistency guards (no internet service implies no add-ons, no phone service implies no multiple lines), and shows the probability, verdict, and expected cost of each decision. The threshold is adjustable with a slider.

---

## Results

All numbers come from `artifacts/training_report.json` and are reproducible with `python -m src.train`.

### Candidate comparison, 5-fold cross-validated ROC-AUC on the training split

| Pipeline                        | Defaults | Tuned  |
| ------------------------------- | -------- | ------ |
| Logistic regression (balanced)  | 0.8443   | 0.8444 |
| Random Forest (class-weighted)  | 0.8227   | 0.8453 |
| Random Forest + SMOTE           | 0.8216   | 0.8437 |
| XGBoost + SMOTE                 | 0.8157   | 0.8453 |
| XGBoost (class-weighted)        | 0.8132   | **0.8481** |

Class-weighted XGBoost was selected after tuning (350 shallow trees, depth 4, learning rate 0.01, subsample 0.85, column subsample 0.7, L2 regularisation 5). Tuning barely moves logistic regression (best: `l2`, `C` = 2.64), and it stays within 0.004 ROC-AUC of the tuned tree models, which is typical for this dataset. Its fold-to-fold standard deviation (0.021) is larger than that gap, so the ranking among the top candidates is not decisive.

### Held-out splits

| Metric                    | Validation (1,409) | Test (1,409) |
| ------------------------- | ------------------ | ------------ |
| ROC-AUC                   | 0.8596             | 0.8361       |
| Brier score               | 0.1302             | 0.1402       |
| Cost-optimal threshold    | 0.08               | 0.05         |
| Recall (churn) at 0.08    | 0.968              | 0.944        |
| Precision (churn) at 0.08 | 0.39               | 0.40         |
| Accuracy at 0.08          | 0.594              | 0.606        |

Test confusion matrix at the deployed threshold of 0.08:

```
              predicted
              no    yes
actual  no   501    534
       yes    21    353
```

**Why not accuracy.** The majority class is 73.5% of the data, so a model that predicts "nobody churns" scores 73.5% accuracy while identifying zero churners. ROC-AUC measures ranking quality independent of the threshold, and the Brier score measures whether the probabilities themselves are trustworthy, which matters because the app shows them to a human.

### About the threshold

The threshold is chosen to minimise expected cost under the defaults in `src/features.py`, which are derived from the dataset's mean monthly charge of about 65: a wasted retention offer is assumed to cost one month of revenue (65) and a missed churner twelve months of revenue (780). With a 12:1 cost ratio the break-even probability is about 7.7%, and the validation split picks 0.08. On the test split that flags 63% of customers and catches 94% of churners. The test split's own cost curve prefers 0.05, so the optimum is flat in that region and the exact value should not be over-interpreted.

A missed churner is expensive relative to an offer, so the cost-optimal policy is deliberately generous with offers. If the team can only contact a fixed share of customers, rank by probability and take the top slice instead, or override the costs through the environment variables described under Usage. The app's threshold slider exists so this tradeoff can be explored interactively.

---

## Feature importance

Permutation importance (drop in test ROC-AUC when the column is shuffled, mean of 10 repeats):

| Feature         | Importance |
| --------------- | ---------- |
| Contract        | 0.087      |
| tenure          | 0.030      |
| InternetService | 0.016      |
| MonthlyCharges  | 0.010      |
| TotalCharges    | 0.007      |
| OnlineSecurity  | 0.005      |
| TechSupport     | 0.002      |
| StreamingMovies | 0.002      |

Permutation importance is preferred over Gini importance because it is measured on held-out data and is not biased toward high-cardinality or continuous columns. Under this measure contract type is by far the strongest signal.

## Key business finding

| Contract       | Churn rate |
| -------------- | ---------- |
| Month-to-month | 42.7%      |
| One year       | 11.3%      |
| Two year       | 2.8%       |

Month-to-month customers churn at roughly **15x** the rate of two-year customers. Contract length is both the top predictor and the variable a retention team can actually act on, through migration offers and incentives.

---

## Project structure

```
├── app.py                                     # Streamlit interface
├── src/
│   ├── features.py                            # dataset loading, preprocessing, form schema, cost constants
│   ├── train.py                               # candidate comparison, tuning, calibration, artifact export
│   └── evaluate.py                            # metrics, cost curve, calibration table, permutation importance
├── tests/
│   ├── test_features.py
│   ├── test_evaluate.py
│   └── test_train.py
├── artifacts/                                 # committed; regenerate with python -m src.train
│   ├── churn_model.joblib                     # {model, threshold, report, feature_columns}
│   └── training_report.json
├── Customer_Churn_Prediction_Using_ML.ipynb   # original EDA and first model (superseded by src/)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # dataset
├── requirements.txt                           # app and training dependencies
├── requirements-dev.txt                       # adds pytest for running the tests
└── README.md
```

## Installation

Python 3.11 is recommended; the pinned versions in `requirements.txt` were tested on it.

```bash
git clone https://github.com/poojithareddy19/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Usage

The trained artifact is committed, so the app runs straight after installation. To retrain (a few minutes; prints the test metrics when done):

```bash
python -m src.train
```

Run the app:

```bash
python -m streamlit run app.py
```

If `artifacts/churn_model.joblib` is ever missing, the app trains on first load and shows a warning while it does so.

The decision costs can be overridden without editing code. Set the variables in the same terminal before training and the threshold is recomputed. The app reads the costs saved with the model, so restart it after retraining.

```powershell
# Windows PowerShell
$env:CHURN_FALSE_POSITIVE_COST = "100"
$env:CHURN_FALSE_NEGATIVE_COST = "1500"
python -m src.train
```

```bat
:: Windows Command Prompt (cmd.exe)
set CHURN_FALSE_POSITIVE_COST=100
set CHURN_FALSE_NEGATIVE_COST=1500
python -m src.train
```

```bash
# macOS / Linux
export CHURN_FALSE_POSITIVE_COST=100
export CHURN_FALSE_NEGATIVE_COST=1500
python -m src.train
```

The profit analysis in the report uses three more assumptions, also read at training time: `CHURN_OFFER_COST` (default 65), `CHURN_OFFER_SUCCESS_RATE` (default 0.30, the share of contacted churners who stay, which is an assumption and not measured in this data) and `CHURN_RETENTION_MONTHS` (default 12). They do not change the deployed threshold.

Run the tests (installs pytest on top of the app dependencies):

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

---

## Design decisions

| Decision                                  | Rationale                                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| One-hot encoding inside a pipeline        | No separate encoder files, no false ordinality, and the same preprocessing is reusable by linear and tree models    |
| SMOTE inside the CV pipeline              | Resampling happens per fold on training data only, so cross-validation scores are honest                            |
| ROC-AUC as the selection metric           | Threshold-independent, so model choice is not entangled with the business cost tradeoff                              |
| Separate validation split for the threshold | The threshold is a hyperparameter; choosing it on the test split would make the test metrics optimistic             |
| Calibrate before choosing the threshold   | The deployed model outputs calibrated probabilities, so the threshold must be picked on that same scale               |
| Cost-based threshold                      | Makes the FP/FN tradeoff explicit and tunable instead of hiding it behind the default 0.5                              |
| Permutation importance                    | Measured on held-out data and not biased toward continuous columns, unlike Gini importance                           |
| `TotalCharges` blanks filled with 0.0     | All 11 rows have zero tenure, so the customer had not yet been billed                                                |
| Feature order persisted with the model    | The app builds its input frame from the saved column list, so form and model cannot drift apart                      |

## Limitations

- **Cost assumptions are estimates.** The defaults are derived from the dataset's mean monthly charge, not from a real offer cost or customer lifetime value. Replace them through the environment variables before the threshold is used operationally.
- **Small randomised search.** 10 iterations per model family. A larger search or Bayesian optimisation might change the winner, since the tuned models are within 0.005 ROC-AUC of each other.
- **Static snapshot.** No time dimension, so the model cannot express when a customer is likely to churn, only whether.
- **Correlation, not causation.** A customer flagged as high-risk on a month-to-month contract does not mean moving them to an annual contract will retain them.

## Roadmap

- [x] Wrap SMOTE in an `imblearn` pipeline inside cross-validation
- [x] Hyperparameter tuning for Random Forest and XGBoost
- [x] Cost-based threshold tuning on a separate validation split
- [x] Probability calibration and Brier score reporting
- [x] Make the cost assumptions configurable through environment variables
- [ ] Capacity-based thresholding (top N% of customers by probability)
- [x] SHAP values for per-customer explanations in the app
- [ ] SMOTENC in place of SMOTE for categorical-aware resampling
- [ ] Move the notebook EDA onto `src/` helpers so it runs locally without the Colab path

## License

Apache-2.0. See `LICENSE`.
