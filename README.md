# Telco Customer Churn Prediction

Predicts whether a telecom customer is likely to churn, using a Random Forest classifier trained on the IBM Telco Customer Churn dataset and served through a Streamlit interface that returns a risk verdict and probability.

Selected from three candidate models under 5-fold cross-validation. **ROC-AUC 0.8234** on a held-out test set, against a majority-class baseline that catches zero churners.

---

## Problem

Acquiring a telecom subscriber costs substantially more than retaining one. A retention team can only act on customers it can identify in advance, so the useful output is not a churn label but a ranked probability that lets limited retention budget go to the highest-risk accounts first.

## Dataset

- Source: IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, included in this repo)
- 7,043 customers, 20 features covering demographics, subscribed services, and account information
- Target: `Churn` (Yes/No), imbalanced at roughly 73/27

## Pipeline

1. **Data loading and EDA**
   Distribution plots for every categorical column, correlation heatmap across `tenure`, `MonthlyCharges`, and `TotalCharges`, and box plots for outlier inspection.

2. **Cleaning**
   `TotalCharges` arrives as an object dtype rather than float. 11 rows contain a single space instead of a null value, so `isnull()` does not catch them. All 11 have zero tenure, meaning the customer had not yet been billed, so the semantically correct fill is 0.0 rather than the column mean.

3. **Encoding**
   Categorical columns are label-encoded and the fitted `LabelEncoder` objects are persisted to `encoders.pkl`. Label encoding rather than one-hot keeps the feature space at 19 columns instead of roughly 45, which suits tree models since they split on thresholds rather than assuming ordinal distance.

4. **Splitting**
   80/20 train-test split, `random_state=42`. 5,634 training rows, 1,409 test rows.

5. **Balancing**
   SMOTE is applied **only to the training partition, after the split**, rebalancing 1,496 churners up to 4,138. Applying SMOTE before the split would interpolate synthetic minority points across the train-test boundary and leak test information into training.

6. **Model comparison**
   Decision Tree, Random Forest, and XGBoost compared under 5-fold cross-validation on library defaults.

7. **Serialisation**
   The selected model and the training column order are pickled together into `customer_churn_model.pkl`. Storing the feature names alongside the model is what allows the app to reindex incoming rows into the exact training order.

8. **Deployment**
   Streamlit app loading the pickled model and encoders, returning a risk verdict and probability.

---

## Results

### Model selection, 5-fold cross-validation

| Model         | CV Accuracy |
| ------------- | ----------- |
| Decision Tree | 0.78        |
| Random Forest | 0.84        |
| XGBoost       | 0.83        |

Random Forest was selected. XGBoost is competitive but more sensitive to hyperparameters, which were not tuned in this project, so Random Forest's stronger default behaviour won out.

### Held-out test set, 1,409 customers

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 0.7786 |
| ROC-AUC           | 0.8234 |
| Recall (churn)    | 0.587  |
| Precision (churn) | 0.581  |
| F1 (churn)        | 0.584  |

Confusion matrix:

```
              predicted
              no    yes
actual  no   878    158
       yes   154    219
```

**Why not accuracy.** The majority class is 73.5% of the data, so a model that predicts "nobody churns" scores 73.5% accuracy while identifying zero churners. That model is commercially worthless. ROC-AUC and recall on the churn class are the metrics that reflect whether the system is actually useful to a retention team.

### A known bias in the CV scores

Cross-validation ran on the SMOTE-resampled training set. Because SMOTE interpolates new points between existing minority neighbours, synthetic points derived from the same neighbourhoods end up in both the training and validation folds. That is why the 0.84 CV accuracy sits above the 0.7786 test accuracy.

The correct fix is to wrap SMOTE and the classifier in an `imblearn.pipeline.Pipeline` and pass that to `cross_val_score`, so resampling happens inside each fold on training data only. The test-set numbers above are unaffected, since SMOTE never touched the test partition.

---

## Feature importance

Gini importance from the fitted Random Forest:

| Feature         | Importance |
| --------------- | ---------- |
| TotalCharges    | 14.2%      |
| MonthlyCharges  | 13.7%      |
| Contract        | 12.7%      |
| tenure          | 12.2%      |
| OnlineSecurity  | 8.7%       |
| TechSupport     | 7.4%       |
| PaymentMethod   | 4.4%       |
| OnlineBackup    | 3.8%       |
| Dependents      | 3.0%       |
| InternetService | 3.0%       |

Reproduce with:

```python
import pandas as pd
pd.Series(rfc.feature_importances_, index=x.columns).nlargest(10)
```

## Key business finding

Contract type is not the highest-Gini feature, but it is the strongest business lever:

| Contract       | Churn rate |
| -------------- | ---------- |
| Month-to-month | 42.7%      |
| One year       | 11.3%      |
| Two year       | 2.8%       |

Month-to-month customers churn at roughly **15x** the rate of two-year customers. Charges and tenure carry more raw predictive weight, but contract length is the variable a retention team can actually act on, through migration offers and incentives.

---

## Project structure

```
├── Customer_Churn_Prediction_Using_ML.ipynb   # EDA, preprocessing, training, evaluation
├── app.py                                     # Streamlit interface
├── customer_churn_model.pkl                   # {model, feature_names}
├── encoders.pkl                               # fitted LabelEncoders
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # dataset
├── PROJECT_DOCUMENTATION.md                   # detailed technical documentation
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/poojithareddy19/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
```

## Usage

Retrain and reproduce the results:

```bash
jupyter notebook "Customer_Churn_Prediction_Using_ML.ipynb"
```

This regenerates `customer_churn_model.pkl` and `encoders.pkl`.

Run the app:

```bash
streamlit run app.py
```

The interface exposes 10 input fields covering the highest-importance features and defaults the rest to modal values, trading a small amount of predictive coverage for a form a retention agent will actually complete.

---

## Design decisions

| Decision                            | Rationale                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| SMOTE after the split               | Prevents synthetic minority points from crossing the train-test boundary                                        |
| Label encoding over one-hot         | 19 features instead of ~45; tree models split on thresholds so the artificial ordering does not mislead them    |
| `TotalCharges` blanks filled with 0.0 | Semantically correct: all 11 rows have zero tenure, so the customer had not been billed                       |
| Pickling the encoders               | The app must map "Mailed check" to the same integer the model learned, which refitting on one row would not do  |
| Persisting the training column order | Reindexing incoming rows guarantees columns reach the model in the order it was fitted on                       |
| Random Forest over XGBoost          | Stronger performance on untuned defaults; XGBoost needs tuning to compete                                       |
| Decoupled training and inference     | The app loads two pickle files and needs neither the dataset nor the training code                             |

## Limitations

- **No hyperparameter tuning.** All three models were compared on library defaults. Tuning XGBoost would likely change the selection.
- **CV scores are optimistic.** See the note under Results.
- **Label encoding imposes false ordinality.** Acceptable for tree splits, but this encoding could not be reused for a linear model without switching to one-hot, and SMOTE on label-encoded categoricals interpolates between category codes. SMOTENC would be the correct variant.
- **Static snapshot.** No time dimension, so the model cannot express when a customer is likely to churn, only whether.
- **Correlation, not causation.** A customer flagged as high-risk on a month-to-month contract does not mean moving them to an annual contract will retain them.

## Roadmap

- [ ] Wrap SMOTE in an `imblearn` pipeline inside cross-validation
- [ ] Hyperparameter tuning for Random Forest and XGBoost
- [ ] Threshold tuning to raise churn recall above 0.587 at an accepted precision cost
- [ ] SHAP values for per-customer explanations in the app
- [ ] SMOTENC in place of SMOTE for categorical-aware resampling

## License

Apache-2.0. See `LICENSE`.
