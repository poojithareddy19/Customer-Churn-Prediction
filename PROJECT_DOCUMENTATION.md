# Detailed Project Documentation

## 1. Project Overview
The **Customer Churn Prediction** project is a Machine Learning application designed to identify customers who are likely to discontinue their service with a telecommunications provider. By analyzing customer demographics, account information, and service usage, the model predicts the probability of "Churn" (leaving the service).

## 2. Technical Architecture

### 2.1. Technology Stack
*   **Language**: Python 3.x
*   **Machine Learning**: Scikit-Learn (Random Forest Classifier)
*   **Data Handling**: Pandas, NumPy
*   **Imbalance Handling**: Imbalanced-Learn (SMOTE)
*   **Frontend**: Streamlit
*   **Serialization**: Pickle

### 2.2. File Descriptions
*   **`Customer_Churn_Prediction_Using_ML.ipynb`**: The research and development notebook.
    *   *EDA*: Exploratory Data Analysis to understand distributions and correlations.
    *   *Preprocessing*: Label encoding categorization, handling missing values in `TotalCharges`.
    *   *SMOTE*: Synthetic Minority Over-sampling Technique used to balance the dataset (since Churn=Yes is typically the minority class).
    *   *Training*: Trained multiple models (Decision Tree, Random Forest, XGBoost) and selected Random Forest for best performance.
*   **`app.py`**: The production-ready web application script.
    *   Loads the trained model and encoders.
    *   Renders a user-friendly form.
    *   Maps user inputs to the format expected by the model.
    *   Displays prediction results (Risk Level and Probability).
*   **`customer_churn_model.pkl`**: A serialized dictionary containing:
    *   The trained `RandomForestClassifier` object.
    *   A list of feature names to ensure input column order matches training.
*   **`encoders.pkl`**: Serialized `LabelEncoder` objects for transforming categorical text inputs (e.g., "Yes"/"No") into numbers (1/0).

## 3. Model Details

### 3.1. Algorithm
**Random Forest Classifier** was chosen for its robustness against overfitting and its ability to handle non-linear relationships in data.

### 3.2. Feature Importance
The interface focuses on the most significant predictors of churn to streamline the user experience:
1.  **Contract**: Month-to-month contracts are strong indicators of potential churn.
2.  **Tenure**: Lower tenure customers are at higher risk.
3.  **Charges (Monthly & Total)**: Pricing sensitivity is a major factor.
4.  **Tech Support / Online Security**: Customers without these services churn more often.

### 3.3. Class Imbalance Strategy
To prevent the model from being biased towards the majority class (No Churn), **SMOTE** was applied during training. This creates synthetic examples of "Churners" so the model learns to identify them effectively.

## 4. User Interface Guide

### 4.1. Inputs
The `app.py` simplifies data entry by asking only for high-impact fields:
*   **Demographics**: Dependents.
*   **Services**: Internet Service, Online Security, Online Backup, Tech Support.
*   **Account**: Contract Type, Payment Method, Tenure, Monthly Charges, Total Charges.

*Note: Other fields (like Gender, Partner, StreamingTV) are filled with statistical modes (most frequent values) in the background to maintain model compatibility without burdening the user.*

### 4.2. Outputs
*   **High Risk of Churn**: The model predicts the customer is likely to leave. (Red Box)
*   **Low Risk of Churn**: The model predicts the customer will stay. (Green Box)
*   **Probability**: A percentage score indicating the model's confidence.

## 5. Maintenance
To update the model:
1.  Add new data to the CSV file.
2.  Run the Jupyter Notebook to re-train.
3.  The `.pkl` files will be overwritten with the new model.
4.  Restart the Streamlit app to load the new model.
