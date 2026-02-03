# Telco Customer Churn Prediction

This project predicts whether a customer is likely to churn (leave the service) based on their subscription details and usage patterns. It uses a **Random Forest Classifier** trained on the Telco Customer Churn dataset and provides a user-friendly web interface using **Streamlit**.

## 🚀 Project Pipeline

1.  **Data Loading & Understanding**:
    *   Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
    *   Features: Demographics, Services, Account Information.
    *   Target: `Churn` (Yes/No).

2.  **Data Preprocessing**:
    *   **Cleaning**: Handled missing values in `TotalCharges`.
    *   **Encoding**: Converted categorical variables into numeric format using `LabelEncoder`.
    *   **Balancing**: Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance in the target variable.
    *   **Splitting**: Split data into training (80%) and testing (20%) sets.

3.  **Model Training**:
    *   Algorithm: **Random Forest Classifier**.
    *   The model learns patterns from the swaths of customer data to identify high-risk customers.

4.  **Model Deployment**:
    *   Framework: **Streamlit**.
    *   Features: A web-based UI that accepts customer details and outputs the churn probability.
    *   Optimization: The UI only requests the top 10 most important features (like Contract, Tenure, Charges) to simplify the user experience, while defaulting less critical ones.

## 📂 Project Structure

```
├── Customer_Churn_Prediction_Using_ML.ipynb  # Jupyter Notebook for EDA and Training
├── app.py                                    # Streamlit Application for the UI
├── customer_churn_model.pkl                  # Saved trained model
├── encoders.pkl                              # Saved label encoders
├── requirements.txt                          # Python dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Dataset
└── README.md                                 # Project Documentation
```

## 🛠️ Installation & Setup

1.  **Clone the repository** (if applicable) or download the files.

2.  **Install Dependencies**:
    Ensure you have Python installed. Run the following command to install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ How to Run

1.  **Train the Model (Optional)**:
    If you want to retrain the model or explore the data analysis, open and run the Jupyter Notebook:
    ```bash
    jupyter notebook "Customer_Churn_Prediction_Using_ML.ipynb"
    ```
    *This generates `customer_churn_model.pkl` and `encoders.pkl`.*

2.  **Run the Web Application**:
    Launch the Streamlit interface to make predictions:
    ```bash
    streamlit run app.py
    ```

3.  **Usage**:
    *   Open the URL provided in the terminal (usually `http://localhost:8501`).
    *   Enter the customer's details (Contract type, Monthly Charges, etc.).
    *   Click **Predict Churn** to see if the customer is at risk.

## 📊 Key Features Used
The application focuses on the most influential factors driving churn:
*   **Contract Type**: Month-to-month contracts have higher churn.
*   **Tenure**: New customers are more likely to leave.
*   **Charges**: Higher monthly/total charges correlate with churn.
*   **Tech Support / Online Security**: Lack of these services increases risk.