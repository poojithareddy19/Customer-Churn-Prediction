import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to predict if they are likely to churn.")

# Load model and encoders
@st.cache_resource
def load_data():
    with open('customer_churn_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
        
    return model_data, encoders

try:
    model_data, encoders = load_data()
    model = model_data['model']
    feature_names = model_data['features_names']
except FileNotFoundError:
    st.error("Model or Encoder files not found. Please ensure 'customer_churn_model.pkl' and 'encoders.pkl' are in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input form
with st.form("churn_prediction_form"):
    st.subheader("Customer Details")
    
    col1, col2 = st.columns(2)

    with col1:
        # High importance features
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1)
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
        total_charges = st.number_input("Total Charges", min_value=0.0, format="%.2f")
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    submit_button = st.form_submit_button("Predict Churn")

if submit_button:
    # Prepare input data dictionary with user inputs and defaults for hidden columns
    # Defaults based on mode (most frequent value) or low feature importance analysis
    input_data = {
        'gender': 'Male',             # Default/Hidden (Low Importance)
        'SeniorCitizen': 0,           # Default/Hidden (Low Importance)
        'Partner': 'No',              # Default/Hidden (Low Importance)
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': 'Yes',        # Default/Hidden (Low Importance)
        'MultipleLines': 'No',        # Default/Hidden (Low Importance)
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': 'No',     # Default/Hidden (Low Importance)
        'TechSupport': tech_support,
        'StreamingTV': 'No',          # Default/Hidden (Low Importance)
        'StreamingMovies': 'No',      # Default/Hidden (Low Importance)
        'Contract': contract,
        'PaperlessBilling': 'Yes',    # Default/Hidden (Low Importance)
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocessing
    try:
        # Encode categorical variables using loaded encoders
        for col, encoder in encoders.items():
            if col in input_df.columns:
                # Handle unknown labels if necessary, though selectboxes constrain inputs to known values usually
                # To be safe, we map values, but selectboxes match Training data unique values
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Ensure column order matches training
        input_df = input_df[feature_names]
        
        # Prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        churn_prob = prediction_proba[0][1]
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"High Risk of Churn! (Probability: {churn_prob:.2%})")
        else:
            st.success(f"Low Risk of Churn. (Probability: {churn_prob:.2%})")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Debug info: Feature names expected:", feature_names)
        st.write("Input columns:", input_df.columns.tolist())
