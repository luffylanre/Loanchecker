
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load('loan_approval_model.pkl')

st.title("Loan Approval Prediction System")
st.write("Enter applicant details to check loan eligibility.")

# Input form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=9, value=128)
    loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=12, value=360)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create input dataframe
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Add engineered features
input_data['Total_Income'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
input_data['Income_Per_Person'] = input_data['Total_Income'] / (input_data['Dependents'] + 1)
input_data['Loan_To_Income_Ratio'] = input_data['LoanAmount'] / input_data['Total_Income']
input_data['Loan_Amount_Per_1000'] = input_data['LoanAmount'] / 1000

if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {probability:.1%})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {1-probability:.1%})")
    
    st.write(f"Approval Probability: {probability:.1%}")
    
    # Simple risk summary
    if probability > 0.8:
        st.info("Low Risk Application")
    elif probability > 0.6:
        st.warning("Moderate Risk - Consider additional review")
    else:
        st.error("High Risk Application")
