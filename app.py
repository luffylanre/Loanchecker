import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval", layout="centered")
st.title("Loan Approval Prediction")
st.write("Enter applicant details below.")

# Load the FULL pipeline (this is the simplest and most reliable way)
model = joblib.load('loan_approval_model.pkl')

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=128000, step=1000)
    loan_term = st.number_input("Loan Term (months)", min_value=12, value=360)
    credit_history = st.selectbox("Credit History (1=good)", [1.0, 0.0])

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area],
        'Total_Income': [applicant_income + coapplicant_income],
        'Income_Per_Person': [(applicant_income + coapplicant_income) / (dependents + 1)],
        'Loan_To_Income_Ratio': [loan_amount / (applicant_income + coapplicant_income + 1e-6)],
        'Loan_Amount_Per_1000': [loan_amount / 1000]
    })

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"**Loan Approved** with {prob:.1%} confidence")
    else:
        st.error(f"**Loan Rejected** (approval probability: {prob:.1%})")
