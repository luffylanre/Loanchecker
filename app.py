import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="Loan Approval", layout="centered")
st.title("Loan Approval Prediction")
st.write("Enter applicant details below.")

# Rebuild preprocessor (tiny and version-safe)
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Credit_History', 'Total_Income', 
                    'Income_Per_Person', 'Loan_To_Income_Ratio', 'Loan_Amount_Per_1000']
categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
passthrough_features = ['Dependents']

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                          ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features),
                  ('pass', 'passthrough', passthrough_features)],
    remainder='drop'
)

# Load only the classifier
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

if st.button("🔍 Predict", type="primary"):
    input_df = pd.DataFrame({
        'Gender': [gender], 'Married': [married], 'Dependents': [dependents],
        'Education': [education], 'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income], 'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount], 'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history], 'Property_Area': [property_area],
        'Total_Income': [applicant_income + coapplicant_income],
        'Income_Per_Person': [(applicant_income + coapplicant_income) / (dependents + 1)],
        'Loan_To_Income_Ratio': [loan_amount / (applicant_income + coapplicant_income + 1e-6)],
        'Loan_Amount_Per_1000': [loan_amount / 1000]
    })

    X_input = preprocessor.fit_transform(input_df)   # rebuild on the fly
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.success(f"**Loan Approved** with {prob:.1%} confidence")
    else:
        st.error(f"**Loan Rejected** (approval probability: {prob:.1%})")
