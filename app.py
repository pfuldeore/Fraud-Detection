import streamlit as st
import pandas as pd
import pickle
import os

from src.features.build_features import add_application_date_features, add_income_ratios, add_location_features, generate_transaction_aggregation
from src.utils.helper import calculate_credit_score, get_credit_label

# -------- Load the best fraud model --------
@st.cache_data
def load_model_bundle():
    with open("outputs/models/new_best_model_bundle.pkl", "rb") as f:
        return pickle.load(f)

model_bundle = load_model_bundle()
fraud_model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]
feature_columns = model_bundle["feature_columns"]


# -------- Input Form --------
st.title("🔍 Loan Fraud Detection App")

with st.form("fraud_form"):
    st.subheader("Enter Loan Application Details")
    application_date = st.date_input("Application Date")
    loan_type = st.selectbox("Loan Type", ["Personal Loan", "Home Loan", "Auto Loan"])
    loan_amount_requested = st.number_input("Loan Amount Requested", min_value=1000.0, value=50000.0)
    loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1, value=60, step=1)
    interest_rate_offered = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5, step=0.1)
    purpose_of_loan = st.selectbox("Purpose of Loan", ['Medical Emergency', 'Education', 'Debt Consolidation','Business Expansion', 'Wedding', 'Vehicle Purchase','Home Renovation'], index=2)
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed"], index=0)
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=70000.0)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, step=1)
    existing_emis_monthly = st.number_input("Existing EMIs", min_value=0.0, value=10000.0)
    debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, value=14.28)
    property_ownership_status = st.selectbox("Property Ownership", ["Owned", "Rented", "Mortgaged"], index=1)
    residential_address = st.text_input("Address", value="123 Main St, Pune, Maharashtra")
    applicant_age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=2, step=1)
    customer_id = st.text_input("Customer ID", value="C12345")
    application_id = st.text_input("Application ID", value="A1001")
    uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

    submitted = st.form_submit_button("Predict Fraud")

# -------- Prediction Pipeline --------
if submitted:
    if uploaded_file is None:
        st.error("❗ Please upload a transaction CSV file.")
    else:
        # Step 1: Input dict
        input_dict = {
            "application_id": application_id,
            "customer_id": customer_id,
            "application_date": str(application_date),
            "loan_type": loan_type,
            "loan_amount_requested": loan_amount_requested,
            "loan_tenure_months": loan_tenure_months,
            "interest_rate_offered": interest_rate_offered,
            "purpose_of_loan": purpose_of_loan,
            "employment_status": employment_status,
            "monthly_income": monthly_income,
            "cibil_score": cibil_score,
            "existing_emis_monthly": existing_emis_monthly,
            "debt_to_income_ratio": debt_to_income_ratio,
            "property_ownership_status": property_ownership_status,
            "residential_address": residential_address,
            "applicant_age": applicant_age,
            "gender": gender,
            "number_of_dependents": number_of_dependents,
        }

        df = pd.DataFrame([input_dict])

        # Step 2: Feature Engineering
        df = add_application_date_features(df)
        df = add_income_ratios(df)
        df = add_location_features(df)

        # Step 3: Add transaction features
        transaction_df = pd.read_csv(uploaded_file)
        transaction_features = generate_transaction_aggregation(df, transaction_df)
        df = df.merge(transaction_features, on="application_id", how="left")

        # Step 4: Drop unwanted columns
        drop_cols = [
            'application_date', 'application_id', 'customer_id',
            'loan_status', 'fraud_flag', 'fraud_type', 'residential_address'
        ]
        df = df.drop(columns=drop_cols, errors='ignore')

        # Step 5: Handle missing columns efficiently
        missing_cols = [col for col in feature_columns if col not in df.columns]
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)
        df = df.reindex(columns=preprocessor.feature_names_in_, fill_value=0)

        # Step 6: Predict
        X_transformed = preprocessor.transform(df)
        fraud_flag = fraud_model.predict(X_transformed)[0]
        fraud_proba = fraud_model.predict_proba(X_transformed)[0][1]

        # Step 7: Output
        st.subheader("Prediction Result")
        st.write(f"Fraud Prediction: {'⚠️ Fraudulent' if fraud_flag == 1 else '✅ Not Fraudulent'}")
        st.write(f"Fraud Probability: {fraud_proba:.2%}")

        if fraud_flag == 1:
            st.subheader("❌ Application flagged as fraudulent.")
        else:
            st.subheader("✅ Loan Application is clean.")
            credit_score = calculate_credit_score(df.iloc[0])
            credit_label = get_credit_label(credit_score)

            st.markdown(f"### 🧮 Creditworthiness Score: {credit_score} / 100")
            st.markdown(f"### 💡 Creditworthiness Rating: **{credit_label}**")
    
    
