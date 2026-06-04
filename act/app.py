import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le = joblib.load('models/label_encoder.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

st.title("Financial Approval Agent")
st.subheader("Credit Card Application Review")

st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=99, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=125, value=3)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000, step=500)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=35.0, value=11.0)
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

col3, col4 = st.columns(2)

with col3:
    loan_percent_income = st.slider("Loan as % of Income", 0.0, 1.0, 0.2)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=30, value=4)
    person_gender = st.selectbox("Gender", ["male", "female"])

with col4:
    previous_loan_defaults_on_file = st.selectbox("Prior Default on File", ["No", "Yes"])
    person_education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])

if st.button("Evaluate Application"):

    debt_to_income = loan_amnt / (person_income + 1)
    income_to_loan_ratio = person_income / (loan_amnt + 1)
    credit_to_age_ratio = cb_person_cred_hist_length / (person_age + 1)

    input_dict = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'income_to_loan_ratio': income_to_loan_ratio,
        'credit_to_age_ratio': credit_to_age_ratio,
        f'person_home_ownership_{person_home_ownership}': 1,
        f'loan_intent_{loan_intent}': 1,
        f'previous_loan_defaults_on_file_{previous_loan_defaults_on_file}': 1,
        f'person_gender_{person_gender}': 1,
        f'person_education_{person_education}': 1,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)

    proba = model.predict_proba(input_scaled)[0]
    prediction = model.predict(input_scaled)[0]
    decision = le.inverse_transform([prediction])[0]

    approved_idx = list(le.classes_).index('Approved')
    confidence = proba[approved_idx] * 100

    st.divider()
    st.header("Agent Decision")

    st.metric("Confidence of Repayment", f"{confidence:.1f}%")
    st.progress(float(confidence) / 100)

    if decision == 'Approved':
        st.success("Credit Line Auto-Approved")
        st.write("The applicant meets all credit criteria. Application has been approved automatically.")
    elif decision == 'Review':
        st.warning("Application Sent for Manual Review")
        st.write("The applicant's profile requires further review by a credit officer.")
    else:
        st.error("Application Denied")
        st.write("The applicant does not meet the minimum credit requirements.")

    st.divider()
    st.subheader("Probability Breakdown")
    prob_df = pd.DataFrame({
        'Decision': le.classes_,
        'Probability': [f"{p*100:.1f}%" for p in proba]
    })
    st.table(prob_df)