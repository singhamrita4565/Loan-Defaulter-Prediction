
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Defaulter Prediction", page_icon="💳")

@st.cache_resource

def load():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load()


st.title("💳 Loan Defaulter Prediction")
st.write("Enter customer details and click Predict.")

age = st.number_input("Age",18,100,30)
income = st.number_input("Income",0,10000000,50000)
loan_amount = st.number_input("Loan Amount",0,10000000,100000)
credit_score = st.number_input("Credit Score",300,900,650)
employment_years = st.number_input("Employment Years",0,50,5)

education = st.selectbox(
    "Education Level",
    ["High School","Bachelors","Masters","PhD"]
)

housing = st.selectbox(
    "Housing Status",
    ["Mortgage","Own","Rent"]
)

edu_map = {
    "High School":0,
    "Bachelors":1,
    "Masters":2,
    "PhD":3
}

own = 1 if housing=="Own" else 0
rent = 1 if housing=="Rent" else 0

row = {
    "Age": age,
    "Income": income,
    "Loan_Amount": loan_amount,
    "Credit_Score": credit_score,
    "Employment_Years": employment_years,
    "Education_Level": edu_map[education],
    "Own": own,
    "Rent": rent
}

input_df = pd.DataFrame([row])

# ensure order
input_df = input_df.reindex(columns=list(features))
if st.button("Predict"):

    # Probability of Default
    prob = model.predict_proba(input_df)[0][1]

    st.progress(float(prob))
    st.write(f"Default Probability: **{prob*100:.2f}%**")

    # Threshold = 30%
    if prob >= 0.30:
        st.error("⚠️ High Risk: Customer is likely to DEFAULT.")
    else:
        st.success("✅ Low Risk: Customer is NOT likely to default.")

    if prob is not None:



        st.progress(float(prob))
        st.write(f"Default Probability: **{prob*100:.2f}%**")

with st.expander("Model Information"):
    st.write("Algorithm: Random Forest")