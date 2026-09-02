import streamlit as st
import pandas as pd
import pickle

with open("credit_risk_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0d1b1e 0%, #1a2f2f 50%, #0f2419 100%);
}
.stApp, .stApp p, .stApp span, .stApp div, .stApp label {
    color: #f0f0f0 !important;
}
h1, h2, h3 {
    color: #2ecc71 !important;
}
.stButton > button {
    background-color: #2ecc71;
    color: #0d1b1e;
    font-weight: bold;
    border-radius: 8px;
    border: none;
}
.stButton > button:hover {
    background-color: #27ae60;
    color: #0d1b1e;
}
.stSelectbox > div > div, .stNumberInput > div > div {
    background-color: #ffffff !important;
    color: #000000 !important;
}
.bank-watermark {
    position: fixed;
    top: 10px;
    right: 60px;
    font-size: 180px;
    opacity: 0.06;
    z-index: 0;
    pointer-events: none;
    color: #2ecc71;
}
</style>
<div class="bank-watermark">$</div>
""", unsafe_allow_html=True)

st.title("Credit Risk Predictor")
st.write("Fill in the borrower's details below. The tool will predict whether they are likely to repay the loan (Low Risk) or struggle to repay it (High Risk).")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (in years)",
        min_value=18, max_value=100, value=30,
        help="The borrower's current age."
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"],
        help="The borrower's gender."
    )

    job_options = {
        0: "Unskilled, no fixed address",
        1: "Unskilled, has a fixed address",
        2: "Skilled worker",
        3: "Highly skilled / management",
    }
    job = st.selectbox(
        "Job Type",
        options=list(job_options.keys()),
        format_func=lambda x: job_options[x],
        help="The borrower's employment/skill level."
    )

    housing = st.selectbox(
        "Housing Situation",
        ["own", "rent", "free"],
        help="Does the borrower own their home, rent it, or live rent-free?"
    )

with col2:
    saving_accounts = st.selectbox(
        "Savings Account Balance",
        ["none", "little", "moderate", "quite rich", "rich"],
        help="How much money does the borrower have saved up?"
    )

    checking_account = st.selectbox(
        "Checking Account Balance",
        ["none", "little", "moderate", "rich"],
        help="How much easily accessible cash does the borrower keep for daily spending?"
    )

    credit_amount = st.number_input(
        "Loan Amount ($)",
        min_value=100, value=2000,
        help="How much money is the borrower asking to borrow?"
    )

    duration = st.number_input(
        "Loan Term (months)",
        min_value=1, value=12,
        help="How many months will the borrower take to repay the loan?"
    )

purpose = st.selectbox(
    "Reason for the Loan",
    ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"],
    help="What is the loan being used for?"
)

predict_button = st.button("Predict Risk")

def prepare_input(age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose):
    sex_encoded = 0 if sex == "male" else 1

    savings_order = {"none": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
    checking_order = {"none": 0, "little": 1, "moderate": 2, "rich": 3}

    saving_encoded = savings_order[saving_accounts]
    checking_encoded = checking_order[checking_account]

    row = {
        "Age": age,
        "Sex": sex_encoded,
        "Job": job,
        "Saving accounts": saving_encoded,
        "Checking account": checking_encoded,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Housing_own": 1 if housing == "own" else 0,
        "Housing_rent": 1 if housing == "rent" else 0,
        "Purpose_car": 1 if purpose == "car" else 0,
        "Purpose_domestic appliances": 1 if purpose == "domestic appliances" else 0,
        "Purpose_education": 1 if purpose == "education" else 0,
        "Purpose_furniture/equipment": 1 if purpose == "furniture/equipment" else 0,
        "Purpose_radio/TV": 1 if purpose == "radio/TV" else 0,
        "Purpose_repairs": 1 if purpose == "repairs" else 0,
        "Purpose_vacation/others": 1 if purpose == "vacation/others" else 0,
    }

    return pd.DataFrame([row])

if predict_button:
    input_df = prepare_input(age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.success(f"LOW RISK — This borrower is likely to repay the loan. (Confidence: {probability[1]*100:.1f}%)")
    else:
        st.error(f"HIGH RISK — This borrower may struggle to repay the loan. (Confidence: {probability[0]*100:.1f}%)")