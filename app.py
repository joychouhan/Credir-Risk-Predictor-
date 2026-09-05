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
    background-color: #f7f9fb;
}
h1 {
    color: #1a365d !important;
    font-weight: 700;
}
h2, h3 {
    color: #1a365d !important;
}
.subtitle {
    color: #5a6b7d;
    font-size: 16px;
    margin-top: -10px;
    margin-bottom: 30px;
}
.stButton > button {
    background-color: #1a365d;
    color: #ffffff;
    font-weight: 600;
    border-radius: 6px;
    border: none;
    padding: 8px 24px;
}
.stButton > button:hover {
    background-color: #2c4a75;
    color: #ffffff;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px;
}
.footer-note {
    color: #8a97a6;
    font-size: 13px;
    margin-top: 40px;
    border-top: 1px solid #e0e4e8;
    padding-top: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("Credit Risk Predictor")
st.markdown(
    '<p class="subtitle">A machine learning tool that predicts loan repayment risk based on borrower profile data.</p>',
    unsafe_allow_html=True,
)

st.caption("Model: Gradient Boosting Classifier | Test accuracy: 77% | Trained on 1,000 historical loan records")

with st.container(border=True):
    st.subheader("Borrower Details")

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
            "Loan Amount (₹)",
            min_value=1000, value=150000,
            help="How much money is the borrower asking to borrow?"
        )

        duration = st.number_input(
            "Loan Term (months)",
            min_value=1, value=12,
            help="How many months will the borrower take to repay the loan?"
        )

    interest_rate = st.slider(
        "Assumed Annual Interest Rate (%)",
        min_value=5.0, max_value=20.0, value=10.0, step=0.5,
        help="The dataset doesn't include actual interest rates, so this is an assumed rate used only to estimate the EMI breakdown."
    )

    purpose = st.selectbox(
        "Reason for the Loan",
        ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"],
        help="What is the loan being used for?"
    )

    predict_button = st.button("Predict Risk")

if credit_amount > 1500000:
    st.warning("This loan amount is much higher than what the model was trained on. The prediction below may not be reliable for such a large loan.")

def calculate_emi_breakdown(principal, annual_rate, months):
    """
    Calculates EMI using the standard loan amortization formula,
    then splits total repayment into Principal vs Interest.
    """
    monthly_rate = (annual_rate / 100) / 12

    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)

    total_payment = emi * months
    total_interest = total_payment - principal

    return emi, principal, total_interest, total_payment

def prepare_input(age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose):
    sex_encoded = 0 if sex == "male" else 1

    savings_order = {"none": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
    checking_order = {"none": 0, "little": 1, "moderate": 2, "rich": 3}

    saving_encoded = savings_order[saving_accounts]
    checking_encoded = checking_order[checking_account]

    monthly_payment_simple = credit_amount / duration

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
        "Monthly_Payment": monthly_payment_simple,
    }

    return pd.DataFrame([row])

if predict_button:
    input_df = prepare_input(age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    emi, principal, total_interest, total_payment = calculate_emi_breakdown(credit_amount, interest_rate, duration)

    st.subheader("Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric("Estimated Monthly EMI", f"₹{emi:,.2f}")

    with result_col2:
        if prediction == 1:
            st.success(f"LOW RISK — Likely to repay (Confidence: {probability[1]*100:.1f}%)")
        else:
            st.error(f"HIGH RISK — May struggle to repay (Confidence: {probability[0]*100:.1f}%)")

    st.write("**Principal vs Interest Breakdown**")

    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    with breakdown_col1:
        st.metric("Principal Amount", f"₹{principal:,.2f}")
    with breakdown_col2:
        st.metric("Total Interest", f"₹{total_interest:,.2f}")
    with breakdown_col3:
        st.metric("Total Repayment", f"₹{total_payment:,.2f}")

    breakdown_chart = pd.DataFrame({
        "Component": ["Principal", "Interest"],
        "Amount (₹)": [principal, total_interest],
    })
    st.bar_chart(breakdown_chart.set_index("Component"), color="#1a365d")

    st.write("**Risk Probability Breakdown**")
    chart_data = pd.DataFrame({
        "Outcome": ["Low Risk (Good)", "High Risk (Bad)"],
        "Probability (%)": [probability[1] * 100, probability[0] * 100],
    })
    st.bar_chart(chart_data.set_index("Outcome"), color="#1a365d")

st.markdown(
    '<p class="footer-note">Built with Python, scikit-learn, and Streamlit — as part of a FinTech portfolio project.</p>',
    unsafe_allow_html=True,
)
