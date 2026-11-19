import os
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1180px}
.hero{padding:1.6rem 1.8rem;border-radius:18px;background:linear-gradient(120deg,#101d35,#1d4260);color:white;margin-bottom:1.2rem}
.hero h1{color:white;margin:0;font-size:2rem}
.hero p{color:#d7e5f2;margin:.45rem 0 0}
.metric-card{padding:1rem 1.2rem;border:1px solid rgba(128,128,128,.25);border-radius:14px}
.result-approved{padding:1.2rem;border-radius:14px;background:#e8f7ef;color:#14532d;border:1px solid #86d3a5}
.result-risk{padding:1.2rem;border-radius:14px;background:#fff0ed;color:#8a2418;border:1px solid #f0a49a}
.small-muted{opacity:.72;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model_path = os.getenv("MODEL_PATH", "credit_risk_model.pkl")
    threshold_path = os.getenv("THRESHOLD_PATH", "best_threshold.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Run train_model.py first.")
    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path)) if os.path.exists(threshold_path) else 0.5
    return model, threshold

st.markdown("""
<div class="hero">
  <h1>🏦 Credit Risk Analyzer</h1>
  <p>AI-powered loan default risk assessment · XGBoost with probability calibration</p>
</div>
""", unsafe_allow_html=True)

try:
    model, threshold = load_assets()
except Exception as e:
    st.error(str(e))
    st.info("To prepare the model, place credit_risk_dataset.csv in this folder and run: python train_model.py")
    st.stop()

with st.sidebar:
    st.header("About this model")
    st.write("This application uses the calibrated XGBoost pipeline from the Credit Risk Project notebook.")
    st.metric("Decision threshold", f"{threshold:.3f}")
    st.caption("The threshold is loaded from best_threshold.pkl.")
    st.divider()
    st.caption("For educational / decision-support use only. Do not use as the sole basis for lending decisions.")

st.subheader("Applicant information")
st.caption("Enter the applicant's details. The model will estimate the probability of loan default.")

with st.form("applicant_form"):
    a, b, c = st.columns(3)
    with a:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=28, step=1)
        income = st.number_input("Annual income", min_value=0, value=60000, step=1000)
        employment = st.number_input("Employment length (years)", min_value=0.0, max_value=60.0, value=4.0, step=0.5)
        home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    with b:
        intent = st.selectbox("Loan intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])
        amount = st.number_input("Loan amount", min_value=1, value=10000, step=500)
        rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=100.0, value=11.5, step=0.1)
    with c:
        history = st.number_input("Credit history length (years)", min_value=0, max_value=60, value=5, step=1)
        default_file = st.selectbox("Previous default on file?", ["N", "Y"])
        st.markdown("<div class='small-muted'>Loan-to-income ratio is calculated automatically.</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze credit risk", type="primary", use_container_width=True)

if submitted:
    if employment > age:
        st.error("Employment length cannot be greater than the applicant's age.")
        st.stop()
    if income <= 0:
        st.error("Annual income must be greater than zero.")
        st.stop()

    loan_ratio = amount / income
    applicant = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": employment,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": amount,
        "loan_int_rate": rate,
        "loan_percent_income": loan_ratio,
        "cb_person_default_on_file": default_file,
        "cb_person_cred_hist_length": history
    }])

    probability = float(model.predict_proba(applicant)[0, 1])
    decision = int(probability >= threshold)
    st.divider()
    st.subheader("Risk assessment")
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Estimated default probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Decision threshold: {threshold:.1%}")
    with right:
        if decision == 1:
            st.markdown(f"""<div class="result-risk"><h3>⚠️ Higher default risk</h3>
            <p>The estimated probability of default is <b>{probability:.1%}</b>, which is at or above the model threshold.</p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-approved"><h3>✓ Lower default risk</h3>
            <p>The estimated probability of default is <b>{probability:.1%}</b>, which is below the model threshold.</p></div>""", unsafe_allow_html=True)
    with st.expander("View submitted applicant data"):
        display = applicant.T.rename(columns={0: "Applicant value"})
        st.dataframe(display, use_container_width=True)
    st.caption("Model output is a statistical estimate, not a guarantee of repayment or default.")

st.divider()
st.markdown("<div class='small-muted'>Credit Risk Analyzer · Built with Streamlit and scikit-learn / XGBoost</div>", unsafe_allow_html=True)
