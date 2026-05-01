import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ====================== CONFIG & HEADER ======================
st.set_page_config(page_title="KKBox Churn Risk Dashboard", layout="wide", page_icon="🚨")
st.title("🚨 KKBox Early Churn Detection & Retention Dashboard")
st.markdown("**Real-time churn probability scoring for subscription retention teams**")
st.caption("**Team:** Ann-Marie Burton • Azeezat Bello • Krishna Charitha Kidambi | DSC 4350 Data Science Capstone")

# ====================== MODEL SELECTOR ======================
st.sidebar.header("🔧 Select Model")
model_option = st.sidebar.selectbox(
    "Choose your deployed model",
    ["Random Forest", "XGBoost", "Logistic Regression"]
)

# ====================== LOAD REAL MODEL + PREPROCESS ======================
@st.cache_resource
def load_artifacts(model_name):
    if model_name == "Random Forest":
        model = joblib.load("random_forest_model.pkl")
        preprocess = joblib.load("random_forest_preprocess.pkl")
    elif model_name == "XGBoost":
        model = joblib.load("xgboost_model.pkl")
        preprocess = joblib.load("xgboost_preprocess.pkl")
    else:  # Logistic Regression
        model = joblib.load("logistic_model.pkl")
        preprocess = joblib.load("logistic_preprocess.pkl")
    return model, preprocess

model, preprocess = load_artifacts(model_option)

st.sidebar.success(f"✅ Loaded {model_option} model")

# ====================== KEY BEHAVIORAL INPUTS ======================
st.sidebar.header("📊 Customer Behavior Inputs")

days_since_last = st.sidebar.number_input("Days Since Last Transaction", min_value=0, max_value=365, value=45)
num_log_days = st.sidebar.number_input("Active Log Days (last 30 days)", min_value=0, max_value=30, value=12)
avg_amount_paid = st.sidebar.number_input("Average Amount Paid ($)", min_value=0.0, max_value=100.0, value=9.99)
membership_duration = st.sidebar.number_input("Membership Duration (days)", min_value=1, max_value=2000, value=365)
renewal_frequency = st.sidebar.number_input("Renewal Frequency", min_value=0, max_value=50, value=3)
total_listening_time = st.sidebar.number_input("Total Listening Time (hours)", min_value=0.0, max_value=1000.0, value=45.5)

# ====================== PREDICT BUTTON ======================
if st.sidebar.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    # Create base input row with key features
    input_dict = {
        "days_since_last_transaction": [days_since_last],
        "num_log_days": [num_log_days],
        "avg_amount_paid": [avg_amount_paid],
        "membership_duration": [membership_duration],
        "renewal_frequency": [renewal_frequency],
        "total_listening_time": [total_listening_time]
    }
    input_df = pd.DataFrame(input_dict)

    # Apply same preprocessing logic as your deploy scripts
    if "encoder_maps" in preprocess:  # RF & XGBoost use label encoding
        for col in preprocess.get("categorical_cols", []):
            if col in input_df.columns:
                mapping = preprocess["encoder_maps"].get(col, {})
                input_df[col] = input_df[col].astype(str).map(mapping).fillna(0)

    # Fill missing columns with saved medians (or 0)
    feature_cols = preprocess.get("feature_columns", [])
    for col in feature_cols:
        if col not in input_df.columns:
            if col in preprocess.get("numeric_medians", {}):
                input_df[col] = preprocess["numeric_medians"][col]
            else:
                input_df[col] = 0

    # Reorder exactly as the model was trained
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # ====================== REAL PREDICTION ======================
    probability = model.predict_proba(input_df)[0][1]

    # Risk level + recommendation
    if probability >= 0.70:
        risk = "🔴 HIGH"
        color = "red"
        rec = "🚨 Immediate retention action needed — offer discount, call, or personalized engagement!"
    elif probability >= 0.40:
        risk = "🟠 MEDIUM"
        color = "orange"
        rec = "Consider targeted offer or loyalty campaign"
    else:
        risk = "🟢 LOW"
        color = "green"
        rec = "Customer appears loyal — monitor only"

    # ====================== DISPLAY RESULTS ======================
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("🔥 Churn Prediction")
        st.metric("Churn Probability", f"{probability:.1%}")
        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{risk} RISK</h2>", unsafe_allow_html=True)
        st.progress(probability)

    with col2:
        st.subheader("📋 Recommendation")
        st.write(rec)

    # Profile summary
    st.subheader("📊 Customer Profile")
    profile = pd.DataFrame({
        "Feature": ["Days Since Last Transaction", "Active Log Days", "Avg Amount Paid", "Membership Duration", "Renewal Frequency", "Total Listening Time"],
        "Value": [days_since_last, num_log_days, f"${avg_amount_paid:.2f}", membership_duration, renewal_frequency, f"{total_listening_time} hrs"]
    })
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.success("✅ Real model prediction complete!")

# ====================== SAMPLE AT-RISK TABLE ======================
st.subheader("📋 Sample At-Risk Customers (Ranked)")
sample_data = {
    "Customer ID": ["CUST-7842", "CUST-2910", "CUST-5631", "CUST-1098"],
    "Churn Probability": [0.87, 0.64, 0.51, 0.29],
    "Risk Level": ["HIGH", "MEDIUM", "MEDIUM", "LOW"],
    "Key Signal": ["45 days inactive", "Low log days", "Declining payments", "Stable"]
}
sample_df = pd.DataFrame(sample_data)
st.dataframe(sample_df.style.format({"Churn Probability": "{:.1%}"}), use_container_width=True)

# ====================== FOOTER ======================
