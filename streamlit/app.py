import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="KKBox Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
preprocess = joblib.load("logistic_preprocess.pkl")

features = preprocess["features"]
medians = preprocess["medians"]

st.markdown("""
<style>
.stButton > button {
    width: 100%;
    height: 4rem;
    font-size: 1.35rem;
    font-weight: 700;
    border-radius: 14px;
    background-color: #2563eb;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

.result-box {
    font-size: 2rem;
    font-weight: 800;
    padding: 1.5rem;
    border-radius: 18px;
    text-align: center;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 KKBox Customer Churn Prediction Dashboard")

st.write(
    "This dashboard predicts churn risk for KKBox users based on subscription behavior, "
    "listening activity, renewal patterns, and account history."
)

st.divider()

# -----------------------------
# PROFILE BUILDER
# -----------------------------
def make_profile(overrides):
    profile = {feature: float(medians.get(feature, 0)) for feature in features}
    profile.update(overrides)
    return profile

profiles = {
    "Low Risk Customer": make_profile({
        "days_since_last_transaction": 3,
        "days_since_last_log": 2,
        "renewal_frequency": 3,
        "cancel_frequency": 0,
        "avg_subscription_length": 30,
        "avg_amount_paid": 149,
        "total_listening_time": 6500,
        "avg_daily_usage": 450,
        "skip_rate": 0.18,
        "completion_rate": 0.75,
        "membership_duration": 700,
        "account_age_days_x": 900,
        "num_transactions": 8,
        "num_log_days": 24
    }),

    "Medium Risk Customer": make_profile({
        "days_since_last_transaction": 18,
        "days_since_last_log": 10,
        "renewal_frequency": 1,
        "cancel_frequency": 0,
        "avg_subscription_length": 25,
        "avg_amount_paid": 120,
        "total_listening_time": 3000,
        "avg_daily_usage": 220,
        "skip_rate": 0.35,
        "completion_rate": 0.55,
        "membership_duration": 300,
        "account_age_days_x": 500,
        "num_transactions": 4,
        "num_log_days": 12
    }),

    "High Risk Customer": make_profile({
        "days_since_last_transaction": 45,
        "days_since_last_log": 28,
        "renewal_frequency": 0,
        "cancel_frequency": 1,
        "avg_subscription_length": 15,
        "avg_amount_paid": 80,
        "total_listening_time": 1200,
        "avg_daily_usage": 90,
        "skip_rate": 0.55,
        "completion_rate": 0.35,
        "membership_duration": 120,
        "account_age_days_x": 250,
        "num_transactions": 2,
        "num_log_days": 6
    })
}

priority_features = [
    "days_since_last_transaction",
    "days_since_last_log",
    "renewal_frequency",
    "cancel_frequency",
    "avg_subscription_length",
    "avg_amount_paid",
    "total_listening_time",
    "avg_daily_usage",
    "skip_rate",
    "completion_rate",
    "membership_duration",
    "account_age_days_x",
    "num_transactions",
    "num_log_days"
]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🎛️ Customer Scenario")

selected_profile = st.sidebar.selectbox(
    "Choose a demo profile",
    list(profiles.keys())
)

st.sidebar.caption("Adjust values below if you want to test different customer behaviors.")

user_input = profiles[selected_profile].copy()

st.sidebar.subheader("Main Churn Signals")

for feature in priority_features:
    if feature in features:
        user_input[feature] = st.sidebar.number_input(
            feature.replace("_", " ").title(),
            value=float(user_input[feature])
        )

# -----------------------------
# MAIN LAYOUT
# -----------------------------
left, right = st.columns([1.1, 1])

with left:
    st.subheader("👤 Customer Input Summary")

    input_summary = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in priority_features if f in user_input],
        "Value": [user_input[f] for f in priority_features if f in user_input]
    })

    st.dataframe(input_summary, use_container_width=True, height=430)

with right:
    st.subheader("🚨 Prediction Panel")

    st.write("Click below to generate a churn probability and risk classification.")

    predict_button = st.button("🔮 Predict Customer Churn Risk")

    if predict_button:
        input_df = pd.DataFrame([user_input])

        for col in features:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
            input_df[col] = input_df[col].fillna(medians[col])

        input_scaled = scaler.transform(input_df[features])

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        retention_probability = 1 - probability

        if probability >= 0.70:
            risk_level = "High Risk"
            bg = "#fee2e2"
            color = "#991b1b"
        elif probability >= 0.40:
            risk_level = "Medium Risk"
            bg = "#fef3c7"
            color = "#92400e"
        else:
            risk_level = "Low Risk"
            bg = "#dcfce7"
            color = "#166534"

        st.markdown(
            f"""
            <div class="result-box" style="background-color:{bg}; color:{color};">
                {risk_level}<br>{probability:.2%} Churn Probability
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric("Model Prediction", "Churn" if prediction == 1 else "Retained")
        st.metric("Churn Probability", f"{probability:.2%}")
        st.metric("Retention Probability", f"{retention_probability:.2%}")

        st.subheader("📈 Probability View")

        st.write("Churn likelihood:")
        st.progress(float(probability))

        st.write("Retention likelihood:")
        st.progress(float(retention_probability))

        prob_table = pd.DataFrame({
            "Outcome": ["Churn", "Retention"],
            "Probability": [f"{probability:.2%}", f"{retention_probability:.2%}"]
        })

        st.table(prob_table)

        st.subheader("📊 Key Signal Snapshot")

        chart_features = [
            "days_since_last_transaction",
            "days_since_last_log",
            "renewal_frequency",
            "cancel_frequency",
            "skip_rate",
            "completion_rate"
        ]

        chart_df = pd.DataFrame({
            "Signal": [f.replace("_", " ").title() for f in chart_features],
            "Value": [user_input[f] for f in chart_features]
        })

        st.dataframe(chart_df, use_container_width=True)

st.divider()

st.subheader("🧠 How to Explain This During the Demo")

st.write(
    "Use the dropdown to switch between low, medium, and high-risk customer examples. "
    "The model then turns the customer’s behavior into a churn probability. The key patterns "
    "to point out are inactivity, fewer renewals, cancellation history, lower listening time, "
    "higher skip rate, and lower completion rate."
)