
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Baseline Dashboard", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_csv(path: str):
    if not path:
        return None, "No dataset path provided."
    if not os.path.exists(path):
        return None, f"Dataset not found: {path}"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, f"Could not read dataset: {e}"

@st.cache_resource
def load_pickle(path: str):
    if not path:
        return None, "No model path provided."
    if not os.path.exists(path):
        return None, f"Model not found: {path}"
    try:
        obj = joblib.load(path)
        return obj, None
    except Exception as e:
        return None, f"Could not load model: {e}"

def normalize_bool_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bool_cols = out.select_dtypes(include=["bool"]).columns
    for c in bool_cols:
        out[c] = out[c].astype("int8")
    return out

def prepare_for_logistic(raw_row: pd.DataFrame, preprocess: dict):
    # Defensive copy
    X = raw_row.copy()

    # Drop target/id columns if present
    for col in ["is_churn", "msno"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Fill missing values the same general way as training
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median_map = preprocess.get("numeric_medians", {})
            fill_val = median_map.get(col, 0)
            X[col] = X[col].fillna(fill_val)
        else:
            X[col] = X[col].fillna("Unknown").astype(str)

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)
    X = normalize_bool_df(X)

    # Align columns to training feature space
    feature_columns = preprocess.get("feature_columns", [])
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Drop unexpected columns
    extra_cols = [c for c in X.columns if c not in feature_columns]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    if feature_columns:
        X = X[feature_columns]

    return X

def risk_band(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"

# ---------- Sidebar ----------
st.sidebar.title("Settings")
dataset_path = st.sidebar.text_input(
    "Dataset path",
    value="kkbox_dataset.csv",
    help="Path to the processed dataset used for the project."
)
model_path = st.sidebar.text_input(
    "Logistic model path",
    value="logistic_model.pkl",
    help="Path to the saved logistic regression model."
)
preprocess_path = st.sidebar.text_input(
    "Preprocess artifact path",
    value="logistic_preprocess.pkl",
    help="Path to the saved preprocessing artifact."
)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ---------- Load assets ----------
df, df_err = load_csv(dataset_path)
model, model_err = load_pickle(model_path)
preprocess, prep_err = load_pickle(preprocess_path)

# ---------- Header ----------
st.title("Customer Churn Baseline Dashboard")
st.caption("Logistic Regression baseline for early churn risk review")

if show_debug:
    st.write("Working directory:", os.getcwd())
    st.write("Files in working directory:", os.listdir())

# ---------- Status ----------
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    if df_err:
        st.error(df_err)
    else:
        st.success("Dataset loaded")
with status_col2:
    if model_err:
        st.error(model_err)
    else:
        st.success("Model loaded")
with status_col3:
    if prep_err:
        st.error(prep_err)
    else:
        st.success("Preprocess artifact loaded")

if df is None:
    st.stop()

# ---------- Overview ----------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Project overview")
    st.write(
        "This dashboard presents the logistic regression baseline for churn prediction. "
        "It is designed to demonstrate the end-to-end baseline workflow clearly and reliably."
    )
    if "is_churn" in df.columns:
        churn_rate = float(df["is_churn"].mean())
        st.metric("Observed churn rate", f"{churn_rate:.1%}")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", f"{df.shape[1]:,}")

with right:
    st.subheader("Dataset preview")
    st.dataframe(df.head(5), use_container_width=True)

# ---------- Baseline summary ----------
st.subheader("Baseline model summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", "0.93")
m2.metric("Precision (churn)", "0.95")
m3.metric("Recall (churn)", "0.24")
m4.metric("ROC-AUC", "0.83")

st.write(
    "Interpretation: the baseline model is strong at identifying non-churn users and ranks churn risk reasonably well, "
    "but it misses many actual churn cases. This makes it a useful baseline and comparison point for more complex models."
)

# ---------- Customer explorer ----------
st.subheader("Customer explorer")

available_index = df.index.tolist()
default_idx = int(available_index[0]) if available_index else 0
selected_index = st.selectbox("Select a customer row", options=available_index, index=0)

selected_row = df.loc[[selected_index]].copy()

c1, c2 = st.columns([1.15, 0.85])

with c1:
    st.write("Selected customer record")
    st.dataframe(selected_row, use_container_width=True)

with c2:
    st.write("Baseline prediction")
    if model is not None and preprocess is not None:
        try:
            X_row = prepare_for_logistic(selected_row, preprocess)
            pred = int(model.predict(X_row)[0])
            prob = float(model.predict_proba(X_row)[0][1])

            st.metric("Predicted class", "Churn" if pred == 1 else "No Churn")
            st.metric("Churn probability", f"{prob:.3f}")
            st.metric("Risk band", risk_band(prob))

            # Simple probability bar
            fig, ax = plt.subplots(figsize=(4, 2.2))
            ax.barh(["Churn probability"], [prob])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Baseline risk score")
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning("Prediction could not be generated with current artifacts.")
            if show_debug:
                st.exception(e)
    else:
        st.info(
            "Model artifacts are not loaded yet. The app layout is ready; "
            "once logistic_model.pkl and logistic_preprocess.pkl are available, "
            "this section will generate live baseline predictions."
        )

# ---------- Business interpretation ----------
st.subheader("Business interpretation")
st.write(
    "The baseline suggests that inactivity and pricing-related signals are associated with higher churn risk, "
    "while engagement and longer subscription history are associated with customer retention. "
    "As a baseline, this model is primarily valuable for interpretability and as a benchmark."
)

# ---------- Optional top coefficients ----------
st.subheader("Optional model insights")
st.write(
    "If you have a CSV export of top coefficients from the logistic regression notebook, you can place it next to this app "
    "and extend this section later. For now, the dashboard focuses on a stable baseline demonstration."
)

# ---------- Footer ----------
st.markdown("---")
st.write(
    "This dashboard is intentionally scoped to the logistic regression baseline to provide a stable, presentation-ready deployment artifact."
)
