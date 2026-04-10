import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn XGBoost Dashboard", layout="wide")

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

def prepare_for_xgb(raw_row: pd.DataFrame, preprocess: dict):
    X = raw_row.copy()

    for col in ["is_churn", "msno"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median_map = preprocess.get("numeric_medians", {})
            fill_val = median_map.get(col, 0)
            X[col] = X[col].fillna(fill_val)
        else:
            X[col] = X[col].fillna("Unknown").astype(str)

    X = pd.get_dummies(X, drop_first=True)
    X = normalize_bool_df(X)

    feature_columns = preprocess.get("feature_columns", [])
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    extra_cols = [c for c in X.columns if c not in feature_columns]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    if feature_columns:
        X = X[feature_columns]

    # XGBoost accepts float arrays — cast to avoid dtype warnings
    X = X.astype(float)

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
    "XGBoost model path",
    value="xgb_model.pkl",
    help="Path to the saved XGBoost model."
)
preprocess_path = st.sidebar.text_input(
    "Preprocess artifact path",
    value="xgb_preprocess.pkl",
    help="Path to the saved preprocessing artifact."
)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ---------- Load assets ----------
df, df_err = load_csv(dataset_path)
model, model_err = load_pickle(model_path)
preprocess, prep_err = load_pickle(preprocess_path)

# ---------- Header ----------
st.title("Customer Churn XGBoost Dashboard")
st.caption("XGBoost gradient boosting model for churn risk prediction")

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
        "This dashboard presents the XGBoost gradient boosting model for churn prediction. "
        "XGBoost builds trees sequentially, generally yielding the strongest predictive performance in the churn pipeline."
    )
    if "is_churn" in df.columns:
        churn_rate = float(df["is_churn"].mean())
        st.metric("Observed churn rate", f"{churn_rate:.1%}")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", f"{df.shape[1]:,}")

with right:
    st.subheader("Dataset preview")
    st.dataframe(df.head(5), use_container_width=True)

# ---------- Model summary ----------
st.subheader("XGBoost model summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", "—")
m2.metric("Precision (churn)", "—")
m3.metric("Recall (churn)", "—")
m4.metric("ROC-AUC", "—")

st.write(
    "Replace the metric placeholders above with your evaluated scores from the notebook. "
    "XGBoost typically achieves the highest ROC-AUC in the pipeline by leveraging boosting "
    "to reduce both bias and variance compared to a single tree or bagging ensemble."
)

# ---------- Customer explorer ----------
st.subheader("Customer explorer")

available_index = df.index.tolist()
selected_index = st.selectbox("Select a customer row", options=available_index, index=0)

selected_row = df.loc[[selected_index]].copy()

c1, c2 = st.columns([1.15, 0.85])

with c1:
    st.write("Selected customer record")
    st.dataframe(selected_row, use_container_width=True)

with c2:
    st.write("XGBoost prediction")
    if model is not None and preprocess is not None:
        try:
            X_row = prepare_for_xgb(selected_row, preprocess)
            pred = int(model.predict(X_row)[0])
            prob = float(model.predict_proba(X_row)[0][1])

            st.metric("Predicted class", "Churn" if pred == 1 else "No Churn")
            st.metric("Churn probability", f"{prob:.3f}")
            st.metric("Risk band", risk_band(prob))

            fig, ax = plt.subplots(figsize=(4, 2.2))
            ax.barh(["Churn probability"], [prob], color="darkorange")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("XGBoost risk score")
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning("Prediction could not be generated with current artifacts.")
            if show_debug:
                st.exception(e)
    else:
        st.info(
            "Model artifacts are not loaded yet. Once xgb_model.pkl and xgb_preprocess.pkl "
            "are available, this section will generate live predictions."
        )

# ---------- Feature importances ----------
st.subheader("Feature importances")

if model is not None and preprocess is not None:
    try:
        feature_columns = preprocess.get("feature_columns", [])

        #XGBoost exposes importances via feature_importances_ (sklearn API)
        importances = model.feature_importances_

        if len(feature_columns) == len(importances):
            fi_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(20)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="darkorange")
            ax2.set_xlabel("Importance (gain)")
            ax2.set_title("Top 20 feature importances (XGBoost)")
            plt.tight_layout()
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("Feature column count does not match importances array. Check your preprocess artifact.")
    except Exception as e:
        st.warning("Could not render feature importances.")
        if show_debug:
            st.exception(e)
else:
    st.info(
        "Load the model and preprocess artifacts to view feature importances."
    )

# ---------- Business interpretation ----------
st.subheader("Business interpretation")
st.write(
    "XGBoost's gradient boosting approach captures subtle, high-order interactions between "
    "engagement, pricing, and subscription renewal signals. The feature importance chart uses "
    "gain-based importance, reflecting how much each feature reduces prediction error across "
    "all trees. Comparing this chart with the Random Forest importances can reveal which signals "
    "are robustly predictive versus model-specific."
)

# ---------- Footer ----------
st.markdown("---")
st.write(
    "XGBoost dashboard — part of the KKBox churn prediction project."
)
