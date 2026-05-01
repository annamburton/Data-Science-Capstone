import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("kkbox_sample.csv")

target = "is_churn"

features = [
    "num_transactions",
    "renewal_frequency",
    "cancel_frequency",
    "avg_plan_days",
    "avg_list_price",
    "avg_amount_paid",
    "avg_subscription_length",
    "total_subscription_length",
    "discount_rate",
    "ever_cancelled",
    "total_listening_time",
    "avg_daily_usage",
    "total_unique_songs",
    "total_sessions",
    "num_log_days",
    "skip_rate",
    "completion_rate",
    "membership_duration",
    "days_since_last_transaction",
    "time_until_expiration",
    "account_age_days_x",
    "days_since_last_log",
    "city",
    "bd",
    "registered_via",
    "registration_year",
    "registration_month",
    "is_city_1"
]

df = df[features + [target]].copy()

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

medians = df[features].median()
df[features] = df[features].fillna(medians)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(
    {
        "features": features,
        "medians": medians.to_dict()
    },
    "logistic_preprocess.pkl"
)

print("SUCCESS: logistic_model.pkl, scaler.pkl, and logistic_preprocess.pkl created.")