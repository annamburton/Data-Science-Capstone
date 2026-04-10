import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset 
df = pd.read_csv(r"C:\Users\krish\Downloads\kkbox_dataset.csv")
print("Shape:", df.shape)
print(df.head())


#Basic data checking to understand dataset structure and quality
#Checking number of rows and columns, datatypes, etc. 
print(df.columns.tolist())
print(df["is_churn"].value_counts(dropna=False))
print(df["is_churn"].value_counts(normalize=True, dropna=False))

#Defining target variable y and feature set X
#'is_churn' is the label to be predicted
#'msno' is the user identifier

X = df.drop(columns=["is_churn", "msno"], errors="ignore").copy()
y = df["is_churn"].copy()

print("Initial X shape:", X.shape)
print("Initial y shape:", y.shape)

#Check for missing values in the dataset

print("Missing values before fill:", X.isnull().sum().sum())
print(X.isnull().sum().sort_values(ascending=False).head(20))

# Fill numerical missing values with median (robust to outliers) and categorical missing values with 'Unknown'
df.fillna(df.median(numeric_only=True), inplace=True)

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna("Unknown")

print("Missing values after fill:", X.isnull().sum().sum())

#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder

cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print("Shape after encoding:", X.shape)
X.head()

#Splitting the data into training and testing sets
#Stratify ensures class distribution is preserved
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

#n_estimators=200 builds 200 decision trees and averages their predictions
#class_weight=balanced compensates for the imbalance
#n_jobs=-1 uses all available CPU cores to speed up training

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully.")

#Evaluate model performance using classification metrics
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

#Predict class labels and probabilities on test set
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("RANDOM FOREST RESULTS")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

#Extracting feature importances to understand which variables drive churn predictions
#Higher importance = more influence on the model's decisions

feature_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_features = feature_imp.sort_values(ascending=False)

print("Top 10 Features:")
print(top_features.head(10))