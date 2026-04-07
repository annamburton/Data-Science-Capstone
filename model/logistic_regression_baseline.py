# Logistic Regression baseline for KKBox churn prediction
# Converted from notebook to Python script

import pandas as pd
import numpy as np
import os

# Update this path if needed
file_path = r"C:/Users/dolly/Documents/Capstone/kkbox_sample.csv"

print("Current working directory:", os.getcwd())
print("Loading from:", file_path)

# Load the processed dataset after merging all relevant tables (train, transactions, logs, members)
# This dataset contains user behavior, subscription info, and churn labels
df = pd.read_csv(file_path)
print("Dataset loaded successfully.")
print("Shape:", df.shape)
df.head()  #Preview the dataset ot understand structure and columns

# Basic data inspection to understand dataset structure and quality
# This includes checking the number of rows/columns, data types,
# and previewing the data to identify potential issues early

print(df.columns.tolist())
print(df["is_churn"].value_counts(dropna=False))
print(df["is_churn"].value_counts(normalize=True, dropna=False))

# Define target variable (y) and feature set (X)
# 'is_churn' is the label we are trying to predict
# 'msno' is a user identifier and should not be used as a feature

X = df.drop(columns=["is_churn", "msno"], errors="ignore").copy()
y = df["is_churn"].copy()

print("Initial X shape:", X.shape)
print("Initial y shape:", y.shape)

# Check for missing values in the dataset
# Logistic Regression cannot handle NaNs, so we must clean them

print("Missing values before fill:", X.isnull().sum().sum())
print(X.isnull().sum().sort_values(ascending=False).head(20))


# Fill numerical missing values with median (robust to outliers)
# Fill categorical missing values with 'Unknown'

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna("Unknown")

print("Missing values after fill:", X.isnull().sum().sum())

# Convert categorical variables into numerical format using one-hot encoding
# This allows the model to interpret categorical data

X = pd.get_dummies(X, drop_first=True)
print("Shape after one-hot encoding:", X.shape)
X.head()

# Due to memory constraints, I use a subset of the data for model training
# This allows the model to run efficiently on local hardware

sample_size = min(5000, len(X))  # Increase later if your computer can handle it
X_sample = X.sample(n=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]

# Convert data types to float32 to reduce memory usage
# This is important due to the large number of features after encoding
X_sample = X_sample.astype("float32") # Lighter memory footprint

print("Sample X shape:", X_sample.shape)
print("Sample y shape:", y_sample.shape)
print(y_sample.value_counts(normalize=True))

# Split the data into training and testing sets
# Stratify ensures class distribution (churn vs non-churn) is preserved

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42,
    stratify=y_sample
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Train Logistic Regression model as a baseline classifier
# 'liblinear' solver works well for smaller datasets

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver="liblinear"
)

log_model.fit(X_train, y_train)
print("Logistic regression model trained successfully.")

#Evaluate model perfomance using classification metrics
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

#Predict class labels and probabilities on test set
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Extract model coefficients to understand feature impact
# Positive values increase churn likelihood, negative values decrease it
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": log_model.coef_[0]
})

#sort by absolute importance
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

coef_df.head(15)

# Visualize confusion matrix to understand prediction errors
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

#Plot ROC curve to evaluate model's ability to distinguish class 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.show()

# Extract model coefficients to understand feature impact
# Positive values increase churn likelihood, negative values decrease it

import pandas as pd
import matplotlib.pyplot as plt

coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": log_model.coef_[0]
})

# Sort by absolute importance
coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
top_coef = coef_df.sort_values("Abs_Coefficient", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_coef["Feature"], top_coef["Coefficient"])
plt.xlabel("Coefficient Value")
plt.title("Top 10 Logistic Regression Coefficients")
plt.gca().invert_yaxis()
plt.show()
