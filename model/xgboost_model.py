import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset 
df = pd.read_csv(r"C:\Users\krish\Downloads\kkbox_dataset.csv")
print("Shape:", df.shape)
print(df.head())

#Basic data checking
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

#Fill numerical missing values with median (robust to outliers) and categorical missing values with 'Unknown'
df.fillna(df.median(numeric_only=True), inplace=True)

for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna("Unknown")

print("Missing values after fill:", X.isnull().sum().sum())

#Convert categorical variables into numerical format 

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

#XGBoost model
#scale_pos_weight handles class imbalance
#learning_rate=0.05 with 300 trees gives a careful boosting approach
#max_depth=4 keeps trees shallow to avoid overfitting

from xgboost import XGBClassifier

#Calculate class imbalance ratio for scale_pos_weight
neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
print(f"Negative samples: {neg}, Positive samples: {pos}, Ratio: {neg/pos:.2f}")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    scale_pos_weight=neg / pos,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)
print("XGBoost model trained successfully.")

#Evaluating model performance using classification metrics
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

#Predict class labels and probabilities on test set
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("XGBOOST RESULTS")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

#Extract feature importances from the trained XGBoost model
#Higher importance = feature used more frequently in tree splits

feature_imp = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
top_features = feature_imp.sort_values(ascending=False)

print("Top 10 Features:")
print(top_features.head(10))

#Confusion Matrix Plot
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="Oranges")
plt.title("XGBoost Confusion Matrix")
plt.show()

#ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")
plt.legend()
plt.show()

#Top 15 Feature Importances Plot
import matplotlib.pyplot as plt

top15 = feature_imp.sort_values(ascending=False).head(15).sort_values()

plt.figure(figsize=(10, 6))
top15.plot(kind="barh", color="darkorange")
plt.xlabel("Importance Score")
plt.title("Top 15 XGBoost Feature Importances")
plt.tight_layout()
plt.show()