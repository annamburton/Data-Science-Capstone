import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

#Loading the dataset 
df = pd.read_csv("kkbox_dataset.csv")
print(df.head())
print(df.info())

#Encoding categorical features
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

#Defining features and target
X = df.drop(columns=['is_churn'])
y = df['is_churn']

#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

#Fitting model
rf_model.fit(X_train, y_train)

#Predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]

#Evaluate
print("RANDOM FOREST RESULTS")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

#Feature Importance
feature_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns)
print("\nTop 10 Features:")
print(feature_imp.sort_values(ascending=False).head(10))

