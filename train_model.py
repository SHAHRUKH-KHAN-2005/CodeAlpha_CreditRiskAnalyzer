"""Train the calibrated XGBoost model using credit_risk_dataset.csv."""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

DATA_PATH = "credit_risk_dataset.csv"
df = pd.read_csv(DATA_PATH).drop_duplicates()
df = df[(df.person_age >= 18) & (df.person_age <= 100)]
df = df[(df.person_emp_length <= df.person_age) & (df.person_emp_length <= 60)]
df = df[df.loan_amnt > 0]

X = df.drop("loan_status", axis=1)
y = df["loan_status"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
]
categorical_cols = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file"
]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

no, yes = np.bincount(y_train)
class_weight = no / yes
base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        gamma=0.5, min_child_weight=5, subsample=1.0,
        colsample_bytree=0.8, scale_pos_weight=class_weight,
        random_state=42, eval_metric="logloss", n_jobs=-1
    ))
])

calibrated_model = CalibratedClassifierCV(base_pipeline, cv=5, method="sigmoid")
calibrated_model.fit(X_train, y_train)

prob = calibrated_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, prob)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
best_threshold = float(thresholds[np.argmax(f1[:-1])])

joblib.dump(calibrated_model, "credit_risk_model.pkl")
joblib.dump(best_threshold, "best_threshold.pkl")
print(f"Saved model and threshold. Threshold: {best_threshold:.6f}")
