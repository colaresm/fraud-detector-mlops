import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import mlflow


# Integration with MLflow
mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment(experiment_id=522610720329323041) # insert experiment id

# 1. Generate synthetic data
np.random.seed(42)
n_samples = 1000

monthly_income = np.random.normal(loc=5000, scale=1500, size=n_samples).clip(800, None)
credit_score = np.random.randint(300, 1001, size=n_samples)
current_debt = np.random.normal(loc=10000, scale=5000, size=n_samples).clip(0, None)
late_payments = np.random.randint(0, 11, size=n_samples)

# 2. Risk score calculation function
def calculate_risk(income, score, debt, delays):
    risk_score = (debt / income) * 0.4 + (10 - score / 1000 * 10) * 0.3 + delays * 0.3
    if risk_score < 4:
        return "low"
    elif risk_score < 7:
        return "medium"
    else:
        return "high"

# 3. Apply function to generate labels
loan_risk = [
    calculate_risk(i, s, d, l)
    for i, s, d, l in zip(monthly_income, credit_score, current_debt, late_payments)
]

# 4. Create DataFrame
df = pd.DataFrame({
    "monthly_income": monthly_income,
    "credit_score": credit_score,
    "current_debt": current_debt,
    "late_payments": late_payments,
    "loan_risk": loan_risk
})

# 5. Prepare X and y
X = df.drop("loan_risk", axis=1)
y = df["loan_risk"]

# 6. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2,
)

# 8. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Start experiment on MLflow
with mlflow.start_run():
    mlflow.sklearn.autolog()

    mlp = MLPClassifier(hidden_layer_sizes=(20), max_iter=400, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    y_pred = mlp.predict(X_test_scaled)
    acc_test = accuracy_score(y_test, y_pred)
    mlflow.log_metrics({"acc_test": acc_test})

    mlflow.sklearn.log_model(scaler, "scaler_model")
    
    
