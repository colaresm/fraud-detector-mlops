import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

# MLflow config
mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("Loan Risk Prediction")

# Gerar dados sintéticos
np.random.seed(42)
n_samples = 1000
monthly_income = np.random.normal(loc=5000, scale=1500, size=n_samples).clip(800, None)
credit_score = np.random.randint(300, 1001, size=n_samples)
current_debt = np.random.normal(loc=10000, scale=5000, size=n_samples).clip(0, None)
late_payments = np.random.randint(0, 11, size=n_samples)

# Função para calcular risco
def calculate_risk(income, score, debt, delays):
    risk_score = (debt / income) * 0.4 + (10 - score / 1000 * 10) * 0.3 + delays * 0.3
    if risk_score < 4:
        return "low"
    elif risk_score < 7:
        return "medium"
    else:
        return "high"

loan_risk = [
    calculate_risk(i, s, d, l)
    for i, s, d, l in zip(monthly_income, credit_score, current_debt, late_payments)
]

# Criar DataFrame
df = pd.DataFrame({
    "monthly_income": monthly_income,
    "credit_score": credit_score,
    "current_debt": current_debt,
    "late_payments": late_payments,
    "loan_risk": loan_risk
})

# Separar X e y
X = df.drop("loan_risk", axis=1)
y = LabelEncoder().fit_transform(df["loan_risk"])

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Iniciar experimento MLflow
with mlflow.start_run() as run:
    mlflow.sklearn.autolog()

    model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=400, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("acc_test", acc)

    # Logar o scaler de duas formas
    mlflow.sklearn.log_model(scaler, "scaler_model")
    joblib.dump(scaler, "scaler_model.pkl")
    mlflow.log_artifact("scaler_model.pkl")

    print("Modelo e scaler salvos com sucesso no MLflow!")
    print("Run ID:", run.info.run_id)
