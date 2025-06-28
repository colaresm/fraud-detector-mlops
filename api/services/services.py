import mlflow
import numpy as np
from utils import utils

import mlflow.sklearn

# Configuração e carregamento uma única vez
mlflow.set_tracking_uri("http://localhost:5000")
RUN_ID = "2814d72943704714a6609abcb99fa951"

client = mlflow.client.MlflowClient()
version = max([int(i.version) for i in client.get_latest_versions("risk_model")])

# %%
model = mlflow.sklearn.load_model(f"models:/risk_model/{version}")
scaler_model = mlflow.sklearn.load_model(f"models:/scaler_risk/{version}")


def get_risk(data):

    X_novo = [[
        data["monthly_income"],
        data["credit_score"],
        data["current_debt"],
        data["late_payments"]
    ]]
    
    X_novo_scaled = scaler_model.transform(X_novo)
    prediction = model.predict(X_novo_scaled)
    print(prediction)
    return int(3)  # Retorna 0, 1 ou 2 (por exemplo)
