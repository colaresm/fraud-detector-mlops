import mlflow
import numpy as np
from api.utils import utils
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.client.MlflowClient()
version = max([int(i.version) for i in client.get_latest_versions("risk_model")])
model = mlflow.sklearn.load_model(f"models:/risk_model/{version}")
scaler_model = mlflow.sklearn.load_model(f"models:/scaler_risk/{version}")


def get_risk(data):
    X = [[utils.get_params_by_prediction(data)]]
    X_scaled = scaler_model.transform(X)
    prediction = model.predict(X_scaled)
    return prediction