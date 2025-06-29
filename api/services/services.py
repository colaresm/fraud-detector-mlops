from api.utils import utils
from api.infra import mlflow_server
import numpy as np

def get_prediction(data):    
    model, scaler_model = mlflow_server.load_model_and_scaler()
    X = np.array([utils.get_params_by_prediction(data)])
    X_scaled = scaler_model.transform(X)
    prediction = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    return prediction,proba

