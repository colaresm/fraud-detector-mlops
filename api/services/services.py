from utils import utils
from infra import mlflow_server
import numpy as np

def get_risk(data):    
    model, scaler_model = mlflow_server.load_model_and_scaler()
    X = np.array([utils.get_params_by_prediction(data)])
    print(X)
    X_scaled = scaler_model.transform(X)
    prediction = model.predict(X_scaled)
    return prediction

