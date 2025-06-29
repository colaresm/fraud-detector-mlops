from api.utils import utils
from api.infra import mlflow_server
import pandas as pd
def get_risk(data):    
    model, scaler_model = mlflow_server.load_model_and_scaler()
    X = pd.DataFrame([utils.get_params_by_prediction(data)])
    X_scaled = scaler_model.transform(X)
    prediction = model.predict(X_scaled)
    return prediction

