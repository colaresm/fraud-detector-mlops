from api.utils import utils
from api.infra import mlflow_server

def get_risk(data):
    has_connection = mlflow_server.is_mlflow_online()
    if has_connection:
        model, scaler_model = mlflow_server.load_model_and_scaler()
        X = [[utils.get_params_by_prediction(data)]]
        X_scaled = scaler_model.transform(X)
        prediction = model.predict(X_scaled)
        return prediction
    else:
        return 0
