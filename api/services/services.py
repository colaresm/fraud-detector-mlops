from api.utils import utils
from api.infra import mlflow_server
from api.infra import mlflow_server
from utils import utils
from mlflow.exceptions import MlflowException

def get_risk(data):
    try:
        model, scaler_model = mlflow_server.load_model_and_scaler()
        X = [[utils.get_params_by_prediction(data)]]
        X_scaled = scaler_model.transform(X)
        prediction = model.predict(X_scaled)
        return prediction
    except (MlflowException, ConnectionError, Exception) as e:
        # Logar o erro se quiser
        print(f"[WARN] Falha ao carregar modelo do MLflow: {e}")
        return [0, 0, 0]
