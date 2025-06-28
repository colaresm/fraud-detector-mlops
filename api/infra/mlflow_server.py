import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def load_model_and_scaler(model_name="risk_model", scaler_name="scaler_risk", tracking_uri="http://localhost:5000"):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient()
    
    version = max([int(i.version) for i in client.get_latest_versions(model_name)])
    
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    scaler = mlflow.sklearn.load_model(f"models:/{scaler_name}/{version}")
    
    return model, scaler

def is_mlflow_online(tracking_uri):
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        client.list_experiments()
        return True
    except Exception as e:
        print(f"Erro ao conectar ao MLflow: {e}")
        return False

if is_mlflow_online("http://localhost:5000"):
    print("MLflow está online.")
else:
    print("MLflow está offline ou inacessível.")
