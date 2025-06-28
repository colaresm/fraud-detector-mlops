import mlflow
import mlflow.sklearn

def load_model_and_scaler(model_name="risk_model", scaler_name="scaler_risk", tracking_uri="http://localhost:5000"):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient()
    
    version = max([int(i.version) for i in client.get_latest_versions(model_name)])
    
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    scaler = mlflow.sklearn.load_model(f"models:/{scaler_name}/{version}")
    
    return model, scaler
