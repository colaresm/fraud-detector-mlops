import mlflow.sklearn
import os

def load_model_and_scaler(models_dir: str = "models"):
    model_path = os.path.join(models_dir, "model")
    scaler_path = os.path.join(models_dir, "scaler_model")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    model = mlflow.sklearn.load_model(model_path)
    scaler = mlflow.sklearn.load_model(scaler_path)
    
    return model, scaler
