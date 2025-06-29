import mlflow.sklearn
import joblib

def load_model_and_scaler(models_dir: str = "models"):
   
    model = joblib.load("models/mlp_model_iris.pkl")
    scaler = joblib.load("models/scaler_model.pkl")
    
    return model, scaler
