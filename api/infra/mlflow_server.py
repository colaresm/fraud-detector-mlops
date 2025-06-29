import mlflow.sklearn
import joblib

def load_model_and_scaler():
   
    model = joblib.load("api/models/mlp_model_iris.pkl")
    scaler = joblib.load("api/models/scaler_model.pkl")
    
    return model, scaler
