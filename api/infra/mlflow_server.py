import mlflow.sklearn
import joblib
import os
from dotenv import load_dotenv

def load_model_and_scaler(): 
    model = joblib.load(os.getenv("MODEL"))
    scaler = joblib.load(os.getenv("SCALER_MODEL"))
    
    return model, scaler
