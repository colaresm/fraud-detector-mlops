import mlflow.sklearn
import joblib
import os
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path)

def load_model_and_scaler(): 
    model = joblib.load(os.getenv("MODEL"))
    scaler = joblib.load(os.getenv("SCALER_MODEL"))
    
    return model, scaler
