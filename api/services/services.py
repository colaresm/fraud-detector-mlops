import mlflow
import numpy as np
from api.utils import utils

def get_risk(data,run_id=None):
 
    try:
        scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler_model")
        mlp = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception as e:
        raise RuntimeError(f"Error loading models from MLflow: {str(e)}")
    try:
        monthly_income, credit_score, current_debt, late_payments = utils.get_params_by_prediction(data)
    except Exception as e:
        raise ValueError(f"Error extracting parameters: {str(e)}") 
    
  #  params = [monthly_income, credit_score, current_debt, late_payments]
   # if any(x is None or not np.isfinite(x) for x in params):
    #    raise ValueError("Input data contains invalid or missing values.")
    
    X = np.array([[monthly_income, credit_score, current_debt, late_payments]])
    X_scaled = scaler.transform(X)
    prediction = mlp.predict(X_scaled)
    
    return prediction[0]
