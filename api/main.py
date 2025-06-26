import mlflow

# Configura o URI do servidor MLflow
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("fraud-detection")