from api.utils import utils
from api.infra import mlflow_server
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

def get_prediction(data):    
    model, scaler_model = mlflow_server.load_model_and_scaler()
    X = np.array([utils.get_params_by_prediction(data)])
    X_scaled = scaler_model.transform(X)
    prediction = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    return prediction,proba


def train_model(data):
    try:
        mlflow.set_tracking_uri("http://mlflow_server:5000/")
        mlflow.set_experiment("Iris Classification")

        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        hidden_layer_sizes, max_iter = utils.get_params_to_train(data)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        with mlflow.start_run() as run:
            mlflow.sklearn.autolog()

            model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iter, random_state=42)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("acc_test", acc)

            mlflow.sklearn.log_model(scaler, "iris_model")
            joblib.dump(scaler, "api/models/scaler_model.pkl")
            joblib.dump(model, "api/models/mlp_model_iris.pkl")

        return True
    except Exception as e:
        print(f"Error on train model: {e}")
        return False