import joblib
import numpy as np

def load_model_and_scaler():
    model = joblib.load("api/models/mlp_model_iris.pkl")
    scaler = joblib.load("api/models/scaler_model.pkl")
    return model, scaler

def predict_iris(input_data):
    model, scaler = load_model_and_scaler()
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return prediction[0]

example = [5.1, 3.5, 1.4, 0.2]
result = predict_iris(example)
print(f"Prediction: {result}")
