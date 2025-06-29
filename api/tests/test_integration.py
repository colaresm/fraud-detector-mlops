import pytest
from flask import Flask,request,jsonify
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from api.services.services import get_risk
import numpy as np

@pytest.fixture
def client():
    app = Flask(__name__)
    app.config['TESTING'] = True

    @app.route('/predict', methods=['POST'])
    def predict_route():
        data = request.get_json()
        response = get_risk(data)
        prediction, proba = response[0], response[1]
        proba_max = float(np.max(proba))
        return  jsonify({'prediction': int(prediction), 'proba_max': proba_max})
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post('/predict', json=payload)
    assert response.status_code == 200

    json_data = response.get_json()
    
    assert 'prediction' in json_data
    assert 'proba_max' in json_data
    assert isinstance(json_data['prediction'], int)
    assert isinstance(json_data['proba_max'], float)
