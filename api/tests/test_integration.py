import pytest
from flask import Flask,request,jsonify
from api.services import services


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config['TESTING'] = True

    @app.route('/predict', methods=['POST'])

    def predict_route():
        data = request.get_json()
        result = services.get_risk(data)
        return jsonify({"prediction": result})
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
    assert isinstance(json_data['prediction'], (str, int, float)) 