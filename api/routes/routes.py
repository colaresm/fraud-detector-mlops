from flask import Blueprint, jsonify,request
from api.services import services
import numpy as np
main = Blueprint('main', __name__)

@main.route('/predict',methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    response = services.get_prediction(data)

    prediction, proba = response[0], response[1]
    proba_max = float(np.max(proba))

    return jsonify({'prediction': int(prediction), 'proba_max': proba_max}), 200
@main.route('/healthy')
def healthy():
    return jsonify({"message": "is healthy"})

    
