from flask import Blueprint, jsonify,request
from api.services import services


main = Blueprint('main', __name__)

@main.route('/predict')
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()

    risk = services.get_risk(data)

    return jsonify({'risk':risk,}), 200

@main.route('/healthy')
def healthy():
    return jsonify({"message": "is healthy"})

    
