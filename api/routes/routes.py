from flask import Blueprint, jsonify,request
from services import services

main = Blueprint('main', __name__)

@main.route('/predict',methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    prediction = services.get_risk(data)[0]
    
    return jsonify({'prediction':int(prediction)}), 200

@main.route('/healthy')
def healthy():
    return jsonify({"message": "is healthy"})

    
