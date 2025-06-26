from flask import Blueprint, jsonify

main = Blueprint('main', __name__)

@main.route('/predict')
def predict():
    return "Hello, World from routes!"


@main.route('/healthy')
def healthy():
    return jsonify({"message": "is healthy"})

    
