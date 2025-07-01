from flask import Flask, jsonify, request
from flasgger import Swagger
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Swagger config
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Iris Classifier API",
        "description": "API to classify Iris flowers and train an MLP model",
        "version": "1.0.0"
    },
    "basePath": "/"
})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the species of an Iris flower
    ---
    tags:
      - Prediction
    parameters:
      - name: input
        in: body
        required: true
        schema:
          type: object
          properties:
            sepal_length:
              type: number
            sepal_width:
              type: number
            petal_length:
              type: number
            petal_width:
              type: number
    responses:
      200:
        description: Prediction result with maximum probability
        schema:
          type: object
          properties:
            prediction:
              type: string
              example: setosa
            proba_max:
              type: number
              format: float
              example: 0.97
    """
    # Dummy prediction for now
    prediction = "setosa"
    proba_max = 0.97

    return jsonify({
        "prediction": prediction,
        "proba_max": proba_max
    })

@app.route('/train', methods=['POST'])
def train_model():
    """
    Train an MLP model with the provided parameters
    ---
    tags:
      - Training
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            hidden_layer_sizes:
              type: integer
              example: 59
              description: Number of neurons in the hidden layer(s)
            max_iter:
              type: integer
              example: 5
              description: Maximum number of iterations for training
    responses:
      200:
        description: Successfully trained the model
        schema:
          type: object
          properties:
            accuracy:
              type: number
              format: float
              example: 0.93
              description: Accuracy of the trained model on the test set
    """

    return jsonify({'accuracy': 90})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
