from flask import Flask,jsonify
from api.routes.routes import main
from flasgger import Swagger

app = Flask(__name__)
app.register_blueprint(main)

swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Iris Classifier API",
        "description": "API to classify Iris flowers using ML",
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
    prediction = "setosa"
    proba_max = 0.97

    return jsonify({
        "prediction": prediction,
        "proba_max": proba_max
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)



