import os
import json
import flask
import pickle
import numpy as np

# Define paths
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model/model.pkl')
model = None

import joblib

def load_model():
    global model
    print("Loading model from: {}".format(model_path))
    model = joblib.load(model_path)
    print("Model loaded successfully")
    return model


# Flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Health check - SageMaker calls this to check if container is healthy"""
    # Check if model is loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            return flask.Response(
                response=json.dumps({"status": "unhealthy", "error": str(e)}),
                status=500,
                mimetype='application/json'
            )
    return flask.Response(
        response=json.dumps({"status": "healthy"}),
        status=200,
        mimetype='application/json'
    )

@app.route('/invocations', methods=['POST'])
def predict():
    """Prediction endpoint - SageMaker calls this for predictions"""
    # Load model if not loaded
    if model is None:
        load_model()
           
    # Parse input data
    if flask.request.content_type == 'application/json':
        try:
            data = flask.request.get_json()
            print("Received data: {}".format(data))
            
            # Expect input in format: {"features": [[val1, val2, ...]]}
            features = np.array(data['features'])
        except Exception as e:
            return flask.Response(
                response=json.dumps({"error": f"Failed to parse input: {str(e)}"}),
                status=400,
                mimetype='application/json'
            )
    else:
        return flask.Response(
            response=json.dumps({"error": "This predictor only supports JSON data"}),
            status=415,
            mimetype='application/json'
        )
           
    # Make prediction
    try:
        predictions = model.predict(features)
        result = {
            'predictions': predictions.tolist()
        }
        return flask.Response(
            response=json.dumps(result),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return flask.Response(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype='application/json'
        )

if __name__ == '__main__':
    print("Starting application server")
    load_model()
    app.run(host='0.0.0.0', port=8080)
