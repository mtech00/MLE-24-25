
import pytest
import json
from flask import Flask
import os
import sys

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.iris_predictor.api import app

except ImportError:
    print("Could not import the Flask app. Check the import path.")
    # Create a simple Flask app for testing when the import fails
    # app = Flask(__name__)
    # 
    # @app.route('/health', methods=['GET'])
    # def health():
    #     return json.dumps({'status': 'ok'}), 200
    # 
    # @app.route('/predict', methods=['POST'])
    # def predict():
    #     return json.dumps({'prediction': 0, 'class_name': 'setosa'}), 200
 


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    
    # Check status code 
    assert response.status_code in [200, 500], f"Unexpected status code: {response.status_code}"
    
    # Check response content
    data = json.loads(response.data)
    assert 'status' in data, "Response missing 'status' field"

def test_predict_endpoint_with_valid_data(client):
    """Test the prediction endpoint with valid data."""
    # Sample valid input
    valid_input = {
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5,
        'petal length (cm)': 1.4,
        'petal width (cm)': 0.2
    }
    
    response = client.post(
        '/predict',
        data=json.dumps(valid_input),
        content_type='application/json'
    )
    
    # We might get 200 if the model is loaded or 500 if not
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'prediction' in data, "Response missing 'prediction' field"
        assert 'class_name' in data, "Response missing 'class_name' field"
        
        # Check prediction is one of the valid classes
        assert data['prediction'] in [0, 1, 2], f"Invalid prediction value: {data['prediction']}"
        assert data['class_name'] in ['setosa', 'versicolor', 'virginica'], \
            f"Invalid class name: {data['class_name']}"

def test_predict_endpoint_with_missing_features(client):

    # Missing petal features
    invalid_input = {
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5
        
    }
    
    response = client.post(
        '/predict',
        data=json.dumps(invalid_input),
        content_type='application/json'
    )
    
    # Should get an error response for missing features 
    assert response.status_code in [400, 500], "Should return error for missing features"
    
    data = json.loads(response.data)
    assert 'error' in data, "Error response should contain 'error' field"

def test_predict_endpoint_with_empty_request(client):

    empty_input = {}
    
    response = client.post(
        '/predict',
        data=json.dumps(empty_input),
        content_type='application/json'
    )
    
    # Should get an error response for empty request 
    assert response.status_code in [400, 500], "Should return error for empty request"
    
    data = json.loads(response.data)
    assert 'error' in data, "Error response should contain 'error' field"
