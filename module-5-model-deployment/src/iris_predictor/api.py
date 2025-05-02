from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any
import os 
import sys 
from datetime import datetime



# __file__ is the path to the current script (api.py)
# os.path.dirname gets the directory of the script (src/iris_predictor)
# os.path.abspath ensures we have an absolute path

date_stamp = datetime.now().strftime("%Y%m%d")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Navigate up from src/iris_predictor to model_deployment, then into artifacts
    ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'artifacts')
    MODEL_PATH = os.path.join(ARTIFACTS_DIR, f'iris_model_{date_stamp}.pkl')
    SCALER_PATH = os.path.join(ARTIFACTS_DIR, f'iris_scaler_{date_stamp}.pkl')
except NameError:
     # Handle cases where __file__ might not be defined 
     # Fallback to relative paths assuming execution from project root
     print("Warning: __file__ not defined, using relative paths for artifacts.")
     MODEL_PATH = f'artifacts/iris_model_{date_stamp}.pkl'
     SCALER_PATH = f'artifacts/iris_scaler_{date_stamp}.pkl'


# Iris feature names for input validation
FEATURE_NAMES = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

# Flask App Initialization
app = Flask(__name__)

# Load Model and Scaler
# Load artifacts during app initialization for efficiency
try:
    print(f"API: Attempting to load model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"API: Attempting to load scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print(f"API: Successfully loaded model and scaler.")
except FileNotFoundError as e:
    print(f"API Error: Could not load model/scaler. File not found at expected path.")
    print(f"Error details: {e}")
    print("Ensure artifacts exist in the '/artifacts' directory relative to the project root.")
    model = None
    scaler = None
except Exception as e:
    print(f"API Error: An unexpected error occurred loading artifacts: {e}")
    model = None
    scaler = None

# for safe division if needed in preprocessing
EPSILON = 1e-6
# Preprocessing Function 
def preprocess_api_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocesses input dictionary for the API.
    Returns a DataFrame ready for scaling.
    """
    # Convert input dictionary to a DataFrame 
    df = pd.DataFrame([data], columns=FEATURE_NAMES)

    # Calculate ratios
    df['sepal_ratio'] = df['sepal length (cm)'] / (df['sepal width (cm)'] + EPSILON)
    df['petal_ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + EPSILON)


    df['is_outlier'] = 0

    # columns are in the same order as expected by the scaler
    final_feature_order = FEATURE_NAMES + ['sepal_ratio', 'petal_ratio', 'is_outlier']
    # Reindex to ensure order and handle potential missing columns gracefully 
    df_processed = df.reindex(columns=final_feature_order)

    return df_processed

    
def validate_input(data: Dict[str, Any]) -> bool:
    """Validate input data"""
    try:
        # Check all features present
        if not all(feature in data for feature in FEATURE_NAMES):
            return False
        
        # Check numeric values
        if not all(isinstance(data[feature], (int, float)) for feature in FEATURE_NAMES):
            return False
        
        # Check positive values
        if not all(data[feature] > 0 for feature in FEATURE_NAMES):
            return False
            
        return True
    except:
        return False

        
# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    # Check if artifacts were loaded successfully on startup

    start_time = datetime.now()

    
    if model is None or scaler is None:
         print("Error: Predict endpoint called but model/scaler not loaded.")
         return jsonify({'error': 'Model or scaler failed to load during server initialization.'}), 500

    # Validate input

    try:
        # Get JSON data from POST request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided in JSON format.'}), 400

        # Validate that all required feature names are present in the input data

        if not validate_input(data):

            return jsonify({'error': 'Invalid input data format'}), 400    
            
        # Preprocess the input data 
        print(f"API: Received data for prediction: {data}")
        df_processed = preprocess_api_data(data)
        print(f"API: Preprocessed data: \n{df_processed.to_string()}")

        # Scale the preprocessed data using the loaded scaler
        scaled_data = scaler.transform(df_processed.values)
        print(f"API: Scaled data: {scaled_data}")

        # Make prediction using the loaded model
        prediction_idx = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        print(f"API: Raw prediction index: {prediction_idx}")

        # Format the prediction result
        result = {
            'prediction': int(prediction_idx[0]),
            'class_name': CLASS_NAMES[prediction_idx[0]],
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        print(f"API: Sending prediction result: {result}")
        return jsonify(result)

    except ValueError as e: # Catch errors during preprocessing/scaling/prediction
         print(f"API Prediction ValueError: {e}")
         # more specific error if possible 
         return jsonify({'error': f'Error during prediction processing: {e}'}), 400
    except Exception as e:
        # full exception for debugging server-side
        print(f"API Prediction Unexpected Exception: {e}", file=sys.stderr)
        # Return a generic error message to the client
        return jsonify({'error': f'An unexpected server error occurred.'}), 500

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify service status and artifact loading."""
    if model is not None and scaler is not None:
        return jsonify({'status': 'ok', 'message': 'API is running and artifacts are loaded.'}), 200
    else:
        # Be specific about the error state
        return jsonify({'status': 'error', 'message': 'API is running BUT model/scaler artifacts failed to load.'}), 500

def main():
    app.run(host="127.0.0.1", port=5000)





#  Run Flask App Only for Direct Execution
if __name__ == '__main__':
    # debug=True enables auto-reloading and provides detailed error pages 
    print("Running Flask app directly (for development/testing)...")
    app.run(debug=True, host='127.0.0.1', port=5000) 

