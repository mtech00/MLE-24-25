import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys 
from datetime import datetime


# Default paths relative to the project root directory
# These are primarily used if the script is run directly without arguments
# The wrapper script (`run_batch_packaged.sh`) will override these with its own paths]
date_stamp = datetime.now().strftime("%Y%m%d")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'artifacts')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')

# these are will be overrided by shell scripts
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, f'iris_model_{date_stamp}.pkl')
DEFAULT_SCALER_PATH = os.path.join(ARTIFACTS_DIR, f'iris_scaler_{date_stamp}.pkl')
DEFAULT_INPUT_PATH = os.path.join(DATA_DIR, 'input_batch_iris.csv')
DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, f'predictions_batch_iris_{date_stamp}.csv')

INPUT_FEATURE_NAMES = [
    'sepal length (cm)', 'sepal width (cm)',
    'petal length (cm)', 'petal width (cm)'
]
PREDICTION_INDEX_COLUMN = 'prediction'
PREDICTION_NAME_COLUMN = 'prediction_class_name'
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

EPSILON = 1e-6 # For safe division

# Core Processing Function

def process_batch(input_path, model_path, scaler_path):
    """Loads data, artifacts, preprocesses, scales, and predicts."""
    # Load Artifacts
    try:
        print(f"BATCH: Loading model from: {model_path}")
        model = joblib.load(model_path)
        print(f"BATCH: Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print("BATCH: Artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"BATCH Error: Could not load artifacts. File not found.")
        print(f"BATCH Error details: {e}")
        print(f"BATCH Check paths: Model='{model_path}', Scaler='{scaler_path}'")
        raise # Stop execution if artifacts can't be loaded
    except Exception as e:
        print(f"BATCH Error loading artifacts: {e}")
        raise

    # Load Data
    try:
        print(f"BATCH: Loading input data from: {input_path}")
        df_input = pd.read_csv(input_path)
        if df_input.empty:
            print(f"BATCH Warning: Input file '{input_path}' is empty.")
            # Return empty DataFrame and empty predictions
            return pd.DataFrame(columns=INPUT_FEATURE_NAMES), np.array([])

        # Validate and select columns
        missing_cols = [col for col in INPUT_FEATURE_NAMES if col not in df_input.columns]
        if missing_cols:
             raise ValueError(f"Input CSV missing required columns: {missing_cols}")
        df_input = df_input[INPUT_FEATURE_NAMES] # only necessary columns
        print(f"BATCH: Input data loaded successfully ({len(df_input)} rows).")

    except FileNotFoundError:
        print(f"BATCH Error: Input file not found at '{input_path}'")
        raise
    except Exception as e:
        print(f"BATCH Error loading input data: {e}")
        raise # Stop execution if data loading fails

    # Preprocess Data 
    print("BATCH: Starting preprocessing...")
    df_processed = df_input.copy()
    df_processed['sepal_ratio'] = df_processed['sepal length (cm)'] / (df_processed['sepal width (cm)'] + EPSILON)
    df_processed['petal_ratio'] = df_processed['petal length (cm)'] / (df_processed['petal width (cm)'] + EPSILON)

    # Outlier flag 
    numeric_cols = INPUT_FEATURE_NAMES + ['sepal_ratio', 'petal_ratio']
    # Handle case where std is zero
    std_devs = df_processed[numeric_cols].std()
    std_devs[std_devs == 0] = EPSILON # Replace zero std with epsilon
    z_scores = np.abs((df_processed[numeric_cols] - df_processed[numeric_cols].mean()) / std_devs)
    df_processed['is_outlier'] = (z_scores > 3).any(axis=1).astype(int)
    print(f"BATCH: Outlier flags calculated ({df_processed['is_outlier'].sum()} marked).")

    # correct column order for scaler
    final_feature_order = INPUT_FEATURE_NAMES + ['sepal_ratio', 'petal_ratio', 'is_outlier']
    df_processed = df_processed[final_feature_order]
    print("BATCH: Preprocessing complete.")

    # Scale and Predict
    try:
        print("BATCH: Scaling data...")
        scaled_data = scaler.transform(df_processed.values)
        print("BATCH: Making predictions...")
        predictions = model.predict(scaled_data)
        print("BATCH: Scaling and prediction complete.")
        # Return the original input data and the predictions
        return df_input, predictions
    except Exception as e:
        print(f"BATCH Error during scaling/prediction: {e}")
        raise # Stop execution on prediction error



def save_results(df_original_input, predictions, output_path):
    """Saves the original input data along with predictions."""
    if df_original_input.empty:
        print("BATCH: No data to save (input was empty).")
        # Optionally create an empty file         
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"BATCH: Created output directory: {output_dir}")
            # Define columns for the empty output file
            output_columns = list(df_original_input.columns) + [PREDICTION_INDEX_COLUMN, PREDICTION_NAME_COLUMN]
            pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)
            print(f"BATCH: Empty output file created at: {output_path}")
        except Exception as e:
            print(f"BATCH Error creating empty output file: {e}")
        return

    print(f"BATCH: Preparing output data...")
    df_output = df_original_input.copy()
    df_output[PREDICTION_INDEX_COLUMN] = predictions
    df_output[PREDICTION_NAME_COLUMN] = df_output[PREDICTION_INDEX_COLUMN].apply(
        lambda x: CLASS_NAMES[x] if pd.notna(x) and 0 <= x < len(CLASS_NAMES) else None
    )

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)
             print(f"BATCH: Created output directory: {output_dir}")
        # Save the results
        df_output.to_csv(output_path, index=False)
        print(f"BATCH: Results successfully saved to: {output_path}")
    except Exception as e:
        print(f"BATCH Error saving results: {e}")
        raise # Stop execution if saving fails

def main():
    """Main function to parse arguments and run the batch prediction."""
    parser = argparse.ArgumentParser(description="Batch prediction script for Iris model (packaged).")
    # Arguments now use the updated default paths relative to project root
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to model file (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument('--scaler', type=str, default=DEFAULT_SCALER_PATH,
                        help=f"Path to scaler file (default: {DEFAULT_SCALER_PATH})")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_PATH,
                        help=f"Path to input CSV (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f"Path for output CSV (default: {DEFAULT_OUTPUT_PATH})")
    args = parser.parse_args()

    print("--- Starting Iris Batch Prediction (via package entry point) ---")
    # Log the actual paths being used
    print(f"Using Input Path: {args.input}")
    print(f"Using Output Path: {args.output}")
    print(f"Using Model Path: {args.model}")
    print(f"Using Scaler Path: {args.scaler}")

    try:
        # Process the batch data using the specified or default paths
        df_original, predictions = process_batch(args.input, args.model, args.scaler)

        # Save the results using the specified or default output path
        save_results(df_original, predictions, args.output)

        print("--- Batch Prediction Finished Successfully ---")

    except Exception as e:
        # Catch any exceptions raised during processing or saving
        print(f"--- Batch Prediction Failed ---", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) # Exit with a non-zero status code indicating failure


if __name__ == "__main__":
    # This allows the script to be run directly
    main()
