#!/bin/bash


# IMPORTANT! Set the ABSOLUTE path to your virtual environment's activation script
VENV_PATH="$HOME/model_deployment/venv/bin/activate" # UPDATE THIS 

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


set -e # Exit immediately if a command exits with a non-zero status

# Activate Virtual Environment

echo "Attempting to activate venv at ${VENV_PATH}"

source "${VENV_PATH}"
if [ $? -ne 0 ]; then
  echo "Error: Failed to activate virtual environment at ${VENV_PATH}. Check the path."
  exit 1
fi

echo "Virtual environment activated."

DATE_TAG=$(date +%Y%m%d)

# Define File Paths Add timestamp to output
INPUT_FILE="${PROJECT_DIR}/data/input_batch_iris.csv"
OUTPUT_FILE="${PROJECT_DIR}/data/predictions_batch_${DATE_TAG}.csv" 
MODEL_FILE="${PROJECT_DIR}/artifacts/iris_model_${DATE_TAG}.pkl"
SCALER_FILE="${PROJECT_DIR}/artifacts/iris_scaler_${DATE_TAG}.pkl"
 # simple tag by date
DOCKER_TAG="iris-predictor-api:${DATE_TAG}"

echo "Starting batch run at $(date)"
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Model: ${MODEL_FILE}"
echo "Scaler: ${SCALER_FILE}"

# 'run-iris-batch' comes from pyproject.toml
run-iris-batch \
  --input "${INPUT_FILE}" \
  --output "${OUTPUT_FILE}" \
  --model "${MODEL_FILE}" \
  --scaler "${SCALER_FILE}"

echo "Batch prediction script completed successfully."

# Build Docker Image
echo "Building Docker image..."
# Navigate to the directory containing the Dockerfile 
cd "${PROJECT_DIR}"

# Build the Docker image
docker build -t "${DOCKER_TAG}" .

echo "Docker image built successfully: ${DOCKER_TAG}"

# Optional: Push image to a registry (e.g., Docker Hub)
# echo "Pushing Docker image..."
# docker push "${DOCKER_TAG}" # ensure you are logged in (docker login)

echo "Batch run and Docker build finished successfully at $(date)"

# Deactivate the virtual environment
deactivate

exit 0 # Explicitly exit with success code
