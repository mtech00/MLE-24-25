import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import sys
import os

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)
    
# Input/output paths
input_file = sys.argv[1]
output_dir = sys.argv[2]

# Read data
data = pd.read_csv(input_file, sep=';')

# Simple preprocessing
X = data.drop('quality', axis=1)
y = data['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['data']['test_size'],
    random_state=params['data']['random_state']
)

# Save processed data
os.makedirs(output_dir, exist_ok=True)
pd.concat([X_train, y_train], axis=1).to_csv(f"{output_dir}/train.csv", index=False)
pd.concat([X_test, y_test], axis=1).to_csv(f"{output_dir}/test.csv", index=False)

print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
