import pandas as pd
import numpy as np
import yaml
import json
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load parameters
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Arguments
train_file = sys.argv[1]
test_file = sys.argv[2]
metrics_file = sys.argv[3]

# Load data
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

X_train = train.drop('quality', axis=1)
y_train = train['quality']
X_test = test.drop('quality', axis=1)
y_test = test['quality']

# Select model
if params['model']['type'] == 'linear':
    model = LinearRegression(
        fit_intercept=params['model']['linear']['fit_intercept']
    )
elif params['model']['type'] == 'forest':
    model = RandomForestRegressor(
        n_estimators=params['model']['forest']['n_estimators'],
        max_depth=params['model']['forest']['max_depth'],
        random_state=params['model']['random_state']
    )
else:
    raise ValueError(f"Unknown model type: {params['model']['type']}")

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save metrics
os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
with open(metrics_file, 'w') as f:
    json.dump({'rmse': rmse, 'r2': r2}, f, indent=4)


print(f"Training completed. Model: {params['model']['type']}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
