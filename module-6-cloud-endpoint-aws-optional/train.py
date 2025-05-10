import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame

# Basic preprocessing


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
# Train model
model = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Save model and scaler 
joblib.dump(model, f'model.pkl')
print("Model and preprocessing pipeline saved to disk")

# test of loading and using the model
loaded_model = joblib.load(f'model.pkl')


# Test with sample data
sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)
print(f"Sample prediction: {prediction} ({iris.target_names[prediction[0]]})")
