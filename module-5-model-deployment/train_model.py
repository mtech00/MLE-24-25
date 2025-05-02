import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from datetime import datetime

date_stamp = datetime.now().strftime("%Y%m%d")

os.makedirs('artifacts', exist_ok=True)

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame 
df = pd.DataFrame(X, columns=iris.feature_names)

# Basic preprocessing
def preprocess_data(df):

    df_processed = df
    
    # Check for missing values
    if df_processed.isnull().sum().any():
        print("Found missing values, will be imputed")
    
    # Create feature interactions
    df_processed['sepal_ratio'] = df_processed['sepal length (cm)'] / df_processed['sepal width (cm)']
    df_processed['petal_ratio'] = df_processed['petal length (cm)'] / df_processed['petal width (cm)']
    
    # outliers flag rather than remove 
    z_scores = np.abs((df_processed - df_processed.mean()) / df_processed.std())
    outliers = (z_scores > 3).any(axis=1)
    if outliers.any():
        print(f"Found {outliers.sum()} outliers, marking with feature")
        df_processed['is_outlier'] = outliers.astype(int)
    
    return df_processed


df_processed = preprocess_data(df)
X = df_processed.values
y = iris.target 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Save model and scaler 
joblib.dump(model, f'artifacts/iris_model_{date_stamp}.pkl')
joblib.dump(scaler, f'artifacts/iris_scaler_{date_stamp}.pkl')
print("Model and preprocessing pipeline saved to disk")

# test of loading and using the model
loaded_model = joblib.load(f'artifacts/iris_model_{date_stamp}.pkl')
loaded_scaler = joblib.load(f'artifacts/iris_scaler_{date_stamp}.pkl')

# Test with sample data
sample = X_test[0].reshape(1, -1)
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)
print(f"Sample prediction: {prediction} ({iris.target_names[prediction[0]]})")
