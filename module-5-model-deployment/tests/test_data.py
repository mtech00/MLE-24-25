import pytest
import pandas as pd
import os
import numpy as np

# Create test data directory if it doesn't exist
os.makedirs('tests/test_data', exist_ok=True)

def create_test_data():
    """Create test data files if they don't exist."""
    # Create sample input data
    input_data = pd.DataFrame({
        'sepal length (cm)': [5.1, 6.2, 7.3, 4.9, 5.8],
        'sepal width (cm)': [3.5, 2.8, 2.9, 3.1, 4.0],
        'petal length (cm)': [1.4, 4.7, 6.3, 1.5, 1.2],
        'petal width (cm)': [0.2, 1.3, 1.8, 0.1, 0.2]
    })
    
    # Save to CSV
    input_path = 'tests/test_data/iris_test_input.csv'
    input_data.to_csv(input_path, index=False)
    
    return input_path

def test_input_data_has_required_columns():
    """Check if input data has all required columns."""
    input_path = create_test_data()
    df = pd.read_csv(input_path)
    
    required_columns = [
        'sepal length (cm)', 'sepal width (cm)',
        'petal length (cm)', 'petal width (cm)'
    ]
    
    # Check all required columns are present
    for column in required_columns:
        assert column in df.columns, f"Missing required column: {column}"
    
    # Check data types all should be numeric
    for column in required_columns:
        assert pd.api.types.is_numeric_dtype(df[column]), f"Column {column} should be numeric"

def test_data_values_are_in_expected_range():
    """Check if data values are within expected ranges."""
    input_path = create_test_data()
    df = pd.read_csv(input_path)
    
    # Simple range checks
    assert df['sepal length (cm)'].min() >= 4.0, "Sepal length too small"
    assert df['sepal length (cm)'].max() <= 8.0, "Sepal length too large"
    
    assert df['sepal width (cm)'].min() >= 2.0, "Sepal width too small"
    assert df['sepal width (cm)'].max() <= 4.5, "Sepal width too large"
    
    assert df['petal length (cm)'].min() >= 1.0, "Petal length too small"
    assert df['petal length (cm)'].max() <= 7.0, "Petal length too large"
    
    assert df['petal width (cm)'].min() >= 0.1, "Petal width too small"
    assert df['petal width (cm)'].max() <= 2.5, "Petal width too large"

def test_ratio_calculations():
    """Test simple ratio calculations."""
    # Create test data
    data = pd.DataFrame({
        'sepal length (cm)': [5.0],
        'sepal width (cm)': [2.0],
        'petal length (cm)': [3.0],
        'petal width (cm)': [1.0]
    })
    
    # Calculate ratios
    data['sepal_ratio'] = data['sepal length (cm)'] / data['sepal width (cm)']
    data['petal_ratio'] = data['petal length (cm)'] / data['petal width (cm)']
    
    # Check calculations
    assert data['sepal_ratio'].iloc[0] == 2.5, "Sepal ratio calculation error"
    assert data['petal_ratio'].iloc[0] == 3.0, "Petal ratio calculation error"
