# Students Performance Prediction
# Contributor: Sohail Ahmed

import numpy as np
import pandas as pd
import joblib
import os


def numerical_data(df):
    """Map categorical string values to numerical values."""
    mappings = {
        'school': {'GP': 0, 'MS': 1},
        'sex': {'M': 0, 'F': 1},
        'address': {'U': 0, 'R': 1},
        'famsize': {'LE3': 0, 'GT3': 1},
        'Pstatus': {'T': 0, 'A': 1},
        'Mjob': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
        'Fjob': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
        'reason': {'home': 0, 'reputation': 1, 'course': 2, 'other': 3},
        'guardian': {'mother': 0, 'father': 1, 'other': 2},
        'schoolsup': {'no': 0, 'yes': 1},
        'famsup': {'no': 0, 'yes': 1},
        'paid': {'no': 0, 'yes': 1},
        'activities': {'no': 0, 'yes': 1},
        'nursery': {'no': 0, 'yes': 1},
        'higher': {'no': 0, 'yes': 1},
        'internet': {'no': 0, 'yes': 1},
        'romantic': {'no': 0, 'yes': 1}
    }
    df = df.copy()
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    return df


def feature_scaling(df, scaler_params):
    """Scale features to normalize data."""
    df = df.copy()
    for i in df:
        col = df[i]
        if i in scaler_params['large']:
            Max, mean = scaler_params['large'][i]
            col = (col - mean) / Max
        elif i in scaler_params['small']:
            min_val, max_val = scaler_params['small'][i]
            col = (col - min_val) / max_val
        df[i] = col
    return df


def load_model_and_params(model_file='svm_model.joblib'):
    """Load the trained SVM model and scaler parameters."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found. Please train the model first.")
    model = joblib.load(model_file)

    # Load or compute scaler parameters from training data
    scaler_params = {'large': {}, 'small': {}}
    df = pd.read_csv('student-data.csv')  # Load training data for scaler params
    df = numerical_data(df)
    # Exclude 'passed' from scaler parameters
    if 'passed' in df.columns:
        df = df.drop('passed', axis=1)
    for i in df:
        col = df[i]
        if np.max(col) > 6:
            scaler_params['large'][i] = (max(col), np.mean(col))
        else:
            scaler_params['small'][i] = (np.min(col), np.max(col))
    return model, scaler_params


def predict_student(data, model, scaler_params):
    """Predict pass probability and category for new student data."""
    # Ensure only expected features are processed
    expected_features = [
        'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
        'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
        'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
    ]
    data = numerical_data(data.copy())
    # Drop any unexpected columns, including 'passed'
    data = data[[col for col in data.columns if col in expected_features]]
    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in data.columns:
            data[feature] = 0  # Default value (adjust if needed)
    data = feature_scaling(data, scaler_params)
    data_array = data[expected_features].to_numpy()  # Ensure correct feature order
    prob = model.predict_proba(data_array)[:, 1][0] * 100
    category = 'Low' if prob < 40 else 'Medium' if prob < 70 else 'High'
    return prob, category