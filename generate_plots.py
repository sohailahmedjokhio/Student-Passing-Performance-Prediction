# Generate Plots for Student Performance Prediction
# Contributor: Sohail Ahmed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, f1_score, \
    roc_auc_score
from sql_connector import fetch_student_data
import joblib
import os
from pathlib import Path

# Ensure plots directory exists
Path('static/plots').mkdir(parents=True, exist_ok=True)

# Load Dataset from MySQL
df = fetch_student_data()
if df is None:
    print("Failed to fetch data from MySQL. Exiting.")
    exit(1)


# Data Preprocessing Functions
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
        'romantic': {'no': 0, 'yes': 1},
        'passed': {'no': 0, 'yes': 1}
    }
    df = df.copy()
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    if 'passed' in df.columns:
        col = df['passed']
        df = df.drop('passed', axis=1)
        df['passed'] = col
    return df


def feature_scaling(df):
    """Scale features to normalize data."""
    df = df.copy()
    for i in df:
        col = df[i]
        if np.max(col) > 6:
            Max = max(col)
            mean = np.mean(col)
            col = (col - mean) / Max
            df[i] = col
        elif np.max(col) < 6:
            col = (col - np.min(col))
            col /= np.max(col)
            df[i] = col
    return df


# Apply Preprocessing
df = numerical_data(df)
df = feature_scaling(df)


# Split Data
def split(df, rest_size=0.4, test_size=0.5, randomState=388628375):
    """Split dataset into train, validation, and test sets."""
    data = df.to_numpy()
    n = data.shape[1]
    x = data[:, 0:n - 1]
    y = data[:, n - 1]
    X_train, X_rest, y_train, y_rest = train_test_split(x, y, test_size=rest_size, random_state=randomState)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=test_size, random_state=randomState)
    return X_train, X_val, X_test, y_train, y_val, y_test


def optimal_C_value(X_train, y_train, X_val, y_val):
    """Find optimal C value for SVM by minimizing validation error."""
    Ci = np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 4, 10, 40, 100])
    minError = float('inf')
    optimal_C = float('inf')
    for c in Ci:
        clf = SVC(C=c, kernel='linear', probability=True)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_val)
        error = np.mean(predictions != y_val)
        if error < minError:
            minError = error
            optimal_C = c
    return optimal_C


# Train Model
model_file = 'svm_model.joblib'
X_train, X_val, X_test, y_train, y_val, y_test = split(df)

# Find optimal C value
optimal_C = optimal_C_value(X_train, y_train, X_val, y_val)
print(f"Optimal C value: {optimal_C}")

# Train final model with optimal C
clf = SVC(C=optimal_C, kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, model_file)
print(f"Model saved as {model_file}")

# Evaluate Model
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability for positive class


# Generate Feature Importance Plot
def plot_feature_importance(clf, feature_names):
    """Plot feature importance based on SVM coefficients."""
    coefficients = clf.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefficients)
    })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='Blues_d')
    plt.title('Feature Importance (SVM Coefficients)', fontsize=14)
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png', dpi=300)
    plt.close()


# Generate ROC Curve
def plot_roc_curve(y_test, y_prob):
    """Plot ROC curve and compute AUC."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/plots/roc_curve.png', dpi=300)
    plt.close()


# Generate Confusion Matrix
def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix for test set predictions."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Fail (0)', 'Pass (1)'], yticklabels=['Fail (0)', 'Pass (1)'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/plots/confusion_matrix.png', dpi=300)
    plt.close()


# Generate Plots
feature_names = df.columns[:-1]  # Exclude 'passed' column
plot_feature_importance(clf, feature_names)
plot_roc_curve(y_test, y_prob)
plot_confusion_matrix(y_test, y_pred)

# Print Performance Metrics
print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1 Score (Macro): {f1_score(y_test, y_pred, average='macro'):.2f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nPlots saved in 'static/plots':")
print("- feature_importance.png")
print("- roc_curve.png")
print("- confusion_matrix.png")