# Student Passing Performance Prediction

## Overview
The Student Passing Performance Prediction project is an academic initiative designed to predict whether a student will pass their final exam based on a variety of personal, academic, and lifestyle factors. Utilizing a Support Vector Machine (SVM) with a linear kernel, the system provides a probability score and categorizes the likelihood of passing into **Low (<40%)**, **Medium (40-70%)**, or **High (>70%)**. Built with Flask, the web interface allows users to input 30 features and receive real-time predictions. The project leverages the UCI Student Performance Dataset and integrates Python libraries such as scikit-learn, pandas, and matplotlib for modeling and visualization.

### One-Line Description
This project uses an SVM model to predict student exam success based on personal, academic, and lifestyle factors, offering actionable insights for educators.

### Objectives
- Assist educators and students in identifying at-risk students early.
- Enable targeted interventions to improve academic outcomes.
- Provide a user-friendly web interface for predictions using machine learning.

## Features
- **Prediction Model**: SVM with linear kernel for percentage + category (low < 40%, medium (40-70%), high > 70%).
- **Web Interface**: Flask-based form to input 30 student features.
- **Real-Time Results**: Probability scores and category assignments (Low, Medium, High).
- **Data Visualization**: Plots for feature importance, ROC curves, and confusion matrices.
- **Dataset**: UCI Student Performance Dataset with 395 students and 31 columns.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- Internet connection (for installing dependencies)

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/sohailahmed/student-passing-performance-prediction.git
   cd student-performance-prediction