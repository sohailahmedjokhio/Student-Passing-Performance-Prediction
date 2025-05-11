from flask import Flask, render_template, request
import pandas as pd
from svm_linear_kernel import load_model_and_params, predict_student

app = Flask(__name__)

# Load model and scaler parameters
try:
    model, scaler_params = load_model_and_params()
except FileNotFoundError as e:
    print(e)
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        # Collect form data with default values for missing inputs
        student_data = {
            'school': request.form.get('school', 'GP'),  # Default to 'GP'
            'gender': request.form.get('gender', 'M'),   # Default to 'M'
            'age': int(request.form.get('age', 15)),     # Default to 15
            'address': request.form.get('address', 'U'), # Default to 'U'
            'famsize': request.form.get('famsize', 'LE3'), # Default to 'LE3'
            'Pstatus': request.form.get('Pstatus', 'T'), # Default to 'T'
            'Medu': int(request.form.get('Medu', 0)),    # Default to 0
            'Fedu': int(request.form.get('Fedu', 0)),    # Default to 0
            'Mjob': request.form.get('Mjob', 'other'),   # Default to 'other'
            'Fjob': request.form.get('Fjob', 'other'),   # Default to 'other'
            'reason': request.form.get('reason', 'other'), # Default to 'other'
            'guardian': request.form.get('guardian', 'other'), # Default to 'other'
            'traveltime': int(request.form.get('traveltime', 1)), # Default to 1
            'studytime': int(request.form.get('studytime', 1)), # Default to 1
            'failures': int(request.form.get('failures', 0)), # Default to 0
            'schoolsup': request.form.get('schoolsup', 'no'), # Default to 'no'
            'famsup': request.form.get('famsup', 'no'),   # Default to 'no'
            'paid': request.form.get('paid', 'no'),       # Default to 'no'
            'activities': request.form.get('activities', 'no'), # Default to 'no'
            'nursery': request.form.get('nursery', 'no'), # Default to 'no'
            'higher': request.form.get('higher', 'no'),   # Default to 'no'
            'internet': request.form.get('internet', 'no'), # Default to 'no'
            'romantic': request.form.get('romantic', 'no'), # Default to 'no'
            'famrel': int(request.form.get('famrel', 3)), # Default to 3
            'freetime': int(request.form.get('freetime', 3)), # Default to 3
            'goout': int(request.form.get('goout', 3)),   # Default to 3
            'Dalc': int(request.form.get('Dalc', 1)),     # Default to 1
            'Walc': int(request.form.get('Walc', 1)),     # Default to 1
            'health': int(request.form.get('health', 3)), # Default to 3
            'absences': int(request.form.get('absences', 0)) # Default to 0
        }
        # Convert to DataFrame
        new_student = pd.DataFrame([student_data])
        try:
            probability, category = predict_student(new_student, model, scaler_params)
            prediction = {
                'probability': f"{probability:.2f}%",
                'category': category
            }
        except Exception as e:
            prediction = {'error': str(e)}
    return render_template('index.html', prediction=prediction)

@app.route('/project-info')
def project_info():
    return render_template('project_info.html')

if __name__ == '__main__':
    app.run(debug=True)