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
        # Collect form data
        student_data = {
            'school': request.form.get('school'),
            'sex': request.form.get('sex'),
            'age': int(request.form.get('age')),
            'address': request.form.get('address'),
            'famsize': request.form.get('famsize'),
            'Pstatus': request.form.get('Pstatus'),
            'Medu': int(request.form.get('Medu')),
            'Fedu': int(request.form.get('Fedu')),
            'Mjob': request.form.get('Mjob'),
            'Fjob': request.form.get('Fjob'),
            'reason': request.form.get('reason'),
            'guardian': request.form.get('guardian'),
            'traveltime': int(request.form.get('traveltime')),
            'studytime': int(request.form.get('studytime')),
            'failures': int(request.form.get('failures')),
            'schoolsup': request.form.get('schoolsup'),
            'famsup': request.form.get('famsup'),
            'paid': request.form.get('paid'),
            'activities': request.form.get('activities'),
            'nursery': request.form.get('nursery'),
            'higher': request.form.get('higher'),
            'internet': request.form.get('internet'),
            'romantic': request.form.get('romantic'),
            'famrel': int(request.form.get('famrel')),
            'freetime': int(request.form.get('freetime')),
            'goout': int(request.form.get('goout')),
            'Dalc': int(request.form.get('Dalc')),
            'Walc': int(request.form.get('Walc')),
            'health': int(request.form.get('health')),
            'absences': int(request.form.get('absences'))
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