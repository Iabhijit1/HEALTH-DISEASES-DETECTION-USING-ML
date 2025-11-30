from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model and scaler
ensemble_model_loaded = joblib.load('ensemble_model.joblib')
scaler_loaded = joblib.load('scaler.joblib')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the About page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route for the Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        session['user_info'] = {
            'name': request.form['name'],
            'phone': request.form['phone'],
            'address': request.form['address'],
            'parents_name': request.form['parents_name'],
            'email': request.form['email'],
            'dob': request.form['dob']
        }
        return redirect(url_for('predict'))
    return render_template('register.html')

# Route for the Prediction Form page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_info' not in session:
        return redirect(url_for('register'))
    
    featurs = [
        "Glucose", "Cholesterol", "Hemoglobin", "Platelets", "White Blood Cells",
        "Red Blood Cells", "Hematocrit", "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin",
        "Mean Corpuscular Hemoglobin Concentration", "Insulin", "BMI", "Systolic Blood Pressure",
        "Diastolic Blood Pressure", "Triglycerides", "HbA1c", "LDL Cholesterol", "HDL Cholesterol",
        "ALT", "AST", "Heart Rate", "Creatinine", "Troponin", "C-reactive Protein"
    ]
    
    if request.method == 'POST':
        input_data = [float(request.form[feat]) for feat in featurs]
        input_data = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler_loaded.transform(input_data)
        numeric_predictions = ensemble_model_loaded.predict(input_data_scaled)
        
        disease_mapping = {0: 'Anemia', 1: 'Diabetes', 2: 'Healthy', 3: 'Heart Disease', 4: 'Thalassemia', 5: 'Thrombocytopenia'}
        disease_predictions = [disease_mapping[num] for num in numeric_predictions]
        
        return render_template('result.html', predictions=disease_predictions, user_info=session['user_info'])
    
    return render_template('predict.html', featurs=featurs)

if __name__ == '__main__':
    app.run(debug=True)
