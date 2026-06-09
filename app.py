from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    age = float(data['age'])
    income = float(data['income'])
    loan_amount = float(data['loan_amount'])
    credit_score = float(data['credit_score'])
    interest_rate = float(data['interest_rate'])
    dti_ratio = float(data['dti_ratio'])
    employment_type = data['employment_type']

    emp_full = 1 if employment_type == 'Full-time' else 0
    emp_part = 1 if employment_type == 'Part-time' else 0
    emp_self = 1 if employment_type == 'Self-employed' else 0
    emp_unemp = 1 if employment_type == 'Unemployed' else 0

    features = [[age, income, loan_amount, credit_score,
                 interest_rate, dti_ratio,
                 emp_full, emp_part, emp_self, emp_unemp]]

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        result = "⚠️ High Risk — Likely to Default"
    else:
        result = "✅ Low Risk — Unlikely to Default"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)