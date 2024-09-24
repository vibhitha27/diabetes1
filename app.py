from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the saved RandomForest model
model = joblib.load('diabetes_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = request.form['smoking_history']
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        
        # Encode categorical data (assuming you used label encoding for gender and smoking_history)
        if gender == 'Male':
            gender = 1
        else:
            gender = 0
        
        smoking_dict = {'never': 0, 'current': 1, 'former': 2, 'ever': 3, 'No Info': 4}
        smoking_history = smoking_dict.get(smoking_history, 4)  # Use 'No Info' as default

        # Convert the form data into a 2D array for model prediction
        features = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
        
        # Predict using the loaded model
        prediction = model.predict(features)
        
        # Display the result
        if prediction[0] == 1:
            result = "The patient is likely to have diabetes."
        else:
            result = "The patient is unlikely to have diabetes."
            
        return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
