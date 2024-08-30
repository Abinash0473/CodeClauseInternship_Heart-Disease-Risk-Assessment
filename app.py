from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        chest_pain = float(request.form['chest_pain'])
        bp = float(request.form['bp'])
        cholesterol = float(request.form['cholesterol'])
        fbs = float(request.form['fbs'])
        ekg = float(request.form['ekg'])
        max_hr = float(request.form['max_hr'])
        angina = float(request.form['angina'])
        st_depression = float(request.form['st_depression'])
        slope = float(request.form['slope'])
        vessels = float(request.form['vessels'])
        thallium = float(request.form['thallium'])

        # Create a numpy array for prediction
        features = np.array([[age, sex, chest_pain, bp, cholesterol, fbs, ekg, max_hr, angina, st_depression, slope, vessels, thallium]])
        
        # Predict using the model
        prediction = model.predict(features)
        risk = 'High Risk' if prediction[0] == 1 else 'Low Risk'

        return render_template('index.html', risk=risk)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
