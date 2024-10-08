from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Ensure the correct loading of the model and catch errors
try:
    with open("sklearn_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: The 'predictions.pkl' file was not found. Ensure it is located in the correct directory.")
    model = None  # Set model to None so that it doesn't crash during prediction

# Ensure the correct loading of the dataset and catch errors
try:
    student_data = pd.read_csv('Student_Marks.csv')
except FileNotFoundError:
    print("Error: The 'Student_Marks.csv' file was not found. Ensure it is located in the correct directory.")
    student_data = pd.DataFrame(columns=['number_courses', 'time_study'])  # Empty dataframe as fallback

@app.route('/')
def index():
    if student_data.empty:
        return "Error: Student data not available. Please check the dataset file."
    
    number_courses = sorted(student_data['number_courses'].unique())
    time_study = sorted(student_data['time_study'].unique())
    return render_template('index.html', number_courses=number_courses, time_study=time_study)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not loaded. Please check the model file."

    try:
        number_courses = int(request.form.get('number_courses'))
        time_study = float(request.form.get('time_study'))
        
        # Make prediction using the loaded model
        prediction = model.predict(pd.DataFrame([[number_courses, time_study]], 
                                                columns=['number_courses', 'time_study']))
        
        return str(np.round(prediction[0], 2))

    except Exception as e:
        return f"An error occurred during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
