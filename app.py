from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/details', methods=['GET', 'POST'])
def details():
    global latest_result
    if request.method == 'POST':
            data = {
                'gender': (request.form['gender']),
                'age': (request.form['age']),
                'history': (request.form['history']),
                'patient': (request.form['patient']),
                'take_medication': (request.form['take_medication']),
                'severity': (request.form['severity']),
                'breathe_shortness': (request.form['breathe_shortness']),
                'visual_change': (request.form['visual_change']),
                'nose_bleeding': (request.form['nose_bleeding']),
                'when_diagnosed': (request.form['when_diagnosed']),
                'systolic': (request.form['systolic']),
                'diastolic': (request.form['diastolic']),
                'controlled_diet': (request.form['controlled_diet']),
            }

            data_array = [data['gender'], data['age'], data['history'], data['patient'], data['take_medication'], data['severity'],
                        data['breathe_shortness'], data['visual_change'], data['nose_bleeding'],
                        data['when_diagnosed'], data['systolic'], data['diastolic'], data['controlled_diet']]
            
            
            columns = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication', 'Severity', 'BreathShortness',
                        'VisualChanges', 'NoseBleeding', 'Whendiagnoused', 'Systolic',
                        'Diastolic', 'ControlledDiet']
            
            with open('model/label_encoders.pkl', 'rb') as f:
                label_encoders = pickle.load(f)

            encoded_values = []
            ind = 0
            for col in columns:
                encoded_values.append(label_encoders[col].transform([data_array[ind]])[0])
                ind+=1

            features_values = np.array([encoded_values])
            df = pd.DataFrame(features_values, columns=columns)
            

            with open('model/decision_tree_model.pkl', 'rb') as f:
                model = pickle.load(f)

            
            prediction = model.predict(df)

            prediction_decoded = label_encoders['Stages'].inverse_transform(prediction)

            latest_result = prediction_decoded[0]

            return redirect(url_for("show_result"))


    return render_template('details.html')

@app.route("/result")
def show_result():
     global latest_result

     return render_template("result.html", result = latest_result)
