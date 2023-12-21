import pickle
from flask import Flask, request, render_template
import numpy as np 
import pandas as pd

from sources.pipeline.predict_pipeline import CustomData
from sources.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)
app = application

# route for homepage 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Gender=request.form.get('Gender'),
            education=request.form.get('education'),
            prevalentStroke=request.form.get('prevalentStroke'),
            age=int(request.form.get('age')),
            cigsPerDay=int(request.form.get('cigsPerDay')),
            sysBP=float(request.form.get('sysBP')),
            diaBP=float(request.form.get('diaBP')),
            BMI=float(request.form.get('BMI')),
            heartRate=int(request.form.get('heartRate')),
            glucose=int(request.form.get('glucose')),
            currentSmoker=int(request.form.get('currentSmoker')),
            BPMeds=int(request.form.get('BPMeds')),
            prevalentHyp=int(request.form.get('prevalentHyp')),
            diabetes=int(request.form.get('diabetes')),
            totChol=int(request.form.get('totChol'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)