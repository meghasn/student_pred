import pickle
from flask import Flask,request,render_template,send_from_directory
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application=Flask(__name__)

app=application

##Route for Homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        print(request.form.get('ethnicity'))
        print("IIIIIIIIIIIIIIIIIIIII")
        pred_df=data.get_data_as_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

@app.route('/styles.css')
def serve_css():
    css_folder = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(css_folder, 'style.css')
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)