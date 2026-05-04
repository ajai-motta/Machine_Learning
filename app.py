from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
      return render_template('home.html')
    if request.method=='POST':
       data=CustomData(request.form.get("school"),
        request.form.get("sex"),
        int(request.form.get("age")),
        request.form.get("address"),
        request.form.get("famsize"),
        request.form.get("Pstatus"),
        int(request.form.get("Medu")),
        int(request.form.get("Fedu")),
        request.form.get("Mjob"),
        request.form.get("Fjob"),
        request.form.get("reason"),
        request.form.get("guardian"),
        int(request.form.get("traveltime")),
        int(request.form.get("studytime")),
        int(request.form.get("failures")),
        request.form.get("schoolsup"),
        request.form.get("famsup"),
        request.form.get("paid"),
        request.form.get("activities"),
        request.form.get("nursery"),
        request.form.get("higher"),
        request.form.get("internet"),
        request.form.get("romantic"),
        int(request.form.get("famrel")),
        int(request.form.get("freetime")),
        int(request.form.get("goout")),
        int(request.form.get("Dalc")),
        int(request.form.get("Walc")),
        int(request.form.get("health")),
        int(request.form.get("absences")),
        int(request.form.get("G1")),
        int(request.form.get("G2")))
    data_frame=data.get_dta_as_data_frame()
    print(data_frame)
    predict=PredictPipeline()
    result=predict.predict(data_frame)
    print(result)
    return render_template('home.html',results=result[0])

if __name__=="__main__":
   app.run('0.0.0.0',debug=True)