from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from src.predict_pipeline import CustomData,PredictPipeline

from sklearn.preprocessing import StandardScaler

application=Flask(__name__)

app=application

# route for a home 
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template("home.html")
    else:
        pass