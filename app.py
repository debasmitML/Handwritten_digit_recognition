import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)


load_model=pickle.load(open('SVM_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    test_data=np.array(data).reshape(1,-1)
    output=load_model.predict(test_data)
    output={'predicted_output':int(output[0])}
    print(output)
    return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)