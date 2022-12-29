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

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    tsne_input=np.array(data).reshape(1,-1)
    print(tsne_input)
    output=load_model.predict(tsne_input)[0]
    return render_template("home.html",predicted_output="The predcted output digit is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)