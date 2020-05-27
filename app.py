import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    feature_name = ['sepal_length','sepal_width','petal_length','petal_width']
    
    df = pd.DataFrame(final_features, columns = feature_name)
    prediction = model1.predict(df)

    output = prediction

    return render_template('index.html', prediction_text='Species of flower is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)