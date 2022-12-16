import flask
from flask import render_template
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        
        x_list = []
        for x in range(1,13,1):
            x_list.append(float(flask.request.form[f'x{x}']))
        print(x_list)
        with open ('model_vkr.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
       
        
        
        
       
        y_pred = loaded_model.predict(np.array(x_list).reshape(1, -1))

        return render_template('main.html', result = y_pred)

if __name__== '__main__':
    app.run()
