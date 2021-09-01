from flask import Flask, render_template
from flask import request, jsonify
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from tensorflow.keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

import numpy.random as rand
import scipy.stats as stats 
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor 
from sklearn.model_selection import StratifiedKFold
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

app = Flask(__name__)

amzn_model = keras.models.load_model('C:/Users/seant/stock_analyzer/capstone_3/models')
data = pd.read_csv('C:/Users/seant/stock_analyzer/capstone_3/data/prepped_stock_df.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    inputs = request.args
    for val in inputs:
        print(val)
    
    if len(inputs["day"]) == 1:
        day = f'0{inputs["day"]}'
    else:
        day = inputs["day"]

    if len(inputs["month"]) == 1:
        month = f'0{inputs["month"]}'
    else:
        month = inputs["month"]

    date = f'2021-{month}-{day}'

    if data['date_origin'].str.contains(date):
        result = data.loc[['date_origin'] == date, 'AMZN_rolling_close']
    else:
        result = f'{date} is not a valid stock trading day or no data was collected for this day.'


    return jsonify(model_result=result)

app.run(host='0.0.0.0', port=5000, debug=True)