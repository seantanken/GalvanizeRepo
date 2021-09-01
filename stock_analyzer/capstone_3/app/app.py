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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    inputs = request.args
    results = f'{inputs["month"]}-{inputs["day"]}-2021'

    return jsonify(model_results=results)

app.run(host='0.0.0.0', port=5000, debug=True)