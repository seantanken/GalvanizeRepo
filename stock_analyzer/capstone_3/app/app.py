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

def scale_df(original_df, columns=['Close', 'Volume']):
    df = original_df.copy()
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    
    # get date from index
    if 'Date' not in df.columns:
        df['Date'] = df.index
    
    column_scaler = {}
    # scale the data (prices) from 0 to 1
    #scaler = MinMaxScaler()
    for column in columns:
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler
    # add the MinMaxScaler instances to the result returned
    result["column_scaler"] = column_scaler
    return df, result

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def get_predict_and_past_steps(df, result, past_steps=1950, predict_steps=390, pred_col='Close', columns=['Close', 'Volume']):
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df[pred_col].shift(-predict_steps)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[columns].tail(predict_steps))
    # drop NaNs
    df.dropna(inplace=True)
    
    sequence_data = []
    sequences = deque(maxlen=past_steps)
    for entry, target in zip(df[columns + ['Date']].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == past_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(columns)] for s in sequences]) + list(last_sequence)
    
    last_sequence = np.array(last_sequence, dtype='object')
    #last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    return result, sequence_data

def Xy_split(df, result, sequence_data, shuffle, test_size=0.2):
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # split the dataset into training & testing sets by date (not randomly splitting)
    train_samples = int((1 - test_size) * len(X))
    result["X_train"] = X[:train_samples]
    result["y_train"] = y[:train_samples]
    result["X_test"]  = X[train_samples:]
    result["y_test"]  = y[train_samples:]
    # shuffle the datasets for training
    if shuffle == True:
        shuffle_in_unison(result["X_train"], result["y_train"])
        shuffle_in_unison(result["X_test"], result["y_test"])
    return result

def test_train_split(result, columns=['Close', 'Volume']):
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    df = result['df']
    result["test_df"] = df.loc[df['Date'].isin(dates)]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(columns)].astype(np.float32)
    return result

def data_for_model(input_df, columns, test_size, shuffle):
    df, result = scale_df(input_df, columns=columns)
    result, sequence_data = get_predict_and_past_steps(df, result, past_steps=40, predict_steps=1, pred_col='AMZN_rolling_close', columns=columns)
    result = Xy_split(df, result, sequence_data, shuffle=shuffle, test_size=test_size)
    data = test_train_split(result, columns=columns)
    return data

def scale_result(predictions):
    pred_copy = predictions.copy()
    pred_2der = np.diff(pred_copy, n=1, axis=0)
    pos_2der_ind = pred_2der > 0
    neg_2der_ind = pred_2der < 0
    scale = 0.01
    endcap = np.array([False])
    pred_copy[np.concatenate([endcap, pos_2der_ind])] *= (1 + scale)
    pred_copy[np.concatenate([endcap, neg_2der_ind])] *= (1 - scale)
    return pred_copy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    inputs = request.args
    
    if len(inputs["day"]) == 1:
        day = f'0{inputs["day"]}'
    else:
        day = inputs["day"]

    if len(inputs["month"]) == 1:
        month = f'0{inputs["month"]}'
    else:
        month = inputs["month"]

    date = f'2021-{month}-{day}'

    if data['date_origin'].str.contains(date).any():
        end_idx = data.index[data['date_origin'] == date][0]
        print(end_idx)
        start_idx = end_idx - 50
        data_df = data.iloc[start_idx:end_idx]
    else:
        return jsonify(model_result=f'{date} is not a valid stock trading day or no data was collected for this day.')

    model_data = data_for_model(data_df, ['AMZN_vol', 'AMZN_rolling_close', 'GOOG_rolling_close'], test_size=1.0, shuffle=False)

    result = []
    target_price = model_data['df'].iloc[-1]['AMZN_rolling_close'])

    prediction = amzn_model.predict(model_data['X_test'])
    predictions = np.squeeze(model_data["column_scaler"]["AMZN_rolling_close"].inverse_transform(prediction))
    targets = np.squeeze(model_data["column_scaler"]["AMZN_rolling_close"].inverse_transform(np.expand_dims(model_data['y_test'], axis=0)))
    raw_r2 = r2_score(targets, predictions)
    raw_prediction = float(predictions[-1])

    scaled_predictions = scale_result(predictions)
    final_r2 = r2_score(targets, scaled_predictions)
    final_prediction = float(scaled_predictions[-1])

    result = f"Recorded price on {date}: {target_price}     Predicted price: {raw_prediction} with an R^2 score of {raw_r2}     Scaled predicted price: {final_prediction} with an R^2 score of {final_r2}"
    return jsonify(model_result=result)

app.run(host='0.0.0.0', port=5000, debug=True)