from sklearn.preprocessing import MinMaxScaler
import numpy.random as rand
import pandas as pd
import numpy as np
from collections import deque

class data_prep:

    def __init__(self, input_df):
        self.input_df = input_df
        self.result = {}
    
    def scale_df(self, original_df, columns=['Close', 'Volume']):
        df = original_df.copy()
        # contains all the elements we want to return from this function
        #result = {}
        # also return the original dataframe itself
        self.result['df'] = df.copy()
        
        # get date from index
        if 'Date' not in df.columns:
            df['Date'] = df.index
        
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        self.result["column_scaler"] = column_scaler
        return df

    def get_predict_and_past_steps(self, df, past_steps=1950, predict_steps=390, pred_col='Close', columns=['Close', 'Volume']):
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
        self.result['last_sequence'] = last_sequence
        return sequence_data

    def shuffle_in_unison(self, a, b):
        # shuffle two arrays in the same way
        state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(state)
        np.random.shuffle(b)

    def Xy_split(self, df, sequence_data, shuffle, test_size=0.2):
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
        self.result["X_train"] = X[:train_samples]
        self.result["y_train"] = y[:train_samples]
        self.result["X_test"]  = X[train_samples:]
        self.result["y_test"]  = y[train_samples:]
        # shuffle the datasets for training
        if shuffle == True:
            shuffle_in_unison(self.result["X_train"], self.result["y_train"])
            shuffle_in_unison(self.result["X_test"], self.result["y_test"])

    def test_train_split(self, columns=['Close', 'Volume']):
        # get the list of test set dates
        dates = self.result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        df = self.result['df']
        self.result["test_df"] = df.loc[df['Date'].isin(dates)]
        # remove duplicated dates in the testing dataframe
        self.result["test_df"] = self.result["test_df"][~self.result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        self.result["X_train"] = self.result["X_train"][:, :, :len(columns)].astype(np.float32)
        self.result["X_test"] = self.result["X_test"][:, :, :len(columns)].astype(np.float32)

    def data_for_model(self, columns, test_size, shuffle):
        df = self.scale_df(self.input_df, columns=columns)
        sequence_data = self.get_predict_and_past_steps(df, past_steps=40, predict_steps=1, pred_col='AMZN_rolling_close', columns=columns)
        self.Xy_split(df, sequence_data, shuffle=shuffle, test_size=test_size)
        self.test_train_split(columns=columns)