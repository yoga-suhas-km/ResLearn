"""
MIT License

Copyright (c) 2024 Yoga Suhas Kuruba Manjunath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@Authors: 

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config


def format_X_Y(df: pd.DataFrame, steps: int):
    """
    Formats the input DataFrame into X and y arrays for LSTM model.
    Uses the first feature to predict the next in the sequence and following features
    Univariate input assumes first column of df is the X feature
    Multivariate output assumes all columns of df are the y features
    """

    dataset = df.astype("float32").to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # arrange time steps ie. 2 steps
    # from [[0], [1], [2], [3], [4], [5]] to
    # X = [[0, 1], [1, 2], [2, 3], [3, 4]]
    # [time steps, features]
    # y = [[2], [3], [4], [5]]
    X, y = [], []

    for i in range(dataset.shape[0] - steps):
        X.append(dataset[i : i + steps, 0])
        y.append(dataset[i + steps, :].tolist())

    X = np.array(X)
    y = np.array(y)

    # reshape input to be [samples, time steps, features] for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    unscaled_dataset = scaler.inverse_transform(dataset)

    return X, y, unscaled_dataset, scaler


def config_data(df: pd.DataFrame, steps: int):
    # split into train and test sets
    train_size = int(len(df) * config.training_portion)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    # reshape into X=t and Y=t+1
    X_train, y_train, train_dataset, scaler_train = format_X_Y(train, steps)
    X_test, y_test, test_dataset, scaler_test = format_X_Y(test, steps)

    dataset = np.vstack((train_dataset, test_dataset))

    return X_train, y_train, X_test, y_test, scaler_train, scaler_test, dataset
