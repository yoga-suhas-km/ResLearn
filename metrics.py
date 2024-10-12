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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def rmse(y_actual, y_predict):
    """
    Calculates the Root Mean Squared Error of each feature and puts it into an array.
    """
    rmse = []
    for col in range(y_actual.shape[1]):
        rmse.append(np.sqrt(mean_squared_error(y_actual[:, col], y_predict[:, col])))
    return rmse


def mape(y_actual, y_predict):
    """
    Calculates the Mean Absolute Percentage Error of each feature and puts it into an array.
    """
    mape = []
    for col in range(y_actual.shape[1]):
        mape.append(mean_absolute_percentage_error(y_actual[:, col], y_predict[:, col]))
    return mape


def smape(y_actual, y_predict):
    """
    Calculates the Symmetric Mean Absolute Percentage Error of each feature and puts it into an array.
    """
    smape = []
    for col in range(y_actual.shape[1]):
        numerator = np.abs(y_actual[:, col] - y_predict[:, col])
        denominator = (np.abs(y_actual[:, col]) + np.abs(y_predict[:, col])) / 2.0
        smape.append(np.mean(numerator / denominator) * 100)
    return smape
