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

import os
from tensorflow.keras.models import Sequential

import config


def train_model(model: Sequential, X_train, y_train, model_name: str):

    model.fit(X_train, y_train, epochs=config.epochs, batch_size=1, verbose=config.verbose)

    save_directory = "models"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model_path = os.path.join(save_directory, model_name + ".h5")
    model.save(model_path)


def train_ensemble_model(model, X_train, y_train):
    if type(model.input) == list:
        X = [X_train for _ in range(len(model.input))]
    else:
        X = [X_train]
    model.fit(X, y_train, epochs=config.epochs, verbose=config.verbose)

    save_directory = "models"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model_path = os.path.join(save_directory, "ensemble.h5")
    model.save(model_path)
