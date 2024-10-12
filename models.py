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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Flatten, Dropout, concatenate, SimpleRNN

from tensorflow import keras
from tensorflow.keras import layers

from keras.models import load_model
import glob


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    output_size=1,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(output_size)(x)
    return keras.Model(inputs, outputs)


def get_lstm(steps: int, output_size: int) -> Sequential:
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(128, input_shape=(steps, 1)))
    model.add(Dense(output_size))
    return model


def get_rnn(steps: int, output_size: int) -> Sequential:
    # create and fit the LSTM network
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, input_shape=(steps, 1)))
    model.add(SimpleRNN(63))
    model.add(Dense(output_size))
    return model


def get_stacked_lstm(steps: int, output_size: int) -> Sequential:
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(128, input_shape=(steps, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(output_size))
    return model


def get_gru(steps: int, output_size: int) -> Sequential:
    model = Sequential()
    model.add(GRU(128, input_shape=(steps, 1)))
    model.add(Dense(output_size))
    return model


def get_cnn_lstm(steps: int, output_size: int) -> Sequential:
    activation = "relu"

    model = Sequential()

    model.add(Conv1D(128, strides=1, input_shape=(steps, 1), activation=activation, kernel_size=1, padding="valid"))
    model.add(Conv1D(128, strides=1, activation=activation, kernel_size=1, padding="valid"))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(128, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation=activation))

    return model


def get_dense(steps: int, output_size: int) -> Sequential:
    activation = "relu"

    model = Sequential()
    model.add(Flatten(input_shape=(steps, 1)))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation))

    return model


def get_transformer(steps: int, output_size: int) -> Sequential:
    # create and fit the LSTM network
    model = build_model(
        input_shape=(steps, 1),
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        output_size=output_size,
    )
    return model


def load_all_models(exception_list=None):
    if exception_list is None:
        exception_list = []

    all_models = list()
    model_files = glob.glob("models/*.h5")

    for filename in model_files:
        if filename not in exception_list:
            model = load_model(filename)
            all_models.append(model)
            print(">loaded %s" % filename)
        else:
            print(">skipped %s" % filename)

    return all_models


def define_stacked_model(members, output_size: int):
    if len(members) == 0:
        raise ValueError("No models provided for ensembling.")

    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = False
            layer._name = f"ensemble_{i + 1}_{layer.name}"

    if len(members) == 1:
        # If there's only one model, use it directly without concatenation
        model = members[0]
        hidden = Dense(10, activation="relu")(model.output)
        output = Dense(output_size)(hidden)
        model = Model(inputs=model.input, outputs=output)
    else:
        # If there are multiple models, concatenate their outputs
        ensemble_visible = [model.input for model in members]
        ensemble_outputs = [model.output for model in members]
        merge = concatenate(ensemble_outputs)
        hidden = Dense(10, activation="relu")(merge)
        output = Dense(output_size)(hidden)
        model = Model(inputs=ensemble_visible, outputs=output)

    return model


def get_ensemble(output_size: int):
    members = load_all_models(["ensemble.h5"])
    print("Loaded %d models" % len(members))

    stacked_model = define_stacked_model(members, output_size)

    return stacked_model
