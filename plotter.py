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

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter


def plot_data_train_val(y_train_predict, y_val_predict, dataset, index, steps, feature_label, model_name, segment):
    y_train_predict_plot = np.empty_like(dataset)
    y_train_predict_plot[:, :] = np.nan
    y_train_predict_plot[steps : len(y_train_predict) + steps, :] = y_train_predict

    y_val_predict_plot = np.empty_like(dataset)
    y_val_predict_plot[:, :] = np.nan
    y_val_predict_plot[len(y_train_predict) + (steps * 2) :, :] = y_val_predict

    for feature in range(dataset.shape[1]):
        plt.figure(dpi=300)
        # plt.title(f"Time vs Feature {feature + 1} - steps={steps} - model_name={model_name}")
        plt.title(f"{feature_label[feature]} vs Time- model_name={model_name}")
        plt.xlabel("Time")
        plt.ylabel(f"{feature_label[feature]}")
        plt.plot(index, dataset[:, feature], label="Data")
        plt.plot(index, y_train_predict_plot[:, feature], label=f"{feature_label[feature]} Train Prediction")
        plt.plot(index, y_val_predict_plot[:, feature], label=f"{feature_label[feature]} Val Prediction")

        plt.legend()

        # Adjust y-axis label formatting to prevent it from being cut off
        plt.gca().yaxis.get_offset_text().set_visible(False)
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        plt.tight_layout()

        save_directory = "plots"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(os.path.join(save_directory, "start_index" + str(index[0]) + "_end_index" + str(index[-1]) + "_segment" + str(segment) + "_steps" + str(steps) + "_model" + model_name + ".jpg"), format="jpg")
        plt.close()


def plot_data_and_prediction(y_predict, dataset, index, steps, feature_label, model_name, segment):
    y_predict_plot = np.empty_like(dataset)
    y_predict_plot[:, :] = np.nan
    y_predict_plot[steps : len(y_predict) + steps, :] = y_predict

    for feature in range(dataset.shape[1]):
        plt.figure(dpi=300)
        # plt.title(f"Time vs Feature {feature + 1} - steps={steps} - model={model_name}")
        plt.title(f"{feature_label[feature]} vs Time - model={model_name}")
        plt.xlabel("Time")
        plt.ylabel(f"{feature_label[feature]}")
        plt.plot(index, dataset[:, feature], label="Data")
        plt.plot(index, y_predict_plot[:, feature], label=f"{feature_label[feature]} Prediction")

        plt.legend()

        # Adjust y-axis label formatting to prevent it from being cut off
        plt.gca().yaxis.get_offset_text().set_visible(False)
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        plt.tight_layout()

        save_directory = "plots"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(os.path.join(save_directory, "start_index" + str(index[0]) + "_end_index" + str(index[-1]) + "_segment" + str(segment) + "_steps" + str(steps) + "_model" + model_name + "_testing_segment.jpg"), format="jpg")
        plt.close()


def plot_metric_segments(df, metric):
    save_directory = "plots"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    def extract_first_element(x):
        if isinstance(x, list) and len(x) > 0:
            return x[0]
        return None

    df_train_val = df[[f"train_{metric}", f"val_{metric}"]].map(extract_first_element)
    df_segment = df[[f"segment_{metric}"]].map(extract_first_element)

    if df_train_val.dropna().empty:
        print(f"No valid data to plot for {metric} (train/val).")
    else:
        plt.figure(dpi=300)
        df_train_val.plot()
        plt.xlabel("Segment")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs Segment")
        plt.legend([f"Train {metric.upper()}", f"Val {metric.upper()}"])

        # Adjust y-axis label formatting
        plt.gca().yaxis.get_offset_text().set_visible(False)
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        plt.tight_layout()

        plt.savefig(os.path.join(save_directory, f"metrics_train_val_{metric}.jpg"), format="jpg")
        plt.close()

    if df_segment.dropna().empty:
        print(f"No valid data to plot for {metric} (segment).")
    else:
        plt.figure(dpi=300)
        df_segment.plot()
        plt.xlabel("Segment")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs Segment")
        plt.legend([f"Segment {metric.upper()}"])

        # Adjust y-axis label formatting
        plt.gca().yaxis.get_offset_text().set_visible(False)
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
        plt.tight_layout()

        plt.savefig(os.path.join(save_directory, f"metrics_test_segment_{metric}.jpg"), format="jpg")
        plt.close()


def plot_dataframe(df, filename, x_label="Index", title="Model Error Comparison"):
    save_directory = "plots"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plot each column in the DataFrame
    plt.figure(dpi=300)
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel("Values")
    plt.title(title)
    plt.legend()

    # Adjust y-axis label formatting
    plt.gca().yaxis.get_offset_text().set_visible(False)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_directory, filename + ".jpg"), format="jpg")
    plt.close()
