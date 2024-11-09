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
import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy.stats import entropy
from data_processing import prepare_data
from scipy.stats import wilcoxon
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.ticker import ScalarFormatter

import config
import argparse
from segmentation_processing import load_data_from_packet_segmentation, load_data_from_time_window_segmentation
from extract_raw_features import isolate_data
from rolling_over_window import rolling_over_window


# Setting plot style
# plt.style.use('seaborn-darkgrid')
# matplotlib inline


# Create directory for saving plots if it doesn't exist
save_directory = "explorative_data_analysis"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="Dataset1", help="Dataset1, Dataset2 or Dataset3")
parser.add_argument("--experiment", default="1", help="1 or 2 for dataset3")
args = parser.parse_args()


def plot_data(data):
    plt.figure(dpi=300)
    # Plotting the time series
    plt.plot(data, label=f"{config.required_columns[0]}")
    plt.title(f"Time Series for {args.data}")
    # plt.show()
    ## add save function
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    # Adjust y-axis label formatting to prevent it from being cut off
    plt.gca().yaxis.get_offset_text().set_visible(False)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    plt.tight_layout()

    plt.savefig(os.path.join(save_directory, "time_series.jpg"), format="jpg")
    plt.close()


def rolling_stats(data):
    window_size = 20
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()

    plt.figure(dpi=300, figsize=(12, 6))
    plt.plot(data, label="frame_size")
    plt.plot(rolling_mean, label=f"Rolling Mean ({window_size} window)", color="red")
    plt.plot(rolling_std, label=f"Rolling Std Dev ({window_size} window)", color="black")
    plt.title("Rolling Mean and Standard Deviation")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(save_directory, "rolling_stats.jpg"), format="jpg")
    plt.close()


def decomposition(data):
    # Decomposition using additive model
    plt.figure(dpi=300)
    decomposition = seasonal_decompose(data, model="additive", period=5)

    # Plotting the decomposition
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)

    # Access the first subplot (axes[0])
    axes = fig.get_axes()
    axes[0].set_title("Decomposition - Trend")
    axes[0].set_ylabel("Frame Size")

    # Set the x-axis label (common for all subplots)
    plt.xlabel("Time")

    # Save the figure
    plt.savefig(os.path.join(save_directory, "decomposition.jpg"), format="jpg")
    plt.close()


def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")
    if result[1] <= 0.05:
        print("Conclusion: The time series is stationary.")
    else:
        print("Conclusion: The time series is non-stationary.")


def acf(data):
    # Autocorrelation Plot
    plot_acf(data)
    plt.title("Autocorrelation Function")
    # plt.show()
    ## add save function
    plt.ylabel("Correlation")
    plt.xlabel("Lag")
    plt.savefig(os.path.join(save_directory, "autocorrelation.jpg"), format="jpg", dpi=300)
    plt.close()


def lb_test(data):
    # Perform the Ljung-Box test
    lb_test = acorr_ljungbox(data, lags=[10], return_df=True)
    print("\n\n*****  lb Test *****")
    print(lb_test)


def runs_test(data):
    # Runs test for randomness
    median = np.median(data)
    runs = np.sign(data - median)

    runs_test_result = wilcoxon(runs)
    print("\n\n*****  Runs Test *****")

    print(f"wilcoxon Runs Test Statistic: {runs_test_result.statistic}, p-value: {runs_test_result.pvalue}")

    z_stat, p_value = runstest_1samp(data, median)
    print(f"stats model API Runs Test Statistic: {z_stat}, p-value: {p_value}")


# Shannon Entropy:
def shannon_entropy(data):
    """Compute the Shannon entropy of a time series."""
    data_counts = np.histogram(data, bins=10)[0]
    return entropy(data_counts)


def shannon_test(data):
    entropy_value = shannon_entropy(data)

    print("\n\n*****  Shannon Entropy Test *****")

    print(f"Shannon Entropy: {entropy_value}")


def hist_plot(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=20, kde=True, color="skyblue")
    plt.title("Histogram of Time Series Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    # plt.show()
    plt.savefig(os.path.join(save_directory, "histogram.jpg"), format="jpg")
    plt.close()


def run_tests(data):
    # time_series = prepare_data("Dataset1", "packets", 2)#.iloc[:1400]
    # #time_series = prepare_data("Dataset1", "time_window", 2)#.iloc[:1400]
    # print(time_series)
    # #time_series = time_series["total_pkt_length_per_segment"]/1000000 #mostly predictable with lag 1 for N=1000 Dataset1
    # time_series = time_series["avg_frame_iat"]*1000 #mostly predictable with lag 1 for N=1000 Dataset1
    # #time_series = time_series["total_frame_duartion"]*1000000 #micro seconds #tough to predict with lag 1 for N=1000 Dataset1
    # #time_series = time_series["frame_count"] #mostly predictable with lag 1 for N=1000 Dataset1
    # data = data/1000000
    plot_data(data)
    hist_plot(data)
    rolling_stats(data)
    decomposition(data)
    adf_test(data)
    acf(data)
    lb_test(data)
    runs_test(data)
    shannon_test(data)


def explorative_data_analysis():
    # if config.single_csv:
    #     df = pd.read_csv(config.single_csv_location)
    #     df = isolate_data(df)
    #     if args.segment_type == config.Segment_type[0]:
    #         df = load_data_from_packet_segmentation(df, 500)
    #     else:
    #         df = load_data_from_time_window_segmentation(df, 1)
    #     df = df[config.required_columns]
    # else:
    df = prepare_data(args.data, "packets", int(args.experiment))[config.required_columns[0]]

    df = rolling_over_window(df, 20)

    run_tests(df[:700])


if __name__ == "__main__":
    explorative_data_analysis()
