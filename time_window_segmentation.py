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

import pandas as pd


## this is test function
def feature_derivation_time_window(file_path, time_window_seconds):
    df = pd.read_csv(file_path)

    # Apply the function to the DataFrame column
    # df['Time'] = df['Time'].apply(convert_minutes_to_seconds)

    time_segments = time_window(df, time_window_seconds)

    return time_segments


# API fpr time segmentation
def time_window(df, sec):
    x = sec
    actual_max_time = df["Time"].max()

    # Calculate the largest multiple of x less than or equal to actual_max_time
    max_time = (actual_max_time // x) * x

    segments = []
    start = 0
    while start < max_time:
        end = start + x
        filtered_df = df[(df["Time"] >= start) & (df["Time"] < end)]
        segments.append(filtered_df)
        start = end

    return segments


def convert_minutes_to_seconds(minutes):
    return minutes * 1
