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
"""

import os
import pandas as pd
import config as cfg


from dataset3_preparation import Dataset3_exp1_preparation, Dataset3_exp2_preparation
from dataset2_preparation import Dataset2_exp1_preparation
from dataset1_preparation import Dataset1_exp1_preparation


# call to get the dataset for train or test
# Provide pandas dataframe
# seg_type: 0 to choose segment type one (packet-based segmentation), 1 to choose segment type two (time-based segmentation)
# if 0 is selected for seg_type, then provide seg_size. seg_size = 50 consider 50 packets per segmentation. seg_window will be ignored.
# if 1 is selected for seg_type, then provide seg_window. seg_windows = 1 considers all packets fall within 1 second to for segmenttion. seg_size will be ignored.


def load_or_prepare_data(file_path, preparation_function, seg_type):
    if os.path.exists(file_path):
        # Load the data if the file exists
        df = pd.read_csv(file_path)
        #print(f"Loaded data from {file_path}")
    else:
        # Run the function to generate the data if the file does not exist
        df = preparation_function(seg_type)
        df.to_csv(file_path, index=False)
        #print(f"Generated and saved data to {file_path}")

    return df


def prepare_data(__data, seg_type, exp):
    if __data == cfg.Data_list[0]:  # ****************** for Dataset 1
        if not os.path.exists(cfg.Processed_data1):  # ****************** for Dataset 1
            os.mkdir(cfg.Processed_data1)  # ****************** for Dataset 1

        if exp == cfg.Dataset_exp[0]:  # ****************** for Dataset 1 exp1
            curPath, dirs, files = next(os.walk(cfg.Processed_data1))  # ****************** for Dataset 1
            path = os.path.join(cfg.Processed_data1 + "\\" + ("exp" + str(cfg.Dataset_exp[0])))  # ****************** for Dataset 1 exp1
            if not ("exp" + str(cfg.Dataset_exp[0])) in dirs:  # ****************** for Dataset 1 exp1
                os.mkdir(path)

            if seg_type == cfg.Segment_type[0]:
                return load_or_prepare_data(os.path.join(path, "prediction_data_" + seg_type + "_" + str(cfg.Dataset1_exp1_N) + ".csv"), Dataset1_exp1_preparation, seg_type)
            elif seg_type == cfg.Segment_type[1]:
                return load_or_prepare_data(os.path.join(path, "prediction_data_" + seg_type + "_" + str(cfg.Dataset1_exp1_Time_Window) + ".csv"), Dataset1_exp1_preparation, seg_type)


    if __data == cfg.Data_list[1]:  # ****************** for Dataset 2
        if not os.path.exists(cfg.Processed_data2):  # ****************** for Dataset 2
            os.mkdir(cfg.Processed_data2)  # ****************** for Dataset 2

        if exp == cfg.Dataset_exp[0]:  # ****************** for Dataset 2 exp1
            curPath, dirs, files = next(os.walk(cfg.Processed_data2))  # ****************** for Dataset 2
            path = os.path.join(cfg.Processed_data2 + "\\" + ("exp" + str(cfg.Dataset_exp[0])))  # ****************** for Dataset 2 exp1
            if not ("exp" + str(cfg.Dataset_exp[0])) in dirs:  # ****************** for Dataset 2 exp 1
                os.mkdir(path)  # ****************** for Dataset 2 exp 1
            if seg_type == cfg.Segment_type[0]:
                return load_or_prepare_data(os.path.join(path, "prediction_data_" + seg_type + "_" + str(cfg.Dataset2_exp1_N) + ".csv"), Dataset2_exp1_preparation, seg_type)  # ****************** for Dataset 1 exp3
            elif seg_type == cfg.Segment_type[1]:
                return load_or_prepare_data(os.path.join(path, "prediction_data_" + seg_type + "_" + str(cfg.Dataset2_exp1_Time_Window) + ".csv"), Dataset2_exp1_preparation, seg_type)

    if __data == cfg.Data_list[2]:  # ****************** for Dataset 3
        if not os.path.exists(cfg.Processed_data3):  # ****************** for Dataset 3
            os.mkdir(cfg.Processed_data3)  # ****************** for Dataset 3

        if exp == cfg.Dataset_exp[0]:  # ****************** for Dataset 3 exp1
            curPath, dirs, files = next(os.walk(cfg.Processed_data3))  # ****************** for Dataset 3
            path = os.path.join(cfg.Processed_data3 + "\\" + ("exp" + str(cfg.Dataset_exp[0])))  # ****************** for Dataset 3 exp1
            if not ("exp" + str(cfg.Dataset_exp[0])) in dirs:  # ****************** for Dataset 3 exp 1
                os.mkdir(path)  # ****************** for Dataset 3 exp 1
            if seg_type == cfg.Segment_type[0]:
                return load_or_prepare_data(os.path.join(path, "prediction_data_" + seg_type + "_" + str(cfg.Dataset3_exp1_N) + ".csv"), Dataset3_exp1_preparation, seg_type)  # ****************** for Dataset 1 exp3
            elif seg_type == cfg.Segment_type[1]:
                return load_or_prepare_data(os.path.join(path, "prediction_data_" + seg_type + "_" + str(cfg.Dataset3_exp1_Time_Window) + ".csv"), Dataset3_exp1_preparation, seg_type)

        if exp == cfg.Dataset_exp[1]:  # ****************** for Dataset 3 exp2
            curPath, dirs, files = next(os.walk(cfg.Processed_data3))  # ****************** for Dataset 3
            path = os.path.join(cfg.Processed_data3 + "\\" + ("exp" + str(cfg.Dataset_exp[1])))  # ****************** for Dataset 3 exp2
            if not ("exp" + str(cfg.Dataset_exp[1])) in dirs:  # ****************** for Dataset 3 exp 1
                os.mkdir(path)  # ****************** for Dataset 3 exp 2
            if seg_type == cfg.Segment_type[0]:
                return load_or_prepare_data(os.path.join(path, "prediction_data_" + seg_type + "_" + str(cfg.Dataset3_exp2_N) + ".csv"), Dataset3_exp2_preparation, seg_type)  # ****************** for Dataset 1 exp3
            elif seg_type == cfg.Segment_type[1]:
                return load_or_prepare_data(os.path.join(path, "prediction_data_" + seg_type + "_" + str(cfg.Dataset3_exp2_Time_Window) + ".csv"), Dataset3_exp2_preparation, seg_type)

# def load_data():
# data_path = os.path.join(".",cfg.path)

# df = pd.read_csv(data_path)
# df = df.drop(columns=["Protocol", "Info"])
# df = IsolateData(df)

# return prepare_data(__data, 1, cfg.packet_config[0], 1)


# if __name__ == "__main__":
#    print(prepare_data("Dataset1", "packets", 2))
# print(prepare_data("Dataset1", "time_window", 3)
