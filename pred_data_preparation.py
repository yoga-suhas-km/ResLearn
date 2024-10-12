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
import config as cfg
from tqdm import tqdm
from segmentation_processing import load_data_from_packet_segmentation, load_data_from_time_window_segmentation


def chunks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)


def slice(dfm, chunk_size):
    indices = chunks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)


def load_pred_data(data_path, exp, __data, seg_type):

    D = []
    ntk_data = []

    path, dirs, files = next(os.walk(data_path))

    keywords = []
    req_files = []

    # analyzer = FrameAnalyzer()

    if __data == cfg.Data_list[0]:  # ****************** configs for Dataset 1

        if exp == cfg.Dataset_exp[0]:  # ************** Files for exp 2
            total_len = len(cfg.Dataset1_exp1_cfg1) * len(cfg.Dataset1_exp1_content)

            for i, cfg_1 in enumerate((cfg.Dataset1_exp1_cfg1)):
                for j, game in enumerate((cfg.Dataset1_exp1_content)):
                    keywords.append(cfg_1)
                    keywords.append(game)
                    file_name = "".join(keywords) + "exp1" + ".csv"
                    req_files.append(file_name)
                    keywords.clear()



    elif __data == cfg.Data_list[1]:  # ****************** configs for Dataset 2

        if exp == cfg.Dataset_exp[0]:  # ************** Files for exp 1
            total_len = len(cfg.Dataset2_exp1_cfg1) * len(cfg.Dataset2_exp1_content)

            for i, config_t in enumerate((cfg.Dataset2_exp1_cfg1)):
                for j, game in enumerate((cfg.Dataset2_exp1_content)):
                    keywords.append(config_t)
                    keywords.append(game)
                    file_name = "".join(keywords) + "exp1" + ".csv"
                    req_files.append(file_name)
                    keywords.clear()


    elif __data == cfg.Data_list[2]:  # ****************** configs for Dataset 3


        if exp == cfg.Dataset_exp[0]:  # ************** Files for exp 2

            total_len = len(cfg.Dataset3_exp1_cfg1) * len(cfg.Dataset3_exp1_cfg2)
            # preparing the Dataset1 data to be used in unison with Dataset3
            for i, cfg1 in enumerate(cfg.Dataset3_exp1_cfg1):
                for j, cfg2 in enumerate(cfg.Dataset3_exp1_cfg2):
                    keywords.append(cfg1)
                    keywords.append(cfg2)

                    file_name = "".join(keywords) + "exp1" + ".csv"
                    req_files.append(file_name)
                    keywords.clear()

        if exp == cfg.Dataset_exp[1]:  # ************** Files for exp 3

            total_len = len(cfg.Dataset3_exp2_cfg1)
            # preparing the Dataset1 data to be used in unison with Dataset3
            for i, cfg1 in enumerate(cfg.Dataset3_exp2_cfg1):

                keywords.append(cfg1)

                file_name = "".join(keywords) + "exp2" + ".csv"
                req_files.append(file_name)
                keywords.clear()

    # print("coming to 3\n")
    with tqdm(total=100) as pbar:
        # print(req_files)
        for x, file in enumerate(req_files):

            frame_segment = []

            df = pd.read_csv(os.path.join(data_path, file))
            df = df.dropna()

            D.append(df)

            pbar.update(100 / total_len)

    df_ct = pd.concat(D)

    df_ct = df_ct.sort_values(by=df_ct.columns[0]).reset_index(drop=True)

    if seg_type == cfg.Segment_type[0]:

        if __data == cfg.Data_list[0]:  # ****************** configs for Dataset 1
            if exp == cfg.Dataset_exp[0]:  # **************  for exp 1
                df_t = load_data_from_packet_segmentation(cfg.Data_list[0], df, cfg.Dataset1_exp1_N)

        elif __data == cfg.Data_list[1]:  # ****************** configs for Dataset 2
            if exp == cfg.Dataset_exp[0]:  # **************  for exp 1
                df_t = load_data_from_packet_segmentation(cfg.Data_list[1], df, cfg.Dataset2_exp1_N)


        elif __data == cfg.Data_list[2]:  # ****************** configs for Dataset 3
            if exp == cfg.Dataset_exp[0]:  # **************  for exp 1
                df_t = load_data_from_packet_segmentation(cfg.Data_list[2], df, cfg.Dataset3_exp1_N)
            elif exp == cfg.Dataset_exp[1]:  # **************  for exp 2
                df_t = load_data_from_packet_segmentation(cfg.Data_list[2], df, cfg.Dataset3_exp2_N)


    elif seg_type == cfg.Segment_type[1]:

        if __data == cfg.Data_list[0]:  # ****************** configs for Dataset 1
            if exp == cfg.Dataset_exp[0]:  # **************  for exp 1
                df_t = load_data_from_time_window_segmentation(cfg.Data_list[0], df, cfg.Dataset1_exp1_Time_Window)


        elif __data == cfg.Data_list[1]:  # ****************** configs for Dataset 2
            if exp == cfg.Dataset_exp[0]:  # **************  for exp 1
                df_t = load_data_from_time_window_segmentation(cfg.Data_list[1], df, cfg.Dataset2_exp1_Time_Window)


        elif __data == cfg.Data_list[2]:  # ****************** configs for Dataset 3
            if exp == cfg.Dataset_exp[0]:  # **************  for exp 1
                df_t = load_data_from_time_window_segmentation(cfg.Data_list[2], df, cfg.Dataset3_exp1_Time_Window)
            elif exp == cfg.Dataset_exp[1]:  # **************  for exp 2
                df_t = load_data_from_time_window_segmentation(cfg.Data_list[2], df, cfg.Dataset3_exp2_Time_Window)


    return df_t
