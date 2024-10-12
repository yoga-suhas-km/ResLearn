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
from tqdm import tqdm
from frame_analyzer import FrameAnalyzer
from time_window_segmentation import time_window


def chunks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)


def slice(dfm, chunk_size):
    indices = chunks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)


def load_data_from_packet_segmentation(data, df, seg_size):
    v = []

    df_sp = slice(df, seg_size)

    total_len = len(df_sp)

    with tqdm(total=100) as pbar:
        for i in range(0, len(df_sp)):
            frame_data = []

            if i == 0:
                frame_data = FrameAnalyzer().frame(data, df_sp[i], 1)
            else:
                frame_data = FrameAnalyzer().frame(data, df_sp[i], 0)

            v.append(frame_data)
            pbar.update(100 / total_len)

    df_t = pd.DataFrame(v)
    df_t.columns = ["frame_size", "frame_count", "frame_iat", "frame_duartion"]

    # print(df_t)

    return df_t


def load_data_from_time_window_segmentation(data, df, seg_window):
    v = []

    df_sp = time_window(df, seg_window)

    total_len = len(df_sp)
    with tqdm(total=100) as pbar:
        for i in range(0, len(df_sp)):
            frame_data = []
            if i == 0:
                frame_data = FrameAnalyzer().frame(data, df_sp[i], 1)
            else:
                frame_data = FrameAnalyzer().frame(data, df_sp[i], 0)

            v.append(frame_data)
            pbar.update(100 / total_len)

    df_t = pd.DataFrame(v)
    df_t.columns = ["frame_size", "frame_count", "frame_iat", "frame_duartion"]

    return df_t
