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
import config as cfg
from tqdm import tqdm
from search_data import search_and_concat, find_csv
from pred_data_preparation import load_pred_data

def Dataset1_exp1_preparation(seg_type):

    keywords = []
    data_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(data_path + "\\" + cfg.Processed_data1 + "\\" + "exp1")  ##******* dataset 1 & exp 2

    total_len = len(cfg.Dataset1_exp1_cfg1) * len(cfg.Dataset1_exp1_content)  ##******* config from dataset 1 for exp 2

    with tqdm(total=100) as pbar:
        for i, cfg_1 in enumerate((cfg.Dataset1_exp1_cfg1)):  ##******* keyword 1 config from dataset 1 for exp 2
            for j, game in enumerate((cfg.Dataset1_exp1_content)):  ##******* keyword 2 config from dataset 1 for exp 2
                keywords.append(cfg_1)
                keywords.append(game)

                status = find_csv(path, "".join(keywords) + "exp1" + ".csv")  ##******* config from dataset 1 for exp 2
                if status == 1:
                    keywords.clear()
                    pbar.update(100 / total_len)
                    continue

                search_and_concat(cfg.Dataset1_path, path, keywords, cfg.Dataset_exp[0])  ##******* exp2 config from dataset 1

                # data = search_and_concat(data_path + "\\" + "Dataset2", keywords, debug)
                # dataRF = IsolateData(data)
                # SaveCSV(path, "".join(keywords) +"exp2", dataRF)
                keywords.clear()
                pbar.update(100 / total_len)

    # if training_method == cfg.training_method[0]:
    return load_pred_data(path, cfg.Dataset_exp[0], cfg.Data_list[0], seg_type)  ##******* exp 2 & dataset 1
    # elif training_method == cfg.training_method[1]:
    #    return load_data_time_window_evr(path,cfg.Dataset_exp[1],cfg.Data_list[0],__type) ##******* exp 2 & dataset 1


