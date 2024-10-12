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
from search_data import find_csv, search_and_concat_multiple
from pred_data_preparation import load_pred_data



def Dataset3_exp1_preparation(seg_type):
    keywords = []
    data_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(data_path + "\\" + cfg.Processed_data3 + "\\" + "exp1")  ##******* dataset 3 & exp 2

    total_len = len(cfg.Dataset3_exp1_cfg1) * len(cfg.Dataset3_exp1_cfg2)
    with tqdm(total=100) as pbar:

        # preparing the Dataset1 data to be used in unison with Dataset3
        for i, cfg1 in enumerate(cfg.Dataset3_exp1_cfg1):
            for j, cfg2 in enumerate(cfg.Dataset3_exp1_cfg2):
                keywords.append(cfg1)
                keywords.append(cfg2)

                status = find_csv(path, "".join(keywords) + "exp1" + ".csv")  ##******* dataset 3 & exp 2
                if status == 1:
                    keywords.clear()
                    pbar.update(100 / total_len)
                    continue

                search_and_concat_multiple(cfg.Dataset3_path, path, keywords, cfg.Dataset3_exp1_usercfg, cfg.Dataset_exp[0])

                keywords.clear()
                pbar.update(100 / total_len)

    return load_pred_data(path, cfg.Dataset_exp[0], cfg.Data_list[2], seg_type)



def Dataset3_exp2_preparation(seg_type):
    keywords = []
    data_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(data_path + "\\" + cfg.Processed_data3 + "\\" + "exp2")  ##******* dataset 3 & exp 2

    total_len = len(cfg.Dataset3_exp2_cfg1)
    with tqdm(total=100) as pbar:

        # preparing the Dataset1 data to be used in unison with Dataset3
        for i, cfg1 in enumerate(cfg.Dataset3_exp2_cfg1):

            keywords.append(cfg1)

            status = find_csv(path, "".join(keywords) + "exp2" + ".csv")  ##******* dataset 3 & exp 2
            if status == 1:
                keywords.clear()
                pbar.update(100 / total_len)
                continue

            search_and_concat_multiple(cfg.Dataset3_path, path, keywords, cfg.Dataset3_exp2_usercfg, cfg.Dataset_exp[1])

            keywords.clear()
            pbar.update(100 / total_len)

    return load_pred_data(path, cfg.Dataset_exp[1], cfg.Data_list[2], seg_type)

