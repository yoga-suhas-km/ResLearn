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
required_columns = ["frame_size", "frame_count", "frame_iat"]

# start index
start_index = 0

# training parameters
epochs = 20
verbose = 0
training_portion = 0.6

# segment windows
starting_segment = 1
static_segment_size = 20  # Sets the window size of segments.

# prepare_data integration
Segment_type = ["packets", "time_window"]
Data_list = ["Dataset1", "Dataset2", "Dataset3"]
Dataset_exp = [1, 2, 3, 4, 5, 6]

# ********* Dataset 1 configs ************


# Dataset1_path = r".\Dataset1"

Dataset1_path = r"C:\Users\shimship\LocalDocuments\GitRepos\VR-Network-Traffic-Prediction\Dataset1"
Processed_data1 = "Dataset1_pre_processed"


# ********** Dataset 1 exp1 **************
Dataset1_exp1_cfg1 = ["Cloud"]
Dataset1_exp1_content = ["Bigscreen", "DiRTRally2.0", "RealityMixer", "SolarSystemAR", "VRChat"]  # best results [0.949474 Cloud S_T=50 N=6000] [CLoud_AB 0.932692 S_T=30 N=3000 ] [ cloud_120mbps S_T=50 N= 3000 0.836283][15mbps 0.714286 S_T=50 N= 3000]
Dataset1_end_index = 700

Dataset1_exp1_N = 1000  # size of slice
Dataset1_exp1_Time_Window = 1
Dataset1_frame_size_rolling_over_window = 20
Dataset1_frame_count_rolling_over_window = 22
Dataset1_frame_iat_rolling_over_window = 5

Dataset1_segment_training_windows = 17

# ********* Dataset 2 configs ************

Dataset2_path = r"C:\Users\shimship\LocalDocuments\GitRepos\VR-Network-Traffic-Prediction\Dataset2"
Processed_data2 = "Dataset2_pre_processed"


# ********** Dataset 2 exp1 **************
Dataset2_exp1_cfg1 = ["Cloud"]
Dataset2_exp1_content = ["BeatSaber", "SteamVR"]
Dataset2_exp1_end_index = 800

Dataset2_exp1_N = 1000  # size of slice
Dataset2_exp1_Time_Window = 0.5
Dataset2_exp1_frame_size_rolling_over_window = 20
Dataset2_exp1_frame_count_rolling_over_window = 22
Dataset2_exp1_frame_iat_rolling_over_window = 10

Dataset2_exp1_segment_training_windows = 20

# ********** Dataset 2 exp2 Beatsaber 40mbps**************
Dataset2_exp2_single_csv_location = r"C:\Users\shimship\LocalDocuments\GitRepos\VR-Network-Traffic-Prediction\Dataset2\CloudAdpBitrate\BeatSaber\BeatSaberClim40d5A40H90PC10.64.3.81HMD135.0.132.145.csv"
Dataset2_exp2_end_index = 220

Dataset2_exp2_N = 500  # size of slice
Dataset2_exp2_Time_Window = 0.5
Dataset2_exp2_frame_size_rolling_over_window = 20
Dataset2_exp2_frame_count_rolling_over_window = 22
Dataset2_exp2_frame_iat_rolling_over_window = 5

Dataset2_exp2_segment_training_windows = 5

# ********** Dataset 2 exp3 Beatsaber 54mbps**************
Dataset2_exp3_single_csv_location = r"C:\Users\shimship\LocalDocuments\GitRepos\VR-Network-Traffic-Prediction\Dataset2\CloudAdpBitrate\BeatSaber\BeatSaberClim54A40H90PC10.64.3.81HMD135.0.132.145.csv"
Dataset2_exp3_end_index = 220

Dataset2_exp3_N = 500  # size of slice
Dataset2_exp3_Time_Window = 0.5
Dataset2_exp3_frame_size_rolling_over_window = 20
Dataset2_exp3_frame_count_rolling_over_window = 22
Dataset2_exp3_frame_iat_rolling_over_window = 5

Dataset2_exp3_segment_training_windows = 5

# ********** Dataset 2 exp4 Beatsaber 120mbps**************
Dataset2_exp4_single_csv_location = r"C:\Users\shimship\LocalDocuments\GitRepos\VR-Network-Traffic-Prediction\Dataset2\CloudAdpBitrate\BeatSaber\BeatSaberCNormalA40H90PC10.64.3.81HMD135.0.132.145.csv"
Dataset2_exp4_end_index = 220

Dataset2_exp4_N = 500  # size of slice
Dataset2_exp4_Time_Window = 0.5
Dataset2_exp4_frame_size_rolling_over_window = 20
Dataset2_exp4_frame_count_rolling_over_window = 22
Dataset2_exp4_frame_iat_rolling_over_window = 5

Dataset2_exp4_segment_training_windows = 5

# ********** Dataset 2 exp5 steamvr 40mbps**************
Dataset2_exp5_single_csv_location = r"C:\Users\shimship\LocalDocuments\GitRepos\VR-Network-Traffic-Prediction\Dataset2\CloudAdpBitrate\SteamVRHome\VRhomeClim40d5A40H90PC10.64.3.81HMD135.0.132.145.csv"
Dataset2_exp5_end_index = 220

Dataset2_exp5_N = 500  # size of slice
Dataset2_exp5_Time_Window = 0.5
Dataset2_exp5_frame_size_rolling_over_window = 20
Dataset2_exp5_frame_count_rolling_over_window = 22
Dataset2_exp5_frame_iat_rolling_over_window = 5

Dataset2_exp5_segment_training_windows = 5

# ********** Dataset 2 exp6 steamvr 54mbps**************
Dataset2_exp6_single_csv_location = r"C:\Users\shimship\LocalDocuments\GitRepos\VR-Network-Traffic-Prediction\Dataset2\CloudAdpBitrate\SteamVRHome\VRhomeClim54A40H90PC10.64.3.81HMD135.0.132.145.csv"
Dataset2_exp6_end_index = 220

Dataset2_exp6_N = 500  # size of slice
Dataset2_exp6_Time_Window = 0.5
Dataset2_exp6_frame_size_rolling_over_window = 20
Dataset2_exp6_frame_count_rolling_over_window = 22
Dataset2_exp6_frame_iat_rolling_over_window = 5

Dataset2_exp6_segment_training_windows = 5


# ********** Dataset 3 congifs ***********#

# group1 and slow_traffic == BeatSaber
# group2 and slow_traffic == Medal of honor
# group1 and fast_traffic == Cooking sim
# group2 and fast)traffic == Forklifft sim

# Table of corosponding apps:
#              | group1      | group2
# -----------------------------------------
# slow_traffic | Beast Saber | Medal of honor
# -----------------------------------------
# fast_traffic | Cooking Sim.| Forklift Sim.

Dataset3_path = r"C:\Users\shimship\LocalDocuments\GitRepos\VR-Network-Traffic-Prediction\Dataset3"
Processed_data3 = "Dataset3_pre_processed"


# ********** Dataset 3 exp1 **************


Dataset3_exp1_cfg1 = ["group1", "group2"]  # Differentiating between the two group. Logical equivilant
Dataset3_exp1_cfg2 = ["slow_traffic", "fast_traffic"]  # Differentiation between each application for each group
Dataset3_exp1_usercfg = ["user11"]  # list of users to consider

Dataset3_exp1_N = 12000  # size of slice
Dataset3_exp1_Time_Window = 0.5
Dataset3_end_index = 300

Dataset3_exp1_frame_size_rolling_over_window = 25
Dataset3_exp1_frame_count_rolling_over_window = 20
Dataset3_exp1_frame_iat_rolling_over_window = 5

Dataset3_exp1_segment_training_windows = 8




# ********** Dataset 3 exp2 ****************************


Dataset3_exp2_cfg1 = ["slow_traffic", "fast_traffic"]
Dataset3_exp2_usercfg = ["user11"]

Dataset3_exp2_N = 12000  # size of slice
Dataset3_exp2_Time_Window = 0.5

Dataset3_exp2_frame_size_rolling_over_window = 25
Dataset3_exp2_frame_count_rolling_over_window = 20
Dataset3_exp2_frame_iat_rolling_over_window = 5

Dataset3_exp2_segment_training_windows = 8

