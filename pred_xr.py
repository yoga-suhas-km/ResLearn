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

import argparse
import os
import pandas as pd
from tensorflow.keras.models import load_model
import shutil
from extract_raw_features import isolate_data

from config_data import config_data, format_X_Y
from train import train_model, train_ensemble_model
from plotter import plot_data_and_prediction, plot_data_train_val, plot_metric_segments, plot_dataframe
from metrics import rmse, mape, smape
import models
import config
from data_processing import prepare_data
from segmentation_processing import load_data_from_packet_segmentation, load_data_from_time_window_segmentation
from rolling_over_window import rolling_over_window
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def check_and_load_model(model_name: str):
    # Construct the file path
    save_directory = "models"
    model_path = os.path.join(save_directory, model_name + ".h5")

    # Check if the model exists
    if os.path.exists(model_path):
        # Load the model
        return load_model(model_path)
    else:
        return False


def isolate_df_index(df, static_segment_size, sliding_window_step):
    df = df[config.start_index :]
    shift_window = sliding_window_step - 1
    start_index = static_segment_size * shift_window
    end_index = static_segment_size + static_segment_size * shift_window

    new_df = df[start_index:end_index]
    index_out_of_range = new_df.shape[0] != static_segment_size
    if index_out_of_range:
        print("Index out of range")
    return new_df, index_out_of_range


def save_metrics_to_file(df, model_name):
    # Create test_error_of_models.txt and save segment_mape under model_name
    error_file = "metrics/test_error_of_models.txt"
    os.makedirs(os.path.dirname(error_file), exist_ok=True)

    if not os.path.exists(error_file):
        with open(error_file, "w") as ef:
            ef.write("")

    # Read existing data
    with open(error_file, "r") as ef:
        lines = ef.readlines()

    lines.append(f"{model_name}: {df['segment_smape'].tolist()}\n")

    with open(error_file, "w") as ef:
        ef.writelines(lines)

    output_file = f"metrics/metrics_{model_name}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Define the lambda function as a variable
    flatten_single_element_list = lambda x: x[0] if isinstance(x, list) else x

    # Apply the lambda function to the relevant columns
    df["segment_rmse"] = df["segment_rmse"].apply(flatten_single_element_list)
    df["segment_mape"] = df["segment_mape"].apply(flatten_single_element_list)
    df["segment_smape"] = df["segment_smape"].apply(flatten_single_element_list)

    with open(output_file, "w") as f:
        for index, row in df.iterrows():
            f.write(f"Segment: {index}\n")
            f.write(f"Start Index: {row['start_index']}\n")
            f.write(f"Stop Index: {row['stop_index']}\n")
            if pd.notna(row.get("train_rmse")):
                f.write(f"Train RMSE: {row['train_rmse']}\n")
                f.write(f"Val RMSE: {row['val_rmse']}\n")
                f.write(f"Train MAPE: {row['train_mape']}\n")
                f.write(f"Val MAPE: {row['val_mape']}\n")
                f.write(f"Train SMAPE: {row['train_smape']}\n")
                f.write(f"Val SMAPE: {row['val_smape']}\n")
            if pd.notna(row.get("segment_rmse")):
                f.write(f"RMSE: {row['segment_rmse']}\n")
                f.write(f"MAPE: {row['segment_mape']}\n")
                f.write(f"SMAPE: {row['segment_smape']}\n")
            f.write("\n")

        # Calculate the averages of non-null entries
        avg_rmse = df["segment_rmse"].dropna().mean()
        avg_mape = df["segment_mape"].dropna().mean()
        avg_smape = df["segment_smape"].dropna().mean()

        # Append the averages to the file
        f.write("Averages of testing segments:\n")
        f.write(f"Average RMSE: {avg_rmse}\n")
        f.write(f"Average MAPE: {avg_mape}\n")
        f.write(f"Average SMAPE: {avg_smape}\n")

        print("\n************Averages of testing\n")
        print(f"Average RMSE: {avg_rmse}\n")
        print(f"Average MAPE: {avg_mape}\n")
        print(f"Average SMAPE: {avg_smape}\n")


def read_error_file_to_df(filepath):
    # Initialize an empty dictionary to hold the data
    data = {}

    # Read the file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Parse each line and update the dictionary
    for line in lines:
        if line.strip():
            model_name, values = line.split(": ")
            values = eval(values)
            data[model_name] = values

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data, orient="index").transpose()

    return df


def load_predicted_data(column):
    try:
        df = pd.read_csv("predicted/predicted_data.csv", index_col=0)
        df[column] = None
    except FileNotFoundError:
        os.makedirs("predicted", exist_ok=True)
        df = pd.DataFrame(columns=["segment", "data", column])
    return df


def main():
    parser = argparse.ArgumentParser()
    # deletes all files in metrics, plots and models folders
    parser.add_argument("--clean", default=False, help="Options: True or False")

    parser.add_argument("--data", default="Dataset1", help="Dataset1, Dataset2 or Dataset3")
    parser.add_argument("--feature", default="frame_size", help="frame_size, frame_count, frame_iat")
    parser.add_argument("--experiment", default="1", help="1 or 2")
    parser.add_argument("--residual_learning", default=False, help="Options: True or False")

    parser.add_argument("--model", default="lstm", help="Options: lstm")
    parser.add_argument("--plot_test_errors_of_models", default=False, help="Options: True or False")

    parser.add_argument("--plot_segment_predictions", default="0", help="choose segment to plot")
    parser.add_argument("--plot_total_predictions", default=False, help="Options: True or False")

    args = parser.parse_args()

    time_modifier = 1
    single_csv = 0

    if int(args.residual_learning):
        residual_learning = 1
    else:
        residual_learning = 0

    
    # Dataset 1 
    if (args.data == config.Data_list[0]) and (args.feature == config.required_columns[0]):
        data_name = config.Data_list[0]
        df_end_index = config.Dataset1_end_index
        experiment = config.Dataset_exp[0]
        feature = config.required_columns[0]
        rolling_over_window_config = config.Dataset1_frame_size_rolling_over_window
        segment_training_windows = config.Dataset1_segment_training_windows
    elif (args.data == config.Data_list[0]) and (args.feature == config.required_columns[1]):
        data_name = config.Data_list[0]
        df_end_index = config.Dataset1_end_index
        experiment = config.Dataset_exp[0]
        feature = config.required_columns[1]
        rolling_over_window_config = config.Dataset1_frame_count_rolling_over_window
        segment_training_windows = config.Dataset1_segment_training_windows      
    elif (args.data == config.Data_list[0]) and (args.feature == config.required_columns[2]):
        data_name = config.Data_list[0]
        df_end_index = config.Dataset1_end_index
        experiment = config.Dataset_exp[0]
        feature = config.required_columns[2] 
        time_modifier = 1000        
        rolling_over_window_config = config.Dataset1_frame_iat_rolling_over_window
        segment_training_windows = config.Dataset1_segment_training_windows  

    # Dataset 2, we need to add single csv experiments
    if (args.data == config.Data_list[1]):
        #exp 1
        if int(args.experiment) == int(config.Dataset_exp[0]) and (args.feature == config.required_columns[0]):
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp1_end_index
            experiment = config.Dataset_exp[0]
            feature = config.required_columns[0]        
            rolling_over_window_config = config.Dataset2_exp1_frame_size_rolling_over_window
            segment_training_windows = config.Dataset2_exp1_segment_training_windows
        elif int(args.experiment) == int(config.Dataset_exp[0]) and (args.feature == config.required_columns[1]):
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp1_end_index
            experiment = config.Dataset_exp[0]
            feature = config.required_columns[1] 
            rolling_over_window_config = config.Dataset2_exp1_frame_count_rolling_over_window
            segment_training_windows = config.Dataset2_exp1_segment_training_windows
        elif int(args.experiment) == int(config.Dataset_exp[0]) and (args.feature == config.required_columns[2]):
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp1_end_index
            experiment = config.Dataset_exp[0]
            time_modifier = 1000 
            feature = config.required_columns[2]         
            rolling_over_window_config = config.Dataset2_exp1_frame_iat_rolling_over_window
            segment_training_windows = config.Dataset2_exp1_segment_training_windows     
        #exp 2
        elif int(args.experiment) == int(config.Dataset_exp[1]) and (args.feature == config.required_columns[0]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp2_single_csv_location
            segment_size = config.Dataset2_exp2_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp2_end_index
            experiment = config.Dataset_exp[1]
            feature = config.required_columns[0]         
            rolling_over_window_config = config.Dataset2_exp2_frame_size_rolling_over_window
            segment_training_windows = config.Dataset2_exp2_segment_training_windows 
        elif int(args.experiment) == int(config.Dataset_exp[1]) and (args.feature == config.required_columns[1]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp2_single_csv_location
            segment_size = config.Dataset2_exp2_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp2_end_index
            experiment = config.Dataset_exp[1]
            feature = config.required_columns[1]         
            rolling_over_window_config = config.Dataset2_exp2_frame_count_rolling_over_window
            segment_training_windows = config.Dataset2_exp2_segment_training_windows        
        elif int(args.experiment) == int(config.Dataset_exp[1]) and (args.feature == config.required_columns[2]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp2_single_csv_location
            segment_size = config.Dataset2_exp2_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp3_end_index
            experiment = config.Dataset_exp[1]
            feature = config.required_columns[2]  
            time_modifier = 1000            
            rolling_over_window_config = config.Dataset2_exp2_frame_iat_rolling_over_window
            segment_training_windows = config.Dataset2_exp2_segment_training_windows  
        # exp 3
        elif int(args.experiment) == int(config.Dataset_exp[2]) and (args.feature == config.required_columns[0]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp3_single_csv_location
            segment_size = config.Dataset2_exp3_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp3_end_index
            experiment = config.Dataset_exp[2]
            feature = config.required_columns[0]             
            rolling_over_window_config = config.Dataset2_exp3_frame_size_rolling_over_window
            segment_training_windows = config.Dataset2_exp3_segment_training_windows 
        elif int(args.experiment) == int(config.Dataset_exp[2]) and (args.feature == config.required_columns[1]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp3_single_csv_location
            segment_size = config.Dataset2_exp3_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp3_end_index
            experiment = config.Dataset_exp[2]
            feature = config.required_columns[1]           
            rolling_over_window_config = config.Dataset2_exp3_frame_count_rolling_over_window
            segment_training_windows = config.Dataset2_exp3_segment_training_windows             
        elif int(args.experiment) == int(config.Dataset_exp[2]) and (args.feature == config.required_columns[2]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp3_single_csv_location
            segment_size = config.Dataset2_exp3_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp3_end_index
            experiment = config.Dataset_exp[2]
            feature = config.required_columns[2]   
            time_modifier = 1000              
            rolling_over_window_config = config.Dataset2_exp3_frame_iat_rolling_over_window
            segment_training_windows = config.Dataset2_exp3_segment_training_windows             
        # exp 4
        elif int(args.experiment) == int(config.Dataset_exp[3]) and (args.feature == config.required_columns[0]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp4_single_csv_location
            segment_size = config.Dataset2_exp4_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp4_end_index
            experiment = config.Dataset_exp[3]
            feature = config.required_columns[0]             
            rolling_over_window_config = config.Dataset2_exp4_frame_size_rolling_over_window
            segment_training_windows = config.Dataset2_exp4_segment_training_windows 
        elif int(args.experiment) == int(config.Dataset_exp[3]) and (args.feature == config.required_columns[1]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp4_single_csv_location
            segment_size = config.Dataset2_exp4_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp4_end_index
            experiment = config.Dataset_exp[3]
            feature = config.required_columns[1]           
            rolling_over_window_config = config.Dataset2_exp4_frame_count_rolling_over_window
            segment_training_windows = config.Dataset2_exp4_segment_training_windows             
        elif int(args.experiment) == int(config.Dataset_exp[3]) and (args.feature == config.required_columns[2]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp4_single_csv_location
            segment_size = config.Dataset2_exp4_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp4_end_index
            experiment = config.Dataset_exp[3]
            feature = config.required_columns[2]   
            time_modifier = 1000              
            rolling_over_window_config = config.Dataset2_exp4_frame_iat_rolling_over_window
            segment_training_windows = config.Dataset2_exp4_segment_training_windows 
        # exp 5
        elif int(args.experiment) == int(config.Dataset_exp[4]) and (args.feature == config.required_columns[0]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp5_single_csv_location
            segment_size = config.Dataset2_exp5_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp5_end_index
            experiment = config.Dataset_exp[4]
            feature = config.required_columns[0]             
            rolling_over_window_config = config.Dataset2_exp5_frame_size_rolling_over_window
            segment_training_windows = config.Dataset2_exp5_segment_training_windows 
        elif int(args.experiment) == int(config.Dataset_exp[4]) and (args.feature == config.required_columns[1]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp5_single_csv_location
            segment_size = config.Dataset2_exp5_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp5_end_index
            experiment = config.Dataset_exp[4]
            feature = config.required_columns[1]           
            rolling_over_window_config = config.Dataset2_exp5_frame_count_rolling_over_window
            segment_training_windows = config.Dataset2_exp5_segment_training_windows             
        elif int(args.experiment) == int(config.Dataset_exp[4]) and (args.feature == config.required_columns[2]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp4_single_csv_location
            segment_size = config.Dataset2_exp5_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp5_end_index
            experiment = config.Dataset_exp[4]
            feature = config.required_columns[2]   
            time_modifier = 1000              
            rolling_over_window_config = config.Dataset2_exp5_frame_iat_rolling_over_window
            segment_training_windows = config.Dataset2_exp5_segment_training_windows 
        # exp 6
        elif int(args.experiment) == int(config.Dataset_exp[5]) and (args.feature == config.required_columns[0]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp6_single_csv_location
            segment_size = config.Dataset2_exp5_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp6_end_index
            experiment = config.Dataset_exp[5]
            feature = config.required_columns[0]             
            rolling_over_window_config = config.Dataset2_exp6_frame_size_rolling_over_window
            segment_training_windows = config.Dataset2_exp6_segment_training_windows 
        elif int(args.experiment) == int(config.Dataset_exp[5]) and (args.feature == config.required_columns[1]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp6_single_csv_location
            segment_size = config.Dataset2_exp6_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp6_end_index
            experiment = config.Dataset_exp[5]
            feature = config.required_columns[1]           
            rolling_over_window_config = config.Dataset2_exp6_frame_count_rolling_over_window
            segment_training_windows = config.Dataset2_exp6_segment_training_windows             
        elif int(args.experiment) == int(config.Dataset_exp[5]) and (args.feature == config.required_columns[2]):
            single_csv = 1
            single_csv_location = config.Dataset2_exp6_single_csv_location
            segment_size = config.Dataset2_exp6_N
            data_name = config.Data_list[1]
            df_end_index = config.Dataset2_exp6_end_index
            experiment = config.Dataset_exp[5]
            feature = config.required_columns[2]   
            time_modifier = 1000              
            rolling_over_window_config = config.Dataset2_exp6_frame_iat_rolling_over_window
            segment_training_windows = config.Dataset2_exp6_segment_training_windows 


    # Dataset 3 exp1
    if (args.data == config.Data_list[2]):    
        if args.feature == config.required_columns[0] and int(args.experiment) == int(config.Dataset_exp[0]):
            data_name = config.Data_list[2]
            df_end_index = config.Dataset3_end_index
            experiment = config.Dataset_exp[0]
            feature = config.required_columns[0]   
            rolling_over_window_config = config.Dataset3_exp1_frame_size_rolling_over_window
            segment_training_windows = config.Dataset3_exp1_segment_training_windows 
        elif args.feature == config.required_columns[1] and int(args.experiment) == int(config.Dataset_exp[0]):
            data_name = config.Data_list[2]
            df_end_index = config.Dataset3_end_index
            experiment = config.Dataset_exp[0]
            feature = config.required_columns[1]   
            rolling_over_window_config = config.Dataset3_exp1_frame_count_rolling_over_window   
            segment_training_windows = config.Dataset3_exp1_segment_training_windows         
        elif args.feature == config.required_columns[2] and int(args.experiment) == int(config.Dataset_exp[0]):
            data_name = config.Data_list[2]
            df_end_index = config.Dataset3_end_index
            experiment = config.Dataset_exp[0]
            feature = config.required_columns[2]   
            time_modifier = 1000 
            rolling_over_window_config = config.Dataset3_exp1_frame_iat_rolling_over_window    
            segment_training_windows = config.Dataset3_exp1_segment_training_windows         

    # Dataset 3 exp2
    if (args.data == config.Data_list[2]):
        if (args.feature == config.required_columns[0]) and (int(args.experiment) == int(config.Dataset_exp[1])):
            data_name = config.Data_list[2]
            df_end_index = config.Dataset3_end_index
            experiment = config.Dataset_exp[1]
            feature = config.required_columns[0]   
            rolling_over_window_config = config.Dataset3_exp2_frame_size_rolling_over_window
            segment_training_windows = config.Dataset3_exp2_segment_training_windows 
        elif (args.feature == config.required_columns[1]) and (int(args.experiment) == int(config.Dataset_exp[1])):
            data_name = config.Data_list[2]
            df_end_index = config.Dataset3_end_index
            experiment = config.Dataset_exp[1]
            feature = config.required_columns[1]   
            rolling_over_window_config = config.Dataset3_exp2_frame_count_rolling_over_window      
            segment_training_windows = config.Dataset3_exp2_segment_training_windows 
        elif (args.feature == config.required_columns[2]) and (int(args.experiment) == int(config.Dataset_exp[1])):
            data_name = config.Data_list[2]
            df_end_index = config.Dataset3_end_index
            experiment = config.Dataset_exp[1]
            feature = config.required_columns[2]        
            time_modifier = 1000             
            rolling_over_window_config = config.Dataset3_exp2_frame_iat_rolling_over_window     
            segment_training_windows = config.Dataset3_exp2_segment_training_windows 

    print("************ Configs selected")
    print("data_name",data_name)
    print("df_end_index",df_end_index)
    print("experiment",experiment)
    print("feature",feature)
    print("rolling_over_window_config",rolling_over_window_config)
    print("segment_training_windows",segment_training_windows)


    if args.clean:
        folders_to_delete = ["metrics", "plots", "models", "predicted"]
        for folder in folders_to_delete:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        return

    if args.plot_segment_predictions != "0":
        df = pd.read_csv("predicted/predicted_data.csv", index_col=0)
        df = df[df["segment"] == int(args.plot_segment_predictions)].drop(columns=["segment"])
        plot_dataframe(df, "all_metrics_for_segment_" + args.plot_segment_predictions, title="Model Error Comparison for Segment " + args.plot_segment_predictions)
        return

    if args.plot_total_predictions:
        df = pd.read_csv("predicted/predicted_data.csv", index_col=0)
        df = df.drop(columns=["segment"])
        plot_dataframe(df, "all_metrics_for_all_segments", title="Model Error Comparison for Total")
        return

    if args.plot_test_errors_of_models:
        df = read_error_file_to_df("metrics/test_error_of_models.txt")
        # Preprocess the DataFrame to extract the first element from lists and convert None to NaN
        df_processed = df.map(lambda y: y[0] if isinstance(y, list) else None)
        plot_dataframe(df_processed, "plot_test_errors_of_models", x_label="Index (Segment - 1)")
        return

    if single_csv:
        df = pd.read_csv(single_csv_location)
        df = isolate_data(df)
        df = load_data_from_packet_segmentation(data_name, df, segment_size)
        df = df[feature] * time_modifier
        print(df)
    else:
        df = prepare_data(data_name, "packets", experiment)[feature] * time_modifier

    # aggregate data
    df = (rolling_over_window(df, rolling_over_window_config))

    df = pd.DataFrame(df[:df_end_index])
 
 
    metrics_columns = ["segment", "start_index", "stop_index", "train_rmse", "val_rmse", "train_mape", "val_mape", "train_smape", "val_smape", "segment_rmse", "segment_mape", "segment_smape"]
    metrics = pd.DataFrame(columns=metrics_columns)

    if args.model == "lstm":
        model = models.get_lstm(1, df.shape[1])
    elif args.model == "stacked_lstm":
        model = models.get_stacked_lstm(1, df.shape[1])
    elif args.model == "transformer":
        model = models.get_transformer(1, df.shape[1])
    elif args.model == "gru":
        model = models.get_gru(1, df.shape[1])
    elif args.model == "cnn_lstm":
        model = models.get_cnn_lstm(1, df.shape[1])
    elif args.model == "rnn":
        model = models.get_rnn(1, df.shape[1])
    elif args.model == "ensemble":
        model = models.get_ensemble(df.shape[1])
    else:
        raise ValueError("Invalid model input")
    # compile the model
    model.compile(loss="mean_squared_error", optimizer="adam")

    if residual_learning:
        # residual model
        model_dense = models.get_dense(1, df.shape[1])
        model_dense.compile(loss="mean_squared_error", optimizer="adam")

    predicted_column_name = args.model + "_predicted"
    predicted_data_df = load_predicted_data(predicted_column_name)

    starting_segment = config.starting_segment - 1  # 0 indexing

    # window calculation
    index_range = df.index[-1] - df.index[0]
    #print("df index range: ", index_range)
    total_segments = int(index_range / config.static_segment_size)
    #print("total segments: ", total_segments)
    segment_testing_windows = total_segments - segment_training_windows
    #print("segment testing windows: ", segment_testing_windows)
    if segment_testing_windows < 0:
        raise ValueError("Segment testing windows is less than 0, static_segment_size is too large or too many testing windows")
        return

    for segment in range(1 + starting_segment, starting_segment + segment_training_windows + 1):
        df_temp, index_out_of_range = isolate_df_index(df, config.static_segment_size, segment)
        index = df_temp.index
        if index_out_of_range:
            print("At segment ", segment, index)
            break
        X_train, y_train, X_val, y_val, scaler_train, scaler_val, dataset = config_data(df_temp, 1)

        if args.model == "ensemble":
            train_ensemble_model(model, X_train, y_train)
        else:
            train_model(model, X_train, y_train, args.model)

        if args.model == "ensemble":
            if type(model.input) == list:
                X = [X_train for _ in range(len(model.input))]
            else:
                X = [X_train]
            y_train_predict = model.predict(X)
            if type(model.input) == list:
                X = [X_val for _ in range(len(model.input))]
            else:
                X = [X_val]
            y_val_predict = model.predict(X)
        else:
            y_train_predict = model.predict(X_train)
            y_val_predict = model.predict(X_val)

        if residual_learning:
            y_train_residuals = y_train - y_train_predict.reshape(y_train_predict.shape[0], y_train_predict.shape[1])
            y_train_residuals = y_train_residuals + abs(min(y_train_residuals))

            y_val_residuals = y_val - y_val_predict.reshape(y_val_predict.shape[0], y_val_predict.shape[1])
            y_val_residuals = y_val_residuals

            X_train_res = y_train_residuals.reshape(y_train_residuals.shape[0], y_train_residuals.shape[1], 1)
            X_val_res = y_val_residuals.reshape(y_val_residuals.shape[0], y_val_residuals.shape[1], 1)

            train_model(model_dense, X_train_res, y_train_residuals, "residual")

            y_train_residual_predict = model_dense.predict(X_train_res)
            y_val_residual_predict = model_dense.predict(X_val_res)

            y_train_predict = y_train_predict + y_train_residual_predict
            y_val_predict = y_val_predict + y_val_residual_predict

        # invert predictions
        y_train_predict = scaler_train.inverse_transform(y_train_predict)
        y_train = scaler_train.inverse_transform(y_train)
        y_val_predict = scaler_val.inverse_transform(y_val_predict)
        y_val = scaler_val.inverse_transform(y_val)

        train_rmse = rmse(y_train, y_train_predict)
        val_rmse = rmse(y_val, y_val_predict)
        train_mape = mape(y_train, y_train_predict)
        val_mape = mape(y_val, y_val_predict)
        train_smape = smape(y_train, y_train_predict)
        val_smape = smape(y_val, y_val_predict)

        new_metrics = pd.DataFrame(
            [
                {
                    "segment": segment,
                    "start_index": index[0],
                    "stop_index": index[-1],
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "train_mape": train_mape,
                    "val_mape": val_mape,
                    "train_smape": train_smape,
                    "val_smape": val_smape,
                    "segment_rmse": None,
                    "segment_mape": None,
                    "segment_smape": None,
                }
            ]
        )
        metrics = pd.concat([metrics, new_metrics], ignore_index=True)

        print("\n************Train and Validation Segment:", segment)
        print("Train RMSE:", train_rmse)
        print("Train MAPE:", train_mape)
        print("Train SMAPE:", train_smape)

        print("Val RMSE:", val_rmse)
        print("Val MAPE:", val_mape)
        print("Val SMAPE:", val_smape)

        plot_data_train_val(y_train_predict, y_val_predict, dataset, index, 1, df_temp.columns, args.model, segment)

    for segment in range(starting_segment + segment_training_windows + 1, starting_segment + (segment_training_windows + segment_testing_windows) + 1):
        df_temp, index_out_of_range = isolate_df_index(df, config.static_segment_size, segment)
        index = df_temp.index
        if index_out_of_range:
            print("At segment ", segment, index)
            break

        X, y, dataset, scaler = format_X_Y(df_temp, 1)

        model = check_and_load_model(args.model)
        if residual_learning:
            model_res = check_and_load_model("residual")

        if args.model == "ensemble":
            if type(model.input) == list:
                X_ensemble = [X for _ in range(len(model.input))]
            else:
                X_ensemble = [X]
            y_predict = model.predict(X_ensemble)
        else:
            y_predict = model.predict(X)

            if residual_learning:
                y_error = y - y_predict
                y_error = y_error.reshape(y_error.shape[0], y_error.shape[1], 1)
                y_res = model_res.predict(y_error)
                y_predict = y_predict + y_res

        # invert predictions
        y_predict = scaler.inverse_transform(y_predict)
        y = scaler.inverse_transform(y)

        # implement prediction recording
        try:
            predicted_data_df.loc[index[0] + 1 : index[-1], predicted_column_name] = y_predict[:, 0]
            predicted_data_df.loc[index[0] + 1 : index[-1], "data"] = y[:, 0]
            predicted_data_df.loc[index[0] + 1 : index[-1], "segment"] = segment
        except:
            additional_rows = pd.DataFrame(index=index, columns=predicted_data_df.columns)
            additional_rows.loc[index[0] + 1 : index[-1], predicted_column_name] = y_predict[:, 0]
            additional_rows.loc[index[0] + 1 : index[-1], "data"] = y[:, 0]
            additional_rows["segment"] = segment
            predicted_data_df = pd.concat([predicted_data_df, additional_rows])

        segment_rmse = rmse(y, y_predict)
        segment_mape = mape(y, y_predict)
        segment_smape = smape(y, y_predict)

        new_metrics = pd.DataFrame(
            [
                {
                    "segment": segment,
                    "start_index": index[0],
                    "stop_index": index[-1],
                    "train_rmse": None,
                    "val_rmse": None,
                    "train_mape": None,
                    "val_mape": None,
                    "train_smape": None,
                    "val_smape": None,
                    "segment_rmse": segment_rmse,
                    "segment_mape": segment_mape,
                    "segment_smape": segment_smape,
                }
            ]
        )
        metrics = pd.concat([metrics, new_metrics], ignore_index=True)

        print("\n************Test Segment:", segment)
        print("RMSE:", segment_rmse)
        print("MAPE:", segment_mape)
        print("SMAPE:", segment_smape)

        plot_data_and_prediction(y_predict, dataset, index, 1, df_temp.columns, args.model, segment)

    metrics.set_index("segment", inplace=True)

    # Plot metrics
    plot_metric_segments(metrics, "mape")
    plot_metric_segments(metrics, "rmse")
    plot_metric_segments(metrics, "smape")

    predicted_data_df.to_csv("predicted/predicted_data.csv")
    save_metrics_to_file(metrics, args.model)


if __name__ == "__main__":
    main()
