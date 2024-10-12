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

@Authors: Tim Rozer, Yoga Suhas Kuruba Manjunath and Austin Wissborn

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import config as cfg


class FrameAnalyzer:
    def __init__(self):
        self.length_data = []
        self.time_data = []

    def initialize_parameters(self, segment):
        """
        Initialize min_first_packet_length and frame_duration using the first segment.
        """
        global min_first_packet_length
        global frame_duration
        
        self.length_data.extend(segment['Length'].tolist())
        self.time_data.extend(segment['Time'].tolist())
        
        if min_first_packet_length is None:
            min_first_packet_length = self.calculate_min_first_packet_length(segment)
        if frame_duration is None:
            frame_duration = self.calculate_frame_duration(segment)

    def calculate_min_first_packet_length(self, segment):
        """
        Calculate the minimum first packet length based on the maximum length in the segment.
        """
        if isinstance(segment, list):
            try:
                segment = pd.DataFrame(segment)
            except Exception as e:
                raise ValueError(f"Error converting list to DataFrame: {e}")

        if 'Length' not in segment.columns:
            raise ValueError("The segment DataFrame must contain a 'Length' column.")
        
        max_length = segment['Length'].max()
        
        return round(max_length / 4)

    def calculate_frame_duration(self, segment):
        """
        Calculate frame duration based on the difference between the first two largest peaks
        in the PDF of the 'Time' data from the segment.
        """
        if isinstance(segment, list):
            try:
                segment = pd.DataFrame(segment)
            except Exception as e:
                raise ValueError(f"Error converting list to DataFrame: {e}")

        if 'Time' not in segment.columns:
            raise ValueError("The segment DataFrame must contain a 'Time' column.")
        
        times = segment['inter-arrival'].values
        sns.kdeplot(times, bw_adjust=0.5)
        plt.xlabel('Time')
        plt.ylabel('Density')
        
        plot_data = plt.gca().lines[0].get_xydata()
        plt.close()  
        
        peak_indices = np.argsort(plot_data[:, 1])[-2:] 
        peak_times = plot_data[peak_indices, 0]
        peak_times.sort()  
        
        return peak_times[1] - peak_times[0]

    def calculate_iat(self, frames):
        iats = []
        previous_frame_time = None
        
        for j, frame in enumerate(frames):
            if previous_frame_time is not None:
                iat = frames[j]['Time'].iloc[0] - previous_frame_time
                iats.append(iat)
            previous_frame_time = frames[j]['Time'].iloc[0]
        
        if iats:
            average_iat = np.mean(iats)
            min_iat = np.min(iats)
            max_iat = np.max(iats)
            std_iat = np.std(iats)
            var_iat = np.var(iats)
        else:
            average_iat = 0
            min_iat = 0
            max_iat = 0
            std_iat = 0  
            var_iat = 0        
        
        return average_iat, min_iat, max_iat, std_iat, var_iat


    def calculate_frame_duration_variance(self, frames):
        frame_durations = []
        for j in range(1, len(frames)):
            duration = frames[j]['Time'].iloc[0] - frames[j-1]['Time'].iloc[0]
            frame_durations.append(duration)
        return np.var(frame_durations)

    def calculate_frame_size(self, frames):
        frame_size = [frame['Length'].iloc[0] for frame in frames]
        return np.sum(frame_size)
        
    def frame_parameters(self, frames):
        packets_in_frames = [len(frame) for frame in frames]
        time_between_frames_t = []
        packets_not_in_frames_t = []

        for j in range(1, len(frames)):
            prev_frame_end_time = frames[j-1]['Time'].iloc[-1]
            current_frame_start_time = frames[j]['Time'].iloc[0]
            time_between_frames_t.append(current_frame_start_time - prev_frame_end_time)

            prev_frame_end_row = frames[j-1].index[-1]
            current_frame_start_row = frames[j].index[0]
            packets_between = current_frame_start_row - prev_frame_end_row - 1
            packets_not_in_frames_t.append(packets_between)
            
        return (
            np.var(time_between_frames_t),
            np.var(packets_not_in_frames_t),
            np.std(packets_in_frames),
            sum(packets_in_frames)
        )
  
    def convert_to_dataframe(self, segment):

        first_element = segment[0]
        
        if isinstance(first_element, dict):

            df = pd.DataFrame(segment)
        elif isinstance(first_element, list):

            df = pd.DataFrame(segment, columns=[f'col{i}' for i in range(len(first_element))])
        else:
            raise ValueError("Unsupported format: The segment must be a list of dictionaries or lists.")
        
        return df
    
    def frame(self, dataset, segment, start):

        global min_first_packet_length
        global frame_duration
        
        if dataset == cfg.Data_list[0] or dataset == cfg.Data_list[2]:
            min_first_packet_length = 10000
            frame_duration = 0.005
        elif dataset == cfg.Data_list[1]:
            min_first_packet_length = 500
            frame_duration = 0.009
        
        frames = []
        
        start_time = None
        current_frame = pd.DataFrame()
        
        for idx, row in segment.iterrows():
            if start_time is None:
                if row['Length'] >= min_first_packet_length:
                    start_time = row['Time']
                    current_frame = pd.DataFrame([row])
            else:
                if row['Time'] - start_time <= frame_duration:
                    current_frame = pd.concat([current_frame, pd.DataFrame([row])])
                else:
                    frames.append(current_frame)
                    start_time = None
                    current_frame = pd.DataFrame()
                    if row['Length'] >= min_first_packet_length:
                        start_time = row['Time']
                        current_frame = pd.DataFrame([row])
        
        if not current_frame.empty:
            frames.append(current_frame)
        
        average_iat, min_iat, max_iat, std_iat, var_iat = self.calculate_iat(frames)
        frame_durations = self.calculate_frame_duration_variance(frames)
        frame_size = self.calculate_frame_size(frames)
        fr = len(frames)
        
        avg_frame_duration = frame_durations / fr if fr > 0 else 0    
        
        return frame_size,fr, average_iat, frame_durations

    def print_parameters(self):
        """
        Print the minimum first packet length and frame duration.
        """
        min_packet_length_rounded = round(self.min_first_packet_length) if self.min_first_packet_length is not None else None
        frame_duration_rounded = round(self.frame_duration, 3) if self.frame_duration is not None else None
    
        print(f"Minimum First Packet Length: {min_packet_length_rounded}")
        print(f"Frame Duration: {frame_duration_rounded}")


    def plot_packet_arrival_time_pdf(self):
        """
        Plot the PDF of packet arrival times.
        """
        if self.time_data:
            time_data_filtered = [t for t in self.time_data]
            if time_data_filtered:
                sns.histplot(time_data_filtered, kde=True, stat="density", linewidth=0)
                plt.title('PDF of Packet Arrival Times (First 50 ms)')
                plt.xlabel('Time (ms)')
                plt.ylabel('Density')
                plt.show()
            else:
                print("No packet arrival times within the first 50 ms.")
        else:
            print("No time data to plot.")
            
    