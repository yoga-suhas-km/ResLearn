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
from __future__ import print_function
import pandas as pd

pd.__version__
import sys

version = 1.0


def isolate_data(df):

    featureNames = df.columns.tolist()
    requiredNames = ["Time", "Source", "Destination", "Length", "Srcport"]
    requiredNames2 = ["Time", "Source", "Destination", "Length", "Src Port"]
    requiredNames3 = ["Time", "Source", "Destination", "Length", "Source Port"]
    requiredNames4 = ["time", "size", "direction"]
    if all(item in featureNames for item in requiredNames) or all(item in featureNames for item in requiredNames2) or all(item in featureNames for item in requiredNames3) or all(item in featureNames for item in requiredNames4):
        # print("Found all required features ...")
        # print("\nFound Featutures:", featureNames)
        if all(item in featureNames for item in requiredNames4):  ## for the QUESTSET dataset pre processing
            time = df["time"]
            pktLength = df["size"]
            directionSize = df["direction"]
            direction = []
            srcPort = []
            for n in range(len(directionSize)):
                srcPort.append(0)  ## irrelivant source port
                if directionSize[n] == "UL":
                    direction.append(0)
                else:
                    direction.append(1)

        else:  # for our dataset and the Pixels to Packets Dataset |

            time = df["Time"]  # |
            source = df["Source"]  # |
            destination = df["Destination"]  # |
            pktLength = df["Length"]  # |
            # |
            if "Srcport" in featureNames:  # |
                srcPort = df["Srcport"]  # |
            elif "Src Port" in featureNames:  # |
                srcPort = df["Src Port"]  # |
            elif "Source Port" in featureNames:  # |
                srcPort = df["Source Port"]  # |
                # |
                # |
            direction = []  # |
            # |
            src_ip = source[0]  # |
            dst_ip = destination[0]  # |
            # |
            for n in range(len(destination)):  # |
                if destination[n] == src_ip:  # |
                    direction.append(1)  # |
                else:  # |
                    direction.append(0)  # | /end

        interarrivalTime = []  # interarrival time general calculation for all datasets (once 'time' parameter is determined)

        for n in range(len(time)):
            if n == 0:
                interarrivalTime.append(0)
            else:
                interarrivalTime.append(time[n] - time[n - 1])
        isolatedData = pd.DataFrame({"Time": time, "Length": pktLength, "inter-arrival": interarrivalTime, "Dir": direction, "Source Port": srcPort})
        return isolatedData
    else:
        print("\nThe specified .csv is missing one of the following paramerters, be sure they are all spelt correctly (cap sensitive);")
        print("\nTime\nSource\nDestination\nLength\nSrcport or Src Port")
        sys.exit()
