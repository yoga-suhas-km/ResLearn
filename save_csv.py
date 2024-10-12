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
import os
import pandas as pd

pd.__version__

# save_csv.py was a script created to work with ExtractRawFeatures.py, now has two generic functions.
# first, save_csv which saves a dataframe as a CSV to a specefied location
# second, read_csv which reads and extract one CSV from from a specified location


# Function that saves a dataframe as a csv file based on three parameters
# 'saveLocation' string with the location where the file is to be saved
# 'saveFileName' name of the final that will be saved (can end with or without .csv)
# 'dfData' the dataframe to be converted
def save_csv(saveLocation, saveFileName, dfData):
    if saveFileName == "":
        saveFileName = "UnanmedData"
    if saveLocation == "":
        saveLocation = os.path.dirname(__file__)
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    if saveFileName.endswith(".csv"):
        dfData.to_csv(saveLocation + "\\" + saveFileName, index=False)
    else:
        dfData.to_csv(saveLocation + "\\" + saveFileName + ".csv", index=False)



# Function that reads a CSV to a dataframe based on a specefied location
# 'location' is the path directory to the folder that you wish to read from
# the function will then list all .csv files found in the specefied folder and then will ask the user to select one
# if you suspect multiple files in the same folder, you can specify specific files to save
def read_csv(location, specifiedName="NoSpecifiedFile"):
    append_data = []
    file_name = []
    cnt = 0
    path = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists("./" + location + "/"):
        for file in os.listdir("./" + location + "/"):
            if file.endswith(".csv"):
                cnt += 1
                file_path = os.path.join("./" + location + "/", file)
                #print("File Num.:", cnt, "|| Found:", file)
                df = pd.read_csv(file_path)
                append_data.append(df)
                file_name.append(file)
        if cnt == 0:
            return None
        if cnt == 1:
            return df
        else:
            fileNum = 0
            if specifiedName == "NoSpecifiedFile":
                while 1 > int(fileNum) or int(fileNum) > cnt:
                    fileNum = input("More than one file found, which would you like to work with (File 1 ... 2... 3... n; see above):")
                    if not str(fileNum).isdigit():
                        fileNum = -1
                    return append_data[int(fileNum) - 1]
            else:
                index = 0
                specLoc = 0
                for all in file_name:
                    if specifiedName in file_name[index]:
                        specLoc = index
                    index = index + 1
                return append_data[specLoc]

    else:
        return None
