import subprocess
import signal
import os
import time
import joblib
import numpy as np
import pandas as pd
import os
import threading as td
import time

# x = next(os.walk("flow/"))[2]
# y = "flow/" + next(os.walk("flow/"))[2][0]
# print(sorted(next(os.walk("flow/"))[2]))


std_scaler, mm_scaler, le = joblib.load("exe/new_model/1.5_std_mm_le_all_feature.save")
model = joblib.load("exe/new_model/pkl/1.5_randomForest_all_feature_32.pkl")

while(True):
    try:
        data = pd.read_csv("NewDataSet_500k.csv")
    except pd.errors.EmptyDataError:
        print("meow")    
    else:
        print(data)
        data.columns = data.columns.str.strip()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        data_len = len(data)
        # data = data.drop(["src_ip" , "dst_ip", "src_port", "src_mac", "dst_mac", "protocol", "timestamp"], axis = 1)
        data = data.drop(["Label"], axis = 1)

        data = mm_scaler.transform(std_scaler.transform(data.values))
        f_pred = model.predict(data)
        print(f_pred)

        if f_pred.any() == 1:
            print("----------------------------------------------------------")
            print("     Alert : Anomaly traffic detected \t")
            print("----------------------------------------------------------")
            file = sorted(next(os.walk("flow/"))[2])[0]
            testdata = "flow/" + file
            destdata = "Abnormal/" + time.asctime(time.localtime(time.time())) + ".csv"
            os.replace(testdata, destdata)
            break