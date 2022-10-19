import subprocess
import signal
import os
import time
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

std_scaler,mm_scaler,le = joblib.load("exe/new_model/1.5_std_mm_le_all_feature.save")
model = joblib.load("exe/new_model/pkl/1.5_randomForest_all_feature_32.pkl")

def cicflowmeter(start):
    global pid
    if start:     
        p = subprocess.Popen(["cicflowmeter","-i",
                            "lo","-c",
                            'flows.csv']) # Call subprocess
        pid = p.pid
    elif not start:   
        os.kill(pid, signal.SIGINT)
        # os.kill(pid, signal.SIGKILL)

    return None

def data_preprocess(data):
    data.columns = data.columns.str.strip()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data_len = len(data)
    data = data.drop(["src_ip" , "dst_ip", "src_port", "src_mac", "dst_mac", "protocol", "timestamp"], axis = 1)

    data = mm_scaler.transform(std_scaler.transform(data))

    return data


cicflowmeter(True)
print("cicflowmeter started")
time.sleep(30)
print("cicflowmeter stopped")
cicflowmeter(False)


print("model start!!!!!")
time.sleep(5)
testdata = "flows.csv"
df = pd.read_csv(testdata)

data_flow = data_preprocess(df)

X_pred = model.predict(data_flow)


print(X_pred)
print(X_pred.shape)

print("percentage of Anomaly:", (list(X_pred).count(1)/X_pred.shape[0])*100)
print("percentage of Legit:",(list(X_pred).count(0)/X_pred.shape[0])*100)

print("model end!!!!!")