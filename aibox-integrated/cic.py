import subprocess
import signal
import os
import time

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


cicflowmeter(True)
print("cicflowmeter started")
time.sleep(30)
print("cicflowmeter stopped")
cicflowmeter(False)


print("model start!!!!!")
###### 資料讀取 實際測試時此區改為讀取攔截之封包資料
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
testdata = "flows.csv"
df = pd.read_csv(testdata)

df.columns = df.columns.str.strip()
print("original length of df:", len(df))
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print("after droping null values, the length of df:", len(df))

df = df.drop(["src_ip" , "dst_ip", "src_port", "src_mac", "dst_mac", "protocol", "timestamp"], axis = 1)

# data split and preprocess
from sklearn.model_selection import train_test_split
import joblib

std_scaler,mm_scaler,le = joblib.load("std_mm_le_new.save")

X = df

X = std_scaler.transform(X)

print("after StandardScaler")
print(X.shape)

X = mm_scaler.transform(X)

print("after MinMaxScaler")
print(X.shape)

#標籤編碼 可不跑
# y_train = le.transform(y_train)
# y_test = le.transform(y_test)


###### Random Forest
import joblib

rng = np.random.RandomState(42)

model = joblib.load("new_randomForest_32.pkl")
X_pred = model.predict(X)

print(X_pred)
print(X_pred.shape)

print("percentage of Anomaly:", (list(X_pred).count(1)/X_pred.shape[0])*100)
print("percentage of Legit:",(list(X_pred).count(0)/X_pred.shape[0])*100)

print("model end!!!!!")