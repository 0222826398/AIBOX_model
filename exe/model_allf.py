###### 資料讀取 實際測試時此區改為讀取攔截之封包資料
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
import joblib

std_scaler,mm_scaler,le = joblib.load("exe/new_model/1.5_std_mm_le_all_feature.save")
model = joblib.load("exe/new_model/pkl/1.5_randomForest_all_feature_32.pkl")

def data_preprocess(data):
    data.columns = data.columns.str.strip()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data_len = len(data)
    data = data.drop(["src_ip" , "dst_ip", "src_port", "src_mac", "dst_mac", "protocol", "timestamp"], axis = 1)

    data = mm_scaler.transform(std_scaler.transform(data))

    return data


df = pd.read_csv("exe/testflow.csv")

data_flow = data_preprocess(df)

X_pred = model.predict(data_flow)

print(X_pred)
print(X_pred.shape)
print("percentage of Anomaly:", (list(X_pred).count(1)/X_pred.shape[0])*100)
print("percentage of Legit:",(list(X_pred).count(0)/X_pred.shape[0])*100)