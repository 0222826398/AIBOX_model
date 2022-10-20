###### 資料讀取 實際測試時此區改為讀取攔截之封包資料
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("exe/bruteflows.csv")

df.columns = df.columns.str.strip()
print("original length of df:", len(df))
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print("after droping null values, the length of df:", len(df))




df = df.drop(["src_ip" , "dst_ip", "src_port", "src_mac", "dst_mac", "protocol", "timestamp"], axis = 1)
# df = df.drop(["Label"], axis = 1)


std_scaler,mm_scaler,le = joblib.load("exe/new_model/25679_std_mm_le_20_feature.save")

X = df
X = std_scaler.transform(X)
print("after StandardScaler")
print(X.shape)

X = mm_scaler.transform(X)
print("after MinMaxScaler")
print(X.shape)


model = joblib.load("exe/new_model/pkl/25679_randomForest_20_feature_32.pkl")
X_pred = model.predict(X)

print(X_pred)
print(X_pred.shape)
print("percentage of Anomaly:", (list(X_pred).count(1)/X_pred.shape[0])*100)
print("percentage of Legit:",(list(X_pred).count(0)/X_pred.shape[0])*100)