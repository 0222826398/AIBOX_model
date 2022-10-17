import subprocess
import signal
import os
import time
import joblib
import numpy as np
import pandas as pd
import os
import threading as td

std_scaler, mm_scaler, le = joblib.load("exe/new_model/1.5_std_mm_le_all_feature.save")
model = joblib.load("exe/new_model/pkl/1.5_randomForest_all_feature_32.pkl")
Flag = True

def cicflowmeter(start,C = 0):
    global pid

    if start:     
        p = subprocess.Popen(["cicflowmeter","-i",
                            "lo","-c",
                            'flow/flows' + str(C) + '.csv']) # Call subprocess
        pid = p.pid
    elif not start:   
        os.kill(pid, signal.SIGINT)

    return None


def data_preprocess(data):
    data.columns = data.columns.str.strip()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data_len = len(data)
    data = data.drop(["src_ip" , "dst_ip", "src_port", "src_mac", "dst_mac", "protocol", "timestamp"], axis = 1)

    data = mm_scaler.transform(std_scaler.transform(data.values))

    return data, data_len


def cic_main():
    global Flag
    C = 0
    while(True):
        cicflowmeter(True, C)
        Flag = True
        print("cicflowmeter started")
        time.sleep(10)
        print("cicflowmeter stopped")
        cicflowmeter(False)
        Flag = False
        C = C + 1

    return None


def model_main():
    global Flag

    cnt = 0
    while(True and cnt < 3):
        while(not Flag):
            print("////flow predict////")
            time.sleep(2)
            file = sorted(next(os.walk("flow/"))[2])[0]
            testdata = "flow/" + file
            df = pd.read_csv(testdata)

            
            '''this need a if loop if df is nan need break'''
            

            data_flow, f_len = data_preprocess(df)

            f_pred = model.predict(data_flow)

            print(f_pred)
            print("----------------------------------------------------------")
            print("     Time                    : \t", time.asctime(time.localtime(time.time())))
            print("     Amount of flows         : \t", f_len)
            print("     Percentage of Anomaly   : \t", (list(f_pred).count(1)/f_pred.shape[0])*100)
            print("     Percentage of Legit     : \t",(list(f_pred).count(0)/f_pred.shape[0])*100)
            print("----------------------------------------------------------")

            os.remove(testdata)
            cnt = cnt + 1
            print("cnt = ",cnt)
        
    return None

t_cic = td.Thread(target = cic_main)
t_model = td.Thread(target = model_main)

t_cic.start()
t_model.start()

t_cic.join()
t_cic.join()