import subprocess
import signal
import os
import time
import joblib
import numpy as np
import pandas as pd
import os
import threading as td

std_scaler, mm_scaler, le = joblib.load("model/std_mm_le_76_feature.save")
model = joblib.load("model/pkl/randomForest_76_feature_32.pkl")
Flag = True
C = 0

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
    data = data.drop(["src_ip" , "dst_ip", "src_port", "dst_port", "src_mac", "dst_mac", "protocol", "timestamp"], axis = 1)
    #      data = data[["flow_duration" , "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std", "flow_iat_std", "flow_iat_max", "fwd_iat_tot", "fwd_iat_std", "fwd_iat_max", 
    #      "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var", "pkt_size_avg", "bwd_seg_size_avg", "idle_mean", "idle_max", "idle_min"]]

    data = mm_scaler.transform(std_scaler.transform(data.values))

    return data, data_len


def cic_main():
    global Flag
    global C

    while(True):
        cicflowmeter(True, C)
        print("cicflowmeter started")
        time.sleep(10)
        Flag = False
        print("cicflowmeter stopped")
        cicflowmeter(False)
        C = C + 1

    return None


def model_main():
    global Flag
    global C
    
    while(True):
        while(not Flag):
            print("////flow predict////")
            Flag = True
            time.sleep(1) # wait for cicflowmeter save file
            testdata = 'flow/flows' + str(C-1) + '.csv'
            
            try:
                df = pd.read_csv(testdata)

            except pd.errors.EmptyDataError: # or BaseException
                f_len, P_A, P_L, f_pred = 0, None, None, None
            
            else: #predict network flows
                data_flow, f_len = data_preprocess(df)

                f_pred = model.predict(data_flow)
                P_A = (list(f_pred).count(1)/f_pred.shape[0])*100
                P_L = (list(f_pred).count(0)/f_pred.shape[0])*100

            finally:
                if (f_len and f_pred.any()) == 1: # Exist Anomaly network flows
                   print("----------------------------------------------------------")
                   print("     Time                    : \t", time.asctime(time.localtime(time.time())))
                   print("     /////////Alert///////// : Anomaly traffic detected \t")
                   print("     flow file               : \t", testdata)
                   print("----------------------------------------------------------")
                   '''move error data to Abnormal file'''
                   desdata = "Abnormal/" + (time.asctime(time.localtime(time.time())).replace(":", "_")) + ".csv"
                   os.replace(testdata, desdata)
                   break

                #print(f_pred)
                # Normal network flows
                print("----------------------------------------------------------")
                print("     Time                    : \t", time.asctime(time.localtime(time.time())))
                print("     Amount of flows         : \t", f_len)
                print("     Percentage of Anomaly   : \t", P_A)
                print("     Percentage of Legit     : \t", P_L)
                print("----------------------------------------------------------")

                os.remove(testdata)
    return None

t_cic = td.Thread(target = cic_main)
t_model = td.Thread(target = model_main)

t_cic.start()
t_model.start()

t_cic.join()
t_model.join()
