#%%
from numpy import asarray
from numpy import save
from matplotlib import pyplot as plt
#import xgboost as xgb
import numpy as np
import pandas as pd

import math
import sklearn
from scipy.stats import spearmanr
from scipy import spatial
from scipy import stats
# from scipy import mean
import seaborn as sns
#from sklearn import svm
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import curve_fit
from sklearn.inspection import PartialDependenceDisplay

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def remove_items(test_list, item):
    # using list comprehension to perform the task
    res = [i for i in test_list if i != item]
    return res

def scaling(l):
    return (l - min(l))/(max(l) - min(l))
def scaling1(l,offset,sf):
    return  (l-offset)/sf
def Gauss(x,A,B):
    y = A*np.exp(-1*B*x**2)
    return y

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

from itertools import chain
delay_filtersize_list = [] #stores (delay,filter_size) tuples
plot_dictionary = {} #index = delay, values = list of tuples (filter size, r2 gain)

import time

#%%
filepath = 'irish_dataset/'
filenames = []
for num in range(1, 3):
    filename = f"{filepath}DD{num}.csv"
    filenames.append(filename)

filter_params = ['Speed','SNR','CQI','RSRP','RSRQ','RSSI','NRxRSRP',
                 'DL_bitrate']

columnnamesdrop = ['State',
 'PINGAVG',
 'PINGMIN',
 'PINGMAX',
 'PINGSTDEV',
 'PINGLOSS',
 'CELLHEX',
 'NODEHEX',
 'LACHEX',
 'RAWCELLID',
 'NRxRSRQ',
 'Operatorname',
 'CellID']

datatypechange = ['Speed','RSRP','RSRQ','SNR','CQI','RSSI','NRxRSRP','DL_bitrate','UL_bitrate']

# TRACE -> LAG
TRACES = len(filenames)
TRACE_COUNT = 1
for TRACE_COUNT in range(1, TRACES + 1):
    # LOOP BEGINS FOR ONE TRACE-----------------------------------------
    filenames = filenames[1:] + filenames[:1]
    TRACE_LEN_LIST = []
    dfDD5glist = []
    for fname in filenames:
        df1 = pd.read_csv(fname)
        df1 = df1.drop(columns = columnnamesdrop)
        # df1 = df1[df1['nrStatus'] == 'CONNECTED']
        # df1 = df1[df1['nrStatus_array'] == '[\'CONNECTED\']']
        for cname in filter_params:
            df1 = df1[df1[cname] != '-']
        for cname in filter_params:
            df1[cname] = df1[cname].astype('float')
        dfDD5glist.append(df1)
        TRACE_LEN_LIST.append(df1.shape[0])
    dfDD5gM = pd.concat(dfDD5glist)
    dfDD5gM = dfDD5gM.dropna().reset_index(drop=True)
    split_index = sum(TRACE_LEN_LIST[:-1])

    fv = ['Speed','SNR','CQI','RSRP','RSRQ','RSSI','NRxRSRP','DL_bitrate']
    measured_dataset = dfDD5gM[fv]  # measured dataset normalized
    measured_dataset_heatmap = dfDD5gM[filter_params]
    MIN_M = min(measured_dataset['DL_bitrate'])
    MAX_M = max(measured_dataset['DL_bitrate'])
    for col in measured_dataset.columns:
        measured_dataset[col] = scaling(measured_dataset[col])
    filter_order = 1
    for delay in [5]:
        for filter_size in [3]:
            # Filtering--------------------------------------------------------------------------------------------------
            filtered_dataset = pd.DataFrame()
            filtered_dataset = measured_dataset.copy()  # deep=True)
            for param in fv:
                MVA = moving_average(list(measured_dataset[param]), filter_size)
                for i in range(len(MVA)):
                    filtered_dataset.at[i, param] = MVA[i]
            for i in range(filter_size):
                filtered_dataset.drop(filtered_dataset.shape[0] - 1, inplace=True)

            filtered_dataset_Future = filtered_dataset[delay:].reset_index(drop=True).copy()
            filtered_dataset_Present = filtered_dataset[:-delay].copy()

            # Train test split---------------------------------------------------------------------------------------------
            MD_Train = measured_dataset[0:split_index].copy(deep=True)
            MD_Test = measured_dataset[split_index:-delay].reset_index(drop=True).copy(deep=True)

            MDF_Train = measured_dataset[delay:delay + split_index].reset_index(drop=True).copy(deep=True)
            MDF_Test = measured_dataset[delay + split_index:].reset_index(drop=True).copy(deep=True)

            # FD_Train = filtered_dataset[0:split_index].copy(deep = True)
            # FD_Test = filtered_dataset[split_index:].reset_index(drop = True).copy(deep = True)

            FDP_Train = filtered_dataset_Present[0:split_index].copy(deep=True)
            FDP_Test = filtered_dataset_Present[split_index:].reset_index(drop=True).copy(deep=True)

            FDF_Train = filtered_dataset_Future[0:split_index].copy(deep=True)
            FDF_Test = filtered_dataset_Future[split_index:].reset_index(drop=True).copy(deep=True)

            # LSTM model----------------------------------------------------------------------------------------------------
            np.random.seed(7)
            filter_order = 1
            X = []
            for i in range(FDP_Train.shape[0] - filter_order + 1):
                mat = []
                for param in fv[:-1]:
                    mat.append(FDP_Train[param][i + filter_order - 1])
                for param in fv[-1:]:
                    for k in range(filter_order):
                        mat.append(FDP_Train[param][i + k])
                X.append(mat[:])
            X = np.array(X)
            Y = np.array(FDF_Train['DL_bitrate'])
            Y = Y[filter_order - 1:]
            # reshape input to be [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            # create and fit the LSTM network
            reg = Sequential()
            reg.add(LSTM(4, input_shape=(1, len(fv))))
            reg.add(Dense(1))
            reg.compile(loss='mean_squared_error', optimizer='adam')
            reg.fit(X, Y, epochs=30, batch_size=1, verbose=2)

            X = []  # FDP
            for i in range(FDP_Test.shape[0] - filter_order + 1):
                mat = []
                for param in fv[:-1]:
                    mat.append(FDP_Test[param][i + filter_order - 1])
                for param in fv[-1:]:
                    for k in range(filter_order):
                        mat.append(FDP_Test[param][i + k])
                X.append(mat[:])
            X = np.array(X)

            # reshape input to be [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

            Y_true = np.array(FDF_Test['DL_bitrate'])
            Y_true = Y_true[filter_order - 1:]
            Y_true1 = np.array(FDP_Test['DL_bitrate'])
            Y_true1 = Y_true1[filter_order - 1:]
            # Y_model = []
            # for i in range(len(X)):
            #    Y_model.append(sum(np.multiply(mlr.coef_,X[i]),mlr.intercept_))
            Y_model = list(reg.predict(X))

            # Measured data set
            MD = []
            for i in range(MD_Test.shape[0] - filter_order + 1):
                mat = []
                for param in fv[:-1]:
                    mat.append(MD_Test[param][i + filter_order - 1])
                for param in fv[-1:]:
                    for k in range(filter_order):
                        mat.append(MD_Test[param][i + k])
                MD.append(mat[:])
            MD = np.array(MD)
            # reshape input to be [samples, time steps, features]
            MD = np.reshape(MD, (MD.shape[0], 1, MD.shape[1]))
            Y_measured_future = np.array(MDF_Test['DL_bitrate'])
            Y_measured_future = Y_measured_future[filter_order - 1:]
            Y_measured = np.array(MD_Test['DL_bitrate'])
            Y_mlr = reg.predict(MD[:])
            # Saving in a file====================================================================
            trace_len = 800
            # initialize
            y_pred = Y_mlr[:trace_len]
            # round off
            y_pred = [np.round(max(0, x), 8) for x in np.add(np.multiply(y_pred, (MAX_M - MIN_M)), MIN_M)]
            # slice
            y_pred = y_pred[0:200] + y_pred[400:]

            # convert to npy format
            data1 = asarray(Y_true)
            data2 = asarray(y_pred)

            # NAMING FORMAT--------------------------------------
            # y_true_DATASET_T<TRACE>
            # y_pred_DATASET_T<TRACE>L<LAG>F<FILTER>_drive
            # ---------------------------------------------------

            # network traces
            save('network_traces/y_pred_TNSA_LSTM_T'+str(TRACE_COUNT)+'L'+str(delay)+'F'+str(filter_size)+'_drive.npy',data2)

            # cooked traces pred
            # f = open('cooked_traces/pred/y_pred_TNSA_LSTM_T'+str(TRACE_COUNT)+'L'+str(delay)+'F'+str(filter_size)+'_drive', "w")
            # write_list=[]
            # for i in range(len(data2)):
            #     write_list.append(str(i)+' '+str(data2[i])+'\n')
            # f.writelines(write_list)
            # f.close()

            # one line
            # f = open('oneline/y_pred_TNSA_LSTM_T'+str(TRACE_COUNT)+'L'+str(delay)+'F'+str(filter_size)+'_driveoneline', "w")
            # write_list=[]
            # for i in range(len(data2)):
            #     write_list.append(str(int(data2[i])/8)+',')
            # f.writelines(write_list)
            # f.close()
    # LOOP ENDS FOR ONE TRACE--------------------------------------------


#%%
plt.plot(data2)
#%%
plt.plot(Y_true)