#%%
from numpy import asarray
from numpy import save

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import sklearn
from scipy.stats import spearmanr
from scipy import spatial
from scipy import stats
# from scipy import mean
#from distfit import distfit
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import curve_fit
from sklearn.inspection import PartialDependenceDisplay

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
#from pmdarima.arima import auto_arima

from statsmodels.tsa.stattools import adfuller

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
#this should work: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

import time
#%%
filepath = "irish_dataset/"
filenames = []
for i in range(1,14):
    filename = f"{filepath}DD{i}.csv"
    filenames.append(filename)

filter_params = ['Speed','SNR','RSSI','DL_bitrate']
#TRACE -> LAG
TRACES = len(filenames)
TRACE_COUNT=1
for TRACE_COUNT in range(1,TRACES+1):
  #LOOP BEGINS FOR ONE TRACE-----------------------------------------
  filenames = filenames[1:]+filenames[:1]
  TRACE_LEN_LIST=[]
  dfDD5glist=[]
  for fname in filenames:
    df1 = pd.read_csv(fname)
    # df1 = df1[df1['nrStatus'] == 'CONNECTED']
    # df1 = df1[df1['nrStatus_array'] == '[\'CONNECTED\']']
    for cname in filter_params:
        df1 = df1[df1[cname] != '-']
        df1[cname] = df1[cname].astype('float')
    dfDD5glist.append(df1)
    TRACE_LEN_LIST.append(df1.shape[0])
  dfDD5gM = pd.concat(dfDD5glist)
  dfDD5gM = dfDD5gM.dropna().reset_index(drop=True)
  split_index = sum(TRACE_LEN_LIST[:-1])

  fv = ['Speed','SNR','DL_bitrate']
  measured_dataset = dfDD5gM[fv] #measured dataset normalized
  measured_dataset_heatmap = dfDD5gM[filter_params]
  MIN_M = min(measured_dataset['DL_bitrate'])
  MAX_M = max(measured_dataset['DL_bitrate'])
  for col in measured_dataset.columns:
    measured_dataset[col] = scaling(measured_dataset[col])
  filter_order=1
  for delay in [5]:
      for filter_size in [3]:
          #Filtering--------------------------------------------------------------------------------------------------
          filtered_dataset = pd.DataFrame()
          filtered_dataset = measured_dataset.copy()#deep=True)
          for param in fv:
                  MVA= moving_average(list(measured_dataset[param]), filter_size)
                  for i in range(len(MVA)):
                      filtered_dataset.at[i,param] = MVA[i]
          for i in range(filter_size):
              filtered_dataset.drop(filtered_dataset.shape[0]-1,inplace = True)
          filtered_dataset_Future = filtered_dataset[delay:].reset_index(drop = True).copy()#deep = True)
          filtered_dataset_Present=filtered_dataset[:-delay].copy()#deep = True)
          
          #Train test split---------------------------------------------------------------------------------------------
          MD = measured_dataset[:-delay].reset_index(drop = True).copy(deep = True)
          MDF = measured_dataset[delay:].reset_index(drop=True).copy(deep = True)
          FDP = filtered_dataset_Present.copy(deep=True)
          FDF = filtered_dataset_Future.copy(deep=True)
          #ARIMA model----------------------------------------------------------------------------------------------------
          X_measured_test=MD[split_index:].copy()#.reset_index(drop=True)
          Y_measured_test= MDF['DL_bitrate'][split_index:].copy() #measured test set
          X_measured_train=MD[:split_index].copy()#.reset_index(drop=True)
          Y_measured_train= MDF['DL_bitrate'][:split_index].copy()
          #True dataset
          X_true_test = FDP[split_index:].copy()#.reset_index(drop=True)
          Y_true_test = FDF['DL_bitrate'][split_index:].copy() #true test set
          X_true_train = FDP[:split_index].copy()#.reset_index(drop=True)
          Y_true_train = FDF['DL_bitrate'][:split_index].copy()  
          Y_model = list() #predicted test set
          model = SARIMAX(Y_true_train,exog= X_true_train, order=(0,0,1),enforce_invertibility=False, enforce_stationarity=False)
          model_fit = model.fit()
          Y_model = model_fit.forecast(steps=X_measured_test.shape[0], exog = X_measured_test)
          #Saving in a file====================================================================
          trace_len=800
          #initialize
          y_pred=Y_model[:trace_len]
          #round off
          y_pred = [round(max(0,x),8) for x in np.add(np.multiply(y_pred,(MAX_M-MIN_M)),MIN_M)]
          #slice
          y_pred = y_pred[0:200]+y_pred[400:]
          #convert to npy format
          data2 = asarray(y_pred)

          #NAMING FORMAT--------------------------------------
          #y_true_DATASET_T<TRACE>
          #y_pred_DATASET_T<TRACE>L<LAG>F<FILTER>_drive
          #---------------------------------------------------

          #network traces
          save('network_traces/y_pred_Irish_DD_ARIMA_T'+str(TRACE_COUNT)+'L'+str(delay)+'F'+str(filter_size)+'_drive.npy',data2)

          #cooked traces pred
          f = open('cooked_traces/pred/y_pred_Irish_DD_ARIMA_T'+str(TRACE_COUNT)+'L'+str(delay)+'F'+str(filter_size)+'_drive', "w")
          write_list=[]
          for i in range(len(data2)):
              write_list.append(str(i)+' '+str(data2[i])+'\n')
          f.writelines(write_list)
          f.close()

          #one line
          f = open('oneline/y_pred_Irish_DD_ARIMA_T'+str(TRACE_COUNT)+'L'+str(delay)+'F'+str(filter_size)+'_driveoneline', "w")
          write_list=[]
          for i in range(len(data2)):
              write_list.append(str(int(data2[i])/8)+',')
          f.writelines(write_list)
          f.close()
  #LOOP ENDS FOR ONE TRACE--------------------------------------------
# %%
