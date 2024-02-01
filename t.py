import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

def scaling(list):
    minval = min(list)
    maxval = max(list)

    scaled = []
    for l  in list:
        s = (l - minval)/(maxval - minval)
        scaled.append(s)

    return scaled


filepath = 'irish_dataset/'
filenames = []
for num in range(1,15):
    filename = f"{filepath}DD{num}.csv"
    filenames.append(filename)

filter_params = ['Speed', 'RSRP', 'RSRQ', 'SNR', 'CQI', 'RSSI', 'DL_bitrate'
                 ,'NRxRSRP']

dataset = []
for file in filenames:
    data = pd.read_csv(file)
    for param in filter_params:
        data = data[data[param] != '-']
    for param in filter_params:
        data[param] = data[param].astype('float')
    data = data[data['NetworkMode'] == '5G']
    dataset.append(data)

dataset = pd.concat(dataset).reset_index(drop=True)

measurements = dataset[filter_params]
scaled = measurements.copy()

for param in filter_params:
    L = scaling(measurements[param])
    scaled[param] = L

plt.plot(scaled['DL_bitrate'])


delay = 5

if delay != 0:
    scaled_present = scaled[:-delay]
    scaled_future = scaled[delay:]
else:
    scaled_future = scaled
    scaled_present = scaled

heatmap = []
for i in range(len(filter_params)):
    heatmap.append([])
    for j in range(len(filter_params)):
        param_1 = scaled_future[filter_params[i]]
        param_2 = scaled_present[filter_params[j]]
        corr_coeff = np.corrcoef(param_1, param_2)
        heatmap[i].append(corr_coeff[0][1])

plt.imshow(heatmap,cmap='jet')
plt.xticks(list(range(len(filter_params))),filter_params,rotation=45)
plt.yticks(list(range(len(filter_params))),filter_params)
plt.xlabel('Present')
plt.ylabel('Future')
plt.colorbar()

for i in range(len(filter_params)):
    for j in range(len(filter_params)):
        text = plt.text(j, i, f'{heatmap[i][j]:.2f}', ha='center',
                         va='center', color='k')