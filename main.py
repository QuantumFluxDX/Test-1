#%%
import matplotlib.pyplot as plt

from functions import *
#%%

filepath = 'irish_dataset/'
filenames = []
for num in range(1,14):
    filename = f"{filepath}DD{num}.csv"
    filenames.append(filename)

filter_params = ['Speed','RSRQ','SNR','CQI','RSSI','DL_bitrate','UL_bitrate']

#%%

measurements = extract_measurements(filenames, filter_params)
scaled = scale(measurements, filter_params)
[future_5s, present_5s] = add_delay(scaled, 5)
[future_3s, present_3s] = add_delay(scaled, 3)
[future_1s, present_1s] = add_delay(scaled, 1)

heatmap_5s = create_heatmap(future_5s, present_5s, filter_params)
heatmap_3s = create_heatmap(future_3s, present_3s, filter_params)
heatmap_1s = create_heatmap(future_1s, present_1s, filter_params)

#%%

# plt.plot(measurements['DL_bitrate'])
# plt.title("Throughput")

#%%
#%%
plt.figure()
plt.imshow(heatmap_5s,cmap='jet')
plt.xticks(list(range(len(filter_params))),filter_params,rotation=45)
plt.yticks(list(range(len(filter_params))),filter_params)
plt.xlabel('Present')
plt.ylabel('Future')
plt.title('5s Delay')
plt.colorbar()
for i in range(len(filter_params)):
    for j in range(len(filter_params)):
        text = plt.text(j, i, f'{heatmap_5s[i][j]:.2f}', ha='center',
                         va='center', color='k')
plt.savefig("heatmaps/heatmap_irish_5s.png")

plt.figure()
plt.imshow(heatmap_3s,cmap='jet')
plt.xticks(list(range(len(filter_params))),filter_params,rotation=45)
plt.yticks(list(range(len(filter_params))),filter_params)
plt.xlabel('Present')
plt.ylabel('Future')
plt.title('3s Delay')
plt.colorbar()
for i in range(len(filter_params)):
    for j in range(len(filter_params)):
        text = plt.text(j, i, f'{heatmap_3s[i][j]:.2f}', ha='center',
                         va='center', color='k')
plt.savefig("heatmaps/heatmap_irish_3s.png")

plt.figure()
plt.imshow(heatmap_1s,cmap='jet')
plt.xticks(list(range(len(filter_params))),filter_params,rotation=45)
plt.yticks(list(range(len(filter_params))),filter_params)
plt.xlabel('Present')
plt.ylabel('Future')
plt.title('1s Delay')
plt.colorbar()
for i in range(len(filter_params)):
    for j in range(len(filter_params)):
        text = plt.text(j, i, f'{heatmap_1s[i][j]:.2f}', ha='center',
                         va='center', color='k')
plt.savefig("heatmaps/heatmap_irish_1s.png")
# %%
