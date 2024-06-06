import pandas as pd
import numpy as np

def extract_measurements(filenames, filter_params):
    """Creates a single dataframe from a list of csv files"""
    
    # Initialise the dataframe
    dataframe = []
    for file in filenames:
        # Read data from a csv file in filenames
        data = pd.read_csv(file) 

        # Drop the rows parameter values = '-'
        for param in filter_params:
            data = data[data[param] != '-']

        # Convert parameter values to float
        for param in filter_params:
            data[param] = data[param].astype('float')

        # Only keep rows with 5G Network Mode
        # data = data[data['NetworkMode'] == '5G']

        # Append data to the dataframe
        dataframe.append(data)

    # Concatanate the datas in dataframe
    dataframe = pd.concat(dataframe).reset_index(drop=True)
    return dataframe[filter_params]

def scale(measurements, filter_params):
    scaled = measurements.copy()
    for param in filter_params:
        L = _scaling(measurements[param])
        scaled[param] = L
    return scaled

def _scaling(list):
    min_list = min(list)
    max_list = max(list)
    
    scaled_list = []
    for element in list:
        scaled_element = (element - min_list)/(max_list - min_list)
        scaled_list.append(scaled_element)
    return scaled_list

def add_delay(measurements, delay):
    if delay != 0:
        scaled_present = measurements[:-delay]
        scaled_future = measurements[delay:]
    else:
        scaled_future = measurements
        scaled_present = measurements

    return [scaled_future, scaled_present]

def create_heatmap(future_values, present_values, filter_params):
    heatmap = []
    for i in range(len(filter_params)):
        heatmap.append([])
        for j in range(len(filter_params)):
            param_1 = future_values[filter_params[i]]
            param_2 = present_values[filter_params[j]]
            corr_coeff = np.corrcoef(param_1, param_2)
            heatmap[i].append(corr_coeff[0][1])

    return heatmap