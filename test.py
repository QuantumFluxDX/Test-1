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

y = [array([6200.5303], dtype=float32), array([6200.5303], dtype=float32),]
x = asarray(y)