
#import packages
import pandas as pd
import numpy as np
import glob,os
import matplotlib.dates as mdates

#to plot within notebook
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn

#to set seed
np.random.seed(0) # for reproducibility
from datetime import datetime
import yfinance as yf
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#importing required libraries for Forecasting
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# from pmdarima.arima import auto_arima
#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#for cointegration test
from statsmodels.tsa.stattools import coint

import scipy


#read the file
import pandas_datareader as pdr
from datetime import datetime
import statsmodels.regression.linear_model as rg
import arch.unitroot as at

#3.1 calling the individual Cryptocurrencies


#1.1.4 Normalizing and setting up cumulative returns in a  datafrme
def normalize_and_accumulate_series(data):
    #take tail to drop head NA
    # return data.pct_change(1).cumsum().dropna()   #-------------------------------------------
    return (1+data.pct_change(1)).cumprod().dropna()
def normalize_series(data):
    #take tail to drop head NA
    return data.pct_change(1).dropna()




BTC = yf.download('BTC-USD',start=datetime(2019, 1, 1), end=datetime(2019, 9, 1)) # start=datetime(2017, 11, 9), end=datetime(2018, 12, 31)
ETH = yf.download('ETH-USD',start=datetime(2019, 1, 1), end=datetime(2019, 9, 1))  #start=datetime(2018, 1, 1), end=datetime(2019, 9, 1)

data= pd.concat([BTC['Adj Close'],ETH['Adj Close']], ignore_index=True,axis=1)

dt=normalize_series(data)
dtc =normalize_and_accumulate_series(data)

dt.columns = ['BTC_RET', 'ETH_RET']
dtc.columns = ['BTC_C.RET', 'ETH_C.RET']

# In[1074]:



plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(dtc['BTC_C.RET'],label='BTC CUMMULATIVE RETURNS')
plt.plot(dtc['ETH_C.RET'],label='ETH CUMMULATIVE RETURNS')
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.grid(True)
plt.legend(loc='best')
plt.show()

BTC_ETH = dt['BTC_RET'] - dt['ETH_RET']
plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(BTC_ETH,color='green',label='BTC_ETH SPREAD')
plt.suptitle('BTC-ETH SPREAD')
ax.axhline(BTC_ETH.mean(), color='orange',label='BTC_ETH Mean')
ax.axhline(BTC_ETH.mean()+BTC_ETH.std(), color='red',label='BTC_ETH Upper Threshold')
ax.axhline(BTC_ETH.mean()-BTC_ETH.std(), color='blue',label='BTC_ETH Lower Threshold')
ax.grid(True)
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend(loc='best')
plt.show()





plt.figure(figsize=(16, 12))  # 设置整个图形的大小

# 第一个子图，BTC和ETH累积收益率
plt.subplot(211)  # 创建第一个子图，2行1列的布局中的第1个图
plt.xticks(rotation=45)
ax1 = plt.gca()
ax1.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(dtc['BTC_C.RET'], label='BTC Cumulative Returns')
plt.plot(dtc['ETH_C.RET'], label='ETH Cumulative Returns')
plt.xlabel("Date")
ax1.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.title('BTC-ETH Cumulative Returns')
plt.legend(loc='best')

# 第二个子图，BTC-ETH价差
plt.subplot(212)  # 创建第二个子图，2行1列的布局中的第2个图
plt.xticks(rotation=45)
ax2 = plt.gca()
ax2.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(BTC_ETH, color='green', label='BTC_ETH SPREAD(Daily Return)')
plt.axhline(BTC_ETH.mean(), color='black', label='BTC_ETH Mean')
plt.axhline(BTC_ETH.mean() + 1.5*BTC_ETH.std(), color='yellow', label='BTC_ETH Upper Threshold')
plt.axhline(BTC_ETH.mean() - 1.5*BTC_ETH.std(), color='blue', label='BTC_ETH Lower Threshold')
plt.grid(True)
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.title('BTC-ETH Spread with the estimated Threshold')
plt.legend(loc='best')

plt.suptitle('BTC-ETH Analysis')  # 设置总标题
plt.tight_layout()  # 调整子图之间的间距和边缘
plt.show()




BTC = yf.download('BTC-USD',start=datetime(2018, 10, 1), end=datetime(2019, 9, 1))
ETH = yf.download('ETH-USD',start=datetime(2018, 10, 1), end=datetime(2019, 9, 1))

pair= pd.concat([BTC['Adj Close'],ETH['Adj Close']], ignore_index=True,axis=1)
pair_ret=normalize_series(pair)
pair_ret=pair_ret.tail(len(pair_ret)-1)
pair_ret.columns = ['BTC_RET','ETH_RET']

btc_R_train =  pair_ret['BTC_RET'][:90]
btc_R_test =   pair_ret['BTC_RET'][90:]
eth_R_train = pair_ret['ETH_RET'][:90]
eth_R_test =  pair_ret['ETH_RET'][90:]
tests= pd.concat([btc_R_test ,eth_R_test], ignore_index=False,axis=1)

pair_spread= btc_R_test - rg.OLS(btc_R_train, eth_R_train).fit().params[0] * eth_R_test
beta= rg.OLS(btc_R_train, eth_R_train).fit().params[0]

window= 55
pair_train= btc_R_test - rg.OLS(btc_R_train, eth_R_train).fit().params[0] * eth_R_test

z_score = (pair_train - pair_train.rolling(window=window).mean()) / pair_train.rolling(window=window).std()

z_score.rolling(window=2).mean()
z_score.rolling(window=2).std()
up_th = (z_score.rolling(window=2).mean())+(z_score.rolling(window=2).std()*2) # upper threshold
lw_th = (z_score.rolling(window=2).mean())-(z_score.rolling(window=2).std()*2) # lower threshold


plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(z_score,color='blue',label='Z-score')
plt.plot(up_th,color='red',linestyle='--', label='Upper Threshold')
plt.plot(lw_th,color='brown',linestyle='--', label='Lower  Threshold')
plt.suptitle('Z-score')
ax.axhline(z_score.mean(), color='orange')
ax.grid(True)
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend(loc='best')
plt.show()

z_score=z_score.dropna()

plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(z_score,color='black',label='Z-score')
plt.suptitle('Z-score')
ax.axhline(z_score.mean(), color='orange', label='Mean')
plt.axhline(z_score.mean() + 1.5*z_score.std(),color='yellow',linestyle='--', label='Upper Threshold')
plt.axhline(z_score.mean() - 1.5*z_score.std(),color='blue',linestyle='--', label='Lower  Threshold')
ax.grid(True)
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend(loc='best')
plt.show()