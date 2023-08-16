# import packages
import pandas as pd
import numpy as np
import glob, os
import matplotlib.dates as mdates
from tqdm import tnrange, notebook
# to plot within notebook
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from datetime import timedelta, datetime

# from tensorboard import notebook

# to set seed
np.random.seed(0)  # for reproducibility
from datetime import datetime
import yfinance as yf
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
# setting figure size
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

# importing required libraries for Forecasting
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# from pmdarima.arima import auto_arima
# for normalizing data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# for cointegration test
from statsmodels.tsa.stattools import coint

import scipy

# read the file
import pandas_datareader as pdr
from datetime import datetime
import statsmodels.regression.linear_model as rg
import arch.unitroot as at

BTC = yf.download('BTC-USD', start=datetime(2018, 12, 31),
                  end=datetime(2019, 12, 31))  # start=datetime(2017, 11, 9), end=datetime(2018, 12, 31)
ETH = yf.download('ETH-USD', start=datetime(2018, 12, 31),
                  end=datetime(2019, 12, 31))  # start=datetime(2018, 1, 1), end=datetime(2019, 9, 1)

S1 = BTC['Adj Close']
S2 = ETH['Adj Close']

# startdate = datetime(2018,12,31)
# enddate = datetime(2019,12,31)

# train_prices = prices['2018-12-31':'2019-09-14']
# test_prices = prices['2019-09-14':'2019-12-31']

P1 = BTC['Adj Close']
P2 = ETH['Adj Close']

train_prices_btc = P1['2018-12-31':'2019-09-14']
train_prices_eth = P2['2018-12-31':'2019-09-14']
test_prices_btc = P1['2019-09-14':'2019-12-31']
test_prices_eth = P2['2019-09-14':'2019-12-31']

# price1 = train_prices[P1]
# price2 = train_prices[P2]

# In[24]:


# In[25]:


score, pvalue, _ = coint(S1, S2)
ratios = S1 / S2

plt.figure(figsize=(15, 7))

plt.plot(ratios.index.values, ratios.values, color='blue')
plt.axhline(ratios.mean())
plt.legend([' Ratio'])
plt.show()

print('Mean ratio: {:.2f}'.format(ratios.mean()))


# We should expect the ratio to move around a stable mean.  Normalising the ratio below.

# In[26]:


def zscore(series):
    return (series - series.mean()) / np.std(series)


# In[27]:


plt.figure(figsize=(15, 7))

plt.plot(zscore(ratios).index.values, zscore(ratios).values, color='blue')
plt.axhline(zscore(ratios).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
plt.show()

ratios_mavg5 = ratios.rolling(window=5, center=False).mean()
ratios_mavg60 = ratios.rolling(window=60, center=False).mean()
std_60 = ratios.rolling(window=60, center=False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60

plt.figure(figsize=(15, 7))
plt.plot(ratios.index.values, ratios.values)
plt.plot(ratios_mavg5.index.values, ratios_mavg5.values)
plt.plot(ratios_mavg60.index.values, ratios_mavg60.values)
plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])
plt.ylabel('Ratio')
plt.show()

# Visually, we are expecting the orange line to close up (mean-revert) to the green line in the shortest possible time.  While divergence generates trading opportunities, prolonged divergence is not good for the strategy.  Prolonged divergence will lead to:
# 1. Closing of position as extreme deviated prices leading to potential losses
# 2. Long holding periods

# In[29]:


plt.figure(figsize=(15, 7))

plt.plot(zscore_60_5.index.values, zscore_60_5.values, color='blue')

plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()

# The Z-Score should mean revert to 0.  Possible trading signal when there is extreme deviation from mean 0.
# - When Z-Score is above the red line, Short: Short Coin 1 and Long Coin 2
# - When Z-Score is below the green line, Short: Long Coin 1 and Long Coin 2

# In[30]:


# Plot the ratios and buy and sell signals from z score
plt.figure(figsize=(15, 7))

plt.plot(ratios[60:].index.values, ratios[60:].values)

buy = ratios.copy()
sell = ratios.copy()
buy[zscore_60_5 > -1] = 0
sell[zscore_60_5 < 1] = 0

plt.plot(buy[60:].index.values, buy[60:].values, color='g', linestyle='None', marker='^')
plt.plot(sell[60:].index.values, sell[60:].values, color='r', linestyle='None', marker='^')

x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratios[60:].min() * 0.99, ratios[60:].max() * 1.01))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

# The trading signals to buy and sell.  Green dots are buying signals.  Red dots are selling signals.

# In[31]:


# Plot the prices and buy and sell signals from z score

buyS1 = 0 * S1.copy()
sellS1 = 0 * S1.copy()
buyS2 = 0 * S2.copy()
sellS2 = 0 * S2.copy()

# # When buying the ratio, buy S1 and sell S2
buyS1[buy != 0] = S1[buy != 0]
sellS2[buy != 0] = S2[buy != 0]

# # When selling the ratio, sell S1 and buy S2
buyS2[sell != 0] = S2[sell != 0]
sellS1[sell != 0] = S1[sell != 0]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.set_ylabel(P1.index.values)
ax1.plot(S1[60:].index.values, S1[60:].values, color='b')
ax1.tick_params(axis='y')
ax1.plot(S1[60:].index.values, buyS1[60:].values, color='g', linestyle='None', marker='^', markersize=2)
ax1.plot(S1[60:].index.values, sellS1[60:].values, color='r', linestyle='None', marker='^', markersize=2)
ax1.set_ylim([min(S1[60:]) * 0.95, max(S1[60:]) * 1.05])

ax2.set_ylabel(P2.index.values)  # we already handled the x-label with ax1
ax2.plot(S2[60:].index.values, S2[60:].values, color='b')
ax2.tick_params(axis='y')
ax2.plot(S2[60:].index.values, buyS2[60:].values, color='g', linestyle='None', marker='^', markersize=2)
ax2.plot(S2[60:].index.values, sellS2[60:].values, color='r', linestyle='None', marker='^', markersize=2)
ax2.set_ylim([min(S2[60:]) * 0.95, max(S2[60:]) * 1.05])

ax2.set_xlabel('timestamp')

# ax2.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# Similarly, the above plots show that buying and selling signals of each pairs.  It is best to observe both prices move in tandem with minimal deviation.

# ---
# ## Trading Strategy
#
# Execute trades when the signal meets the following triggering criterias at any particular time
#
# ##### Signal
# - **Z Score**: (Near_MA - Far_MA) / Std(Far_MA)
#
# ##### Trigger
# - **Buy**: When there is no existing position and Z Score < -1
# - **Sell**: When there is no existing position and Z Score > 1
# - **Close all**: When there is existing position and Z Score is between -1 and 1 inclusive
#
# ##### Trade
# *Where n = Price of Coin 1 / Price of Coin 2*
# - **Buy Signal**: Buy 1 x Coin 1 and Sell n x Coin 2
# - **Sell Signal**: Sell 1 x Coin 1 and Buy n x Coin 2
#

# The trade function executes the strategy by taking in the following parameters:
# 1. **P1** - Coin 1 symbol
# 2. **P2** - Coin 2 symbol
# 3. **near** - short term moving average period (used as an estimate of current price)
# 4. **far** - long term moving average period (used as an estimate of mean price in which current price will revert to)
# 5. **test** - Default is False.  Set to True if you want to use the test dataset.
# 6. **verbose** - Default is False.  Set to True for debugging.

# In[32]:


# Trade using a simple strategy
def trade(P1, P2, near, far, test=False, verbose=False):
    # If window length is 0, algorithm doesn't make sense, so exit
    if (near == 0) or (far == 0) or (near >= far):
        return 0

    # Select prices to use: train or test data
    if test:
        if verbose:
            print('Trade on Testing Data')
            print('\n')
        price1 = test_prices_btc
        price2 = test_prices_eth
    else:
        if verbose:
            print('Trade on Training Data')
            print('\n')
        price1 = train_prices_btc
        price2 = train_prices_eth

    # Compute rolling mean and rolling standard deviation
    # Trading signals for execution
    ratios = price1 / price2
    ma1 = ratios.rolling(window=near, center=False).mean()
    ma2 = ratios.rolling(window=far, center=False).mean()
    std = ratios.rolling(window=far, center=False).std()
    zscore = (ma1 - ma2) / std

    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    drawdown = 0

    # Logging all transactions for validation
    transactions = []

    def logTxn(datetime, action1, coin1, price1, qty1,
               action2, coin2, price2, qty2, zscore, hedgeratio, pnl, drawdown=0, holdingperiod=timedelta(0)):

        txn = {}
        txn['datetime'] = datetime
        txn['action1'] = action1
        txn['coin1'] = coin1
        txn['price1'] = price1
        txn['qty1'] = qty1
        txn['action2'] = action2
        txn['coin2'] = coin2
        txn['price2'] = price2
        txn['qty2'] = qty2
        txn['zscore'] = zscore
        txn['hedgeratio'] = hedgeratio
        txn['pnl'] = pnl
        txn['drawdown'] = drawdown
        txn['holdingperiod'] = holdingperiod

        return txn

    # Simulate trading
    # Period by period
    # Trade on trading signals derived based on information only available at the point in time
    for i in range(len(ratios)):

        # Sell short if the z-score is > 1 and when there is no existing position
        if (zscore[i] > 1) & (countS1 == 0) & (countS2 == 0):
            money += price1[i] - price2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]

            opendatetime = ratios.index[i]

            transactions.append(
                logTxn(opendatetime,
                       'Sell', P1, price1[i], countS1,
                       'Buy', P2, price2[i], countS2,
                       zscore[i], ratios[i], money)
            )

        # Buy long if the z-score is < 1 and when there is no existing position
        elif (zscore[i] < -1) & (countS1 == 0) & (countS2 == 0):
            money -= price1[i] - price2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]

            opendatetime = ratios.index[i]

            transactions.append(
                logTxn(opendatetime,
                       'Buy', P1, price1[i], countS1,
                       'Sell', P2, price2[i], countS2,
                       zscore[i], ratios[i], money)
            )

        # Close all positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75 and ((countS1 != 0) or (countS2 != 0)):
            money += price1[i] * countS1 + price2[i] * countS2

            closedatetime = ratios.index[i]
            holdingperiod = closedatetime - opendatetime

            transactions.append(
                logTxn(closedatetime,
                       'Close', P1, price1[i], countS1,
                       'Close', P2, price2[i], countS2,
                       zscore[i], ratios[i], money, min(money, drawdown), holdingperiod)
            )

            countS1 = 0
            countS2 = 0
            drawdown = 0

        # Keeps track of maximum drawdown when there are existing positions
        else:
            drawdown = min(price1[i] * countS1 + price2[i] * countS2, drawdown)

    return transactions


# In[33]:


def showResults(txn):
    txn = pd.DataFrame.from_dict(txn)
    txn = txn[['datetime', 'action1', 'coin1', 'price1', 'qty1',
               'action2', 'coin2', 'price2', 'qty2',
               'zscore', 'hedgeratio', 'pnl', 'drawdown', 'holdingperiod']]
    txn = txn.set_index('datetime')

    plt.figure(figsize=(15, 7))
    plt.plot(txn.pnl.index.values, txn.pnl.values)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.show()

    print('Profit/Loss: {:.2f}'.format(txn.iloc[-1].pnl))
    print('Maximum Drawdown: {:.2f}'.format(min(txn.drawdown)))
    print('Maximum Holding Period: {}'.format(max(txn.holdingperiod)))

    return txn


# ---
# ## Parameter Search
#
# Find the optimal near (Near_NA) and far (Far_MA) moving averages that give the best performance and risk profile on training data.

# In[34]:

#
# import itertools
#
# near = list(range(5,15,1))
# far = list(range(50,90,1))
#
# combine = [near,far]
# params = list(itertools.product(*combine))
#
#
# # In[35]:
#
#
# search_results = []
#
# for param in notebook.tqdm(params):
#     result = {}
#     txn = trade(P1, P2, param[0], param[1])
#
#     txn = pd.DataFrame.from_dict(txn)
#     txn = txn[['datetime', 'action1', 'coin1', 'price1', 'qty1',
#                'action2', 'coin2', 'price2', 'qty2',
#                'zscore', 'hedgeratio', 'pnl', 'drawdown', 'holdingperiod']]
#     txn = txn.set_index('datetime')
#
#     result['parameter'] = param
#     result['PnL'] = txn.iloc[-1].pnl
#     result['max_drawdown'] = min(txn.drawdown)
#     result['score'] = result['PnL']/abs(result['max_drawdown'])
#
#     search_results.append(result)
#
#
# # In[36]:
#
#
# best = pd.DataFrame(search_results).sort_values('score', ascending=False).reset_index().parameter[0]
#
#
# # In[37]:
#
#
# print('Best Parameters: Near: {}, Far: {}'.format(best[0], best[1]))
#

# ---
# ## Trading on Training Data
#
# Trade on training data using the best parameters and show the results

# In[38]:


# txn = trade(P1, P2, best[0], best[1])

txn = trade(P1, P2, 1, 5)

# In[39]:


train_results = showResults(txn)
#

# ---
# ## Out-Of-Sample Result
#
# Trade on testing data using the best parameters and show the results

# In[40]:


txn = trade(P1, P2, best[0], best[1], test=True)
#
#
# # In[41]:
#
#
test_results = showResults(txn)

# ---
# ## Transaction logs
#
# Showing the transaction logs for validation purpose
#
# ### Transaction log for in-sample (train) dataset

# In[42]:


train_results.head()

# ### Transaction log for out-of-sample (test) dataset

# In[43]:


test_results.head()

# ---
# ## Observations and Remarks
#
# Although the strategy may work for pairs that are deemed statistically cointegrated, one will notice that it may not work all the time.  The strategy is too simplistic or robust enough to be a consistent profit generator.
#
# There are many parameters or factors that can be considered and adjusted and like all automated trading strategies, these settings will need to be dynamically changed over time to align with the latest market conditions.
#
# ### Searching for cointegrated pairs using training data
# Period and duration of training data used to determined the cointegrated pairs.  Longer historical time period could be used on the assumption that pairs that are deemed cointegrated will have a higher tendency of being cointegrated at least in the short-term future.
#
# ### Resampling of time series impact to strategy
# Resampling of the time series to determined the cointegrated pairs is also important.  Pairs that are cointegrated pairs on 1 min time intervals may not be cointegrated on daily intervals.  Short time intervals translate to more trading opportunities and more data to determine which pairs are cointegrated at these time intervals.  However, this does not mean better profits.
#
# ### Back-testing of trading strategy
# More backtestings and analysis are required to determine this strategy works with consistently.  This analysis is only done based on a single training and testing period.  Instead one can repeat this analysis on a rolling basis over a longer period to see for themselves whether the strategy really works in the long term.
#
# ### Maximum drawdown and holding period
# In pair trading minimises risk by hedging one asset with another.  However, there is still risk involved if the deviation of the pair continues and does not converge fast enough.  This happens when the cointegration assumption is no longer valid, possibly due to specific events in any of the assets.  Maximum drawdown calculates the maximum loss one is exposed to at any one particular trade.  Note also that the longer it takes for the pair to converge, the longer the holding period, higher event risk the trade is exposed to.
#
# ### Cointegrated pairs changes over time
# The advantage of this strategy is that it does not limit the assets that one use for trading.  Assets can be traded as long as they are deemed cointegrated and dropped the moment they are no longer cointegrated.
#
# ### Portfolio size, position sizing and quantity
# The strategy currently, in the long trade, only buy 1 quantity of the 1st asset and hedge by selling with a fraction quantity of the 2nd asset.  Similarly, in the short trade, only sell 1 quantity of the 1st asset and hedge by buying a fraction quantity of the 2nd asset.  The fraction of quantity to buy or sell in the 2nd asset is determined by a hedge ratio (price of asset 1 / price of asset 2).  The hedge ratio calculation may not be optimal here and also the trade execution could be an issue if the quantity to trade is not a round number.  Portfolio size will be have introduced in order for this strategy to be more realistic.
#
# ### Trading multiple pair simultaneously
# While the strategy is to trade only one pair here, this can be easily extended to multiple cointegrated pairs at the same time.  Different pairs can be tuned to respective optimal parameters.  All within a portfolio with nett exposures constrained by portfolio size.
#

# ---
# ## Future Work
#
# 1. Correlation analysis of cryptocurrencies pairs
# 2. Multiple cointegrated pairs trading
# 3. Portfolio size adjustments
# 4. Optimal time resampling selection
# 5. Optimal hedge ratio and realistic sizing of positions

# ## Reference

# 1. https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a

# In[ ]:




