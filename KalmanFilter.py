#Pair Trading using Kalman Filter Framwork
__author__ = 'Panjun'
#-*- coding: utf - 8 -*-
# further explore pair trading strategy, consider using Kalman Filter to construct hedging strategy
# first changing cointegration test from stock price to stock return
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sts
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter
import seaborn as sns
from operator import itemgetter

stock_prices = pd.read_csv('/Users/panjunwang/PycharmProjects/PythonWork/PairTradingFrameWork/stockprices.csv')
date_range = stock_prices['date'].values
stock_prices.set_index(date_range,inplace = True) #reset index as date_range, remember to add inplace
stock_prices.drop('date', axis = 1, inplace = True)

stock_prices = stock_prices.iloc[::-1]
stock_returns = stock_prices.pct_change()
stock_returns.drop('2013-07-31',axis=0,inplace = True) #calculate stocks' daily returns

#stock_returns = stock_returns.interpolate().dropna(axis=1)
stock_prices = stock_prices.interpolate().dropna(axis=1)

stock_list = list(stock_prices.columns.values)
pairs = [] #define pairs as list

#test cointegration of the stocks
for i in range(0,len(stock_list),1):
    for j in range(i+1, len(stock_list),1):
        results = coint(stock_prices.iloc[:,i],stock_prices.iloc[:,j])
        if results[1]< 0.01: #get the pvalues
            pairs.append([stock_list[i],stock_list[j],results[1]])

pairs_selected = []

#test difference of stock prices using ADF test
for i in range(0,len(pairs),1):
    results = sts.adfuller(stock_prices[pairs[i][0]]-stock_prices[pairs[i][1]],1)
    if results[1] < 0.05:
        pairs_selected.append(pairs[i])

#get the smallest p-value pair in 'pairs_selected'
pairs_selected.sort(key=itemgetter(2)) #ascending order thus the first two stocks are chosen

stock_1 = pairs_selected[0][0]  #China State Ship Building
stock_2 = pairs_selected[0][1]  #China Railway Group

print stock_1
print stock_2

# plot the price chart of stock_1 and stock_2
stock_prices[stock_1].plot(label=stock_1,color='r')
stock_prices[stock_2].plot(label=stock_2,color='b')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.show()

# calculate zscores
stable_series = stock_prices[stock_1] - stock_prices[stock_2]

z_scores = (stable_series - stable_series.mean())/stable_series.std()

plt.plot(z_scores,label='diff')
#z_scores.plot(date_range,label ='diff')
plt.axhline(z_scores.mean(), color='k')
plt.axhline(1, color='r', linestyle='--')
plt.axhline(-1, color='r', linestyle='--')
plt.axhline(1.5, color='b', linestyle='--')
plt.axhline(-1.5, color='b', linestyle='--')
plt.axhline(2, color='g', linestyle='--')
plt.axhline(-2, color='g', linestyle='--')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('diff')
plt.show()

#static hedging, considering
x = stock_prices[stock_1]
y = stock_prices[stock_2]

X = sts.add_constant(x)
result = (sts.OLS(y,X)).fit()
print (result.summary())



##stock price stationarity ADF test##

#we have got the stock list with stop trading for no more than 5 days
#test the stationarity of the stocks in the slected stocks by ADF test
#stock_p_list = []

#for stock in stock_list:
#    adf_result = sts.adfuller(stock_prices[stock],1)
#    if adf_result[1] < 0.05:
#        stock_p_list.append([stock,adf_result[1]])

#print stock_p_list #stock list which satisfies the stationarity test

#stock_prices['601727'].plot()
#plt.show()

#stock_prices['601857'].plot()
#plt.show()

#stock_prices['600000'].plot()
#plt.show()
