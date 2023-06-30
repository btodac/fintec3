#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:26:02 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.featurefunctions import Trend, WeightedTrend
from agenttesting.results import SteeringResults
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone

ticker = "^GDAXI"
timeframe = 15 #mins
data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-06-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

tz = get_ticker_time_zone(ticker) #'^GDAXI'
training_data = training_data.tz_convert(tz)
validation_data = validation_data.tz_convert(tz)


t_string = f"{timeframe}min"
s = training_data['Close'].rolling(t_string).std()
atr = (training_data['High'] - training_data['Low']).rolling('10min').mean()
vol = 0.5 * (s+atr)
vol.index

groups = vol.groupby(by=lambda x: x.date())
x = [vol.loc[groups.groups[g],:].to_numpy() for g in groups.groups.keys()]
y = np.concatenate(x,axis=1)
dff = pd.DataFrame(y, index = groups.groups[list(groups.groups.keys())[0]])
#plt.plot(dff.to_numpy().mean(axis=1))


linreg = training_data.rolling(t_string)['Close'].apply(
    lambda x: np.polynomial.polynomial.Polynomial.fit(
        np.arange(timeframe), np.pad(x, (timeframe-len(x),0), mode='edge'), 1
        ).coef[1], 
    raw=True
    )
t = Trend([t_string])
trend = t(training_data)

wt = WeightedTrend([t_string])
weighted_trend = wt(training_data)
orders = []
for date in np.unique(weighted_trend.index.date):
    index = training_data.index.date == date

    open_trade = np.sign(weighted_trend.iloc[index])
    open_trade = (open_trade.to_numpy()[1:] - open_trade.to_numpy()[:-1]).squeeze()
    
    
    trend_grad = np.diff(weighted_trend.iloc[index].to_numpy().squeeze())
    tgs = np.sign(trend_grad)
    close_trade = tgs[1:] - tgs[:-1]
    
    opening_datetime = training_data.iloc[index].index[np.where(open_trade==2)[0] + 1]
    opening_datetime = pd.DatetimeIndex(
        opening_datetime.to_numpy()[
            np.logical_not(
                pd.DatetimeIndex(opening_datetime) == training_data.iloc[index].index.max()
                )
            ]
        )
    closing_datetimes = training_data.iloc[index].index[np.where(close_trade==-2)[0]+2]
    closing_datetimes = closing_datetimes.to_list()
    closing_datetimes.append( training_data.iloc[index].index.max() )
    closing_datetimes = pd.DatetimeIndex(np.unique(closing_datetimes))
    buys = pd.DataFrame()
    buys['Opening_datetime'] = opening_datetime
    buys['Closing_datetime'] = [closing_datetimes[np.where(closing_datetimes > odt)[0][0]] \
                                 for odt in buys['Opening_datetime']]
    buys['Opening_price'] = training_data.loc[buys.Opening_datetime,'Close'].to_numpy()
    buys['Closing_price'] = training_data.loc[buys.Closing_datetime,'Close'].to_numpy()
    buys['Profit'] = buys.Closing_price - buys.Opening_price #- 1.2
    
    opening_datetime = training_data.iloc[index].index[np.where(open_trade==-2)[0] + 1]
    opening_datetime = pd.DatetimeIndex(
        opening_datetime.to_numpy()[
            np.logical_not(
                pd.DatetimeIndex(opening_datetime) == training_data.iloc[index].index.max()
                )
            ]
        )
    closing_datetimes = training_data.iloc[index].index[np.where(close_trade==2)[0]+2]
    closing_datetimes = closing_datetimes.to_list()
    closing_datetimes.append( training_data.iloc[index].index.max() )
    closing_datetimes = pd.DatetimeIndex(np.unique(closing_datetimes))
    sells = pd.DataFrame()
    sells['Opening_datetime'] = opening_datetime
    sells['Closing_datetime'] = [closing_datetimes[np.where(closing_datetimes > odt)[0][0]] \
                                 for odt in sells['Opening_datetime']]
    sells['Opening_price'] = training_data.loc[sells.Opening_datetime,'Close'].to_numpy()
    sells['Closing_price'] = training_data.loc[sells.Closing_datetime,'Close'].to_numpy()
    sells['Profit'] = sells.Opening_price - sells.Closing_price #- 1.2
    
    buys['Direction'] = 'BUY'
    sells['Direction'] = 'SELL'
    orders.append(buys)
    orders.append(sells)
    
orders = pd.concat(orders)
orders.index = orders.Opening_datetime
orders = orders.sort_index()

for index, order in orders.iterrows():
    start = index + pd.Timedelta(minutes=1)
    orders.loc[index,'Min'] = (training_data.Low[start : order.Closing_datetime]).to_numpy().min()
    orders.loc[index,'Max'] = (training_data.High[start : order.Closing_datetime]).to_numpy().max()

orders.Profit -= 1.2


plt.plot(orders.Profit.cumsum())

t = np.arange(9,18,0.5)
indx = np.digitize([c.hour + c.minute/60 for c in orders.index.time], bins=t)
p = np.zeros(len(t))
for i in np.unique(indx):
    p[i-1] = np.mean(orders.Profit.iloc[indx == i])
    _ = plt.bar(t+0.25, p, width=0.5,
            color=[['red', 'blue'][i] for i in (p > 0)])
#plt.plot(linreg.iloc[:500])
#plt.plot(trend.iloc[:500])
'''
s1 = trend#.iloc[:511]
buy = s1.to_numpy() > 0.75
sell = s1.to_numpy() < -0.75
hold = np.logical_not(np.logical_or(buy, sell))
a = np.zeros(shape=buy.shape, dtype=int)
a[sell] = 1
a[hold] = 2
s = SteeringResults(a.squeeze(), training_data.index, 
                    "^GDAXI", 100,10,500, training_data)
s.position_outcomes['Profit'].sum()
'''