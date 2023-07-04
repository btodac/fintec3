#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:43:29 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.featurefunctions import Trend, WeightedTrend
from agenttesting.results import SteeringResults
from agenttesting.summary import summarise_outcomes
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone

ticker = "^GDAXI"

data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-07-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

tz = get_ticker_time_zone(ticker) #'^GDAXI'
training_data = training_data.tz_convert(tz)
validation_data = validation_data.tz_convert(tz)

results = []
for timeframe in range(15,30): #mins
    t_string = f"{timeframe}min"
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
        buy_datetimes = training_data.iloc[index].index[np.where(close_trade==2)[0]+2]
        buy_datetimes = buy_datetimes.to_list()
        buy_datetimes.append( training_data.iloc[index].index.max() )
        buy_datetimes = pd.DatetimeIndex(np.unique(buy_datetimes))
        sell_datetimes = training_data.iloc[index].index[np.where(close_trade==-2)[0]+2]
        sell_datetimes = sell_datetimes.to_list()
        sell_datetimes.append( training_data.iloc[index].index.max() )
        sell_datetimes = pd.DatetimeIndex(np.unique(sell_datetimes))
        
        buys = pd.DataFrame()
        buys['Opening_datetime'] = buy_datetimes[:-1]
        buys['Closing_datetime'] = [sell_datetimes[np.where(sell_datetimes > odt)[0][0]] \
                                     for odt in buys['Opening_datetime']]
        buys['Opening_price'] = training_data.loc[buys.Opening_datetime,'Close'].to_numpy()
        buys['Closing_price'] = training_data.loc[buys.Closing_datetime,'Close'].to_numpy()
        buys['Profit'] = buys.Closing_price - buys.Opening_price #- 1.2
        
        sells = pd.DataFrame()
        sells['Opening_datetime'] = sell_datetimes[:-1]
        sells['Closing_datetime'] = [buy_datetimes[np.where(buy_datetimes > odt)[0][0]] \
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
    
    stop_loss = 10
    
    for index, order in orders.iterrows():
        start = index + pd.Timedelta(minutes=1)
        d = training_data.loc[start : order.Closing_datetime:]
        orders.loc[index,'Min'] = d.Low.to_numpy().min()
        orders.loc[index,'Max'] = d.High.to_numpy().max()
        min_first = d.Low.to_numpy().argmin() <= d.High.to_numpy().argmax()
        if (min_first and order.Direction == 'BUY' and \
                order.Opening_price - orders.loc[index,'Min'] > stop_loss)\
            or (not min_first and order.Direction == 'SELL' and \
                orders.loc[index,'Max'] - order.Opening_price > stop_loss):
            orders.loc[index,'Profit'] = -stop_loss
    
    
    orders.loc[orders.index[(orders.Profit < -stop_loss).to_numpy()],'Profit'] = -stop_loss
    
    orders.Profit -= 1.2
    
    summary = summarise_outcomes(orders)
    results.append(summary)
    
    fig, ax = plt.subplots()
    ax.set_title(f"Timeframe {t_string}")
    plt.plot(orders.Profit.cumsum())
    fig, ax = plt.subplots()
    ax.set_title(f"Timeframe {t_string}")
    t = np.arange(9,18,0.5)
    indx = np.digitize([c.hour + c.minute/60 for c in orders.index.time], bins=t)
    p = np.zeros(len(t))
    for i in np.unique(indx):
        p[i-1] = np.mean(orders.Profit.iloc[indx == i])
        _ = plt.bar(t+0.25, p, width=0.5,
                color=[['red', 'blue'][int(i)] for i in (p > 0)])
        
results = pd.DataFrame(results)
results = results.set_index('T')
results = results.sort_index()
print(results.to_string())