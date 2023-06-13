#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:38:28 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

from agents.featurefunctions import Trend
from agenttesting.results import SteeringResults
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone

ticker = "^GDAXI"
split_time = pd.Timestamp("2023-03-01", tz='UTC')

data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]
validation_data.columns = validation_data.columns.droplevel(1)
tz = get_ticker_time_zone(ticker) 
validation_data = validation_data.tz_convert(tz)

trend = Trend(['30min'])
t = trend(validation_data)

signals = 2 * np.ones(len(t))
signals[t > 2] = 0
signals[t < -2] = 1

is_start = validation_data.index.time < pd.Timestamp("09:30",tz=validation_data.index.tz).time()
is_end = validation_data.index.time == max(validation_data.index.time)
signals[is_start] = 2
signals[is_end] = 2
results = SteeringResults(signals, validation_data.index, ticker, np.inf, 20, np.inf, 
                          data=validation_data)
print(results.positions_results.to_string(float_format=lambda x : f'{x:.3f}'))
results.plot_daily_profit()