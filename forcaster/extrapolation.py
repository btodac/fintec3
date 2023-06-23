#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:26:02 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.featurefunctions import Trend
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone

ticker = "^GDAXI"
data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-03-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

tz = get_ticker_time_zone(ticker) #'^GDAXI'
training_data = training_data.tz_convert(tz)
validation_data = validation_data.tz_convert(tz)

s = training_data['Close'].rolling('10min').std()
atr = (training_data['High'] - training_data['Low']).rolling('10min').mean()
vol = 0.5 * (s+atr)
vol.index

groups = vol.groupby(by=lambda x: x.date())
x = [vol.loc[groups.groups[g],:].to_numpy() for g in groups.groups.keys()]
y = np.concatenate(x,axis=1)
dff = pd.DataFrame(y, index = groups.groups[list(groups.groups.keys())[0]])
plt.plot(dff.to_numpy().mean(axis=1))


linreg = training_data.rolling('20min')['Close'].apply(
    lambda x: np.polynomial.polynomial.Polynomial.fit(
        np.arange(20), np.pad(x, (20-len(x),0), mode='edge'), 1
        ).coef[1], 
    raw=True
    )
t = Trend(['20min'])
trend = t(training_data)
