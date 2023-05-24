#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:41:50 2023

@author: mtolladay
"""
import sys
sys.path.insert(1, '/home/mtolladay/jobfiles/PyProjects/fintec3')
import matplotlib.pyplot as plt

import pandas as pd

from utillities.datastore import Market_Data_File_Handler
from agents.naivebayes.interdaybayesmodel import InterdayBayesModel, interday_results

save_model = False
ticker = "^NDX"
# Observation parameters
columns = [
        '60min_mom',#'8min_mom',#'16min_mom',#'32min_mom',#'60min_mom',
        '60min_mean',#'10min_mean',#'20min_mean','40min_mean',
        #'15min_30min_mean_diff',# '10min_20min_mean_diff',
        '60min_std',#'60min_std',#'15min_std',
        '60min_skew',#'30min_skew',#'15min_skew',
        '60min_kurt', #'30min_kurt',
    ]

split_time = pd.Timestamp("2023-02-01", tz='UTC')
data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
training_data = all_data.iloc[all_data.index.to_numpy() < split_time]
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]
# Get stats of previous day: for each hour - mom, mean diff,
# Get opening value of next day
model = InterdayBayesModel(training_data, ticker, columns)

profits = interday_results(model, validation_data)

print(sum(profits>0)/sum(profits!=0))

plt.plot(profits.cumsum())