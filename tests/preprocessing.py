#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 21:31:29 2023

@author: mtolladay
"""
import pandas as pd

from agents.preprocessing import ObservationBuilder
#from results.results import Results
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import opening_and_closing_times

ticker = "^NDX"
# Observation parameters
columns = [
    'Range',
    'Mom','2min_Mom','4min_Mom',#'8min_mom',#'32min_mom','64min_mom','128min_mom','256min_mom',
    '4min_MeanDist',#'8min_mean_dist',
    '3min_High',
    '3min_Low',
    '5min_Mean',
    #'16min_mean_diff',
    '2min_Trend',#'4min_trend','8min_trend','16min_trend','32min_trend','64min_trend',
    #'128min_trend',#'256min_trend','512min_trend',
    '5min_15min_MeanDiff',
    '30min_Std',#'8min_std','16min_std','32min_std',#'min_std',#'15min_std',
    '10min_Skew',#'16min_skew','32min_skew',#'120min_skew','240min_skew',
    '20min_Kurt', #'5min_kurt',
    ]
data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-03-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
opening_time, closing_time = opening_and_closing_times(ticker)
observer = ObservationBuilder(columns, opening_time, closing_time, 'UTC')
observation = observer.make_observations(training_data)



