#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 21:31:29 2023

@author: mtolladay
"""
import pandas as pd

from agents.naivebayes.bayesmodel import BayesAgent, BayesResults
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone


save_model = False
ticker = "^NDX"
params =  {
    'take_profit' : 60,
    'stop_loss' : 10,
    'time_limit' : 15,
    'up' : 30,
    'down' : -30,
    'to' : 8,
    'live_tl' : 8,
    }
# Observation parameters
columns = [
        '2min_mom','4min_mom',#'8min_mom','32min_mom','64min_mom','128min_mom','256min_mom',
        '8min_mean_dist','16min_mean_dist','32min_mean_dist','64min_mean_dist',
        '128min_mean_dist','256min_mean_dist','512min_mean_dist',
        #'5min_15min_mean_diff',
        #'30min_std',#'30min_std',#'30min_std',#'60min_std',#'15min_std',
        '15min_skew','30min_skew','60min_skew','120min_skew','240min_skew',
        #'20min_kurt', #'5min_kurt',
    ]
# Ideal for GDAXI
#columns = ['2min_mom','4min_mom','8min_mean_dist','16min_mean_dist','32min_mean_dist','64min_mean_dist','5min_15min_mean_diff','10min_std','10min_skew',]
#columns = ['2min_mom','32min_mom','64min_mom','10min_skew', '20min_skew', '20min_kurt']
#columns = ['5min_mean', '20min_kurt', '5min_mom', '10min_skew','20min_skew', '15min_std']
#columns = ['5min_mom','15min_mom','30min_mom','60min_mom','10min_skew','20min_skew']
# Ideal for NDX
#columns = ['2min_mom','4min_mom','8min_mom','16min_mom','32min_mean_dist','64min_mean_dist','128min_mean_dist','30min_std','10min_skew','20min_skew',]
#columns = ['2min_mom','4min_mom','8min_mom','16min_mom','32min_mom','64min_mom','10min_skew','20min_skew']
#columns = ['5min_mom','15min_mom','30min_mom','60min_mom','10min_skew','30min_skew','10min_kurt']
#columns = ['10min_mean', '20min_kurt', '5min_mom', '10min_skew','20min_skew', '15min_std']
# Ideal for EUR=X
#columns = ['2min_mom','16min_mom','32min_mom','64min_mom','10min_skew','20min_skew',]


data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2022-11-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
'''
start_time = pd.Timestamp.now().normalize() - pd.Timedelta(weeks=32)
start_time = start_time.tz_localize('UTC')
split_time = pd.Timestamp.now().normalize() - pd.Timedelta(weeks=8) #
split_time = split_time.tz_localize('UTC')
training_data = all_data.iloc[
    np.logical_and(all_data.index.to_numpy() < split_time,
                       all_data.index.to_numpy() > start_time,
                       )
    ]
'''
'''
Fed Rates: 2022-07-26, 2022-09-20, 2022-11-01, 2022-12-13, 2023-01-31, 2023-03-21, 2023-05-02
US CPI: 2022-07-14, 2022-08-10, 2022-09-13, 2022-10-13, 2022-11-10, 2022-12-13
        2023-01-12, 2023-02-12, 2023-03-14, 2023-04-12, 2023-05-10
'''
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

tz = get_ticker_time_zone(ticker) #'^GDAXI'
training_data = training_data.tz_convert(tz)
validation_data = validation_data.tz_convert(tz)
#training_data = training_data.asfreq('5S')
#validation_data = validation_data.asfreq('5S')
#training_data = training_data.between_time("09:30", '11:30')
#validation_data = validation_data.between_time("09:30", '11:30')
#training_data = training_data.between_time("11:30", '16:00')
#validation_data = validation_data.between_time("11:30", '16:00')

model = BayesAgent(ticker, columns, params=params)
model.fit(training_data, validation_data)

predictions, probabilities, order_datetimes = model.predict(validation_data, ticker)
results = BayesResults(predictions, order_datetimes, ticker, take_profit=40, stop_loss=10,
                       time_limit=10, data=validation_data,)

print(results.positions_results.to_string(float_format=lambda x : f'{x:.3f}'))
results.plot_all_signals_profits()
results.plot_position_profit()
results.plot_position_distribution()
results.plot_position_profit_distribution()
results.plot_weekly_profit()

if save_model:
    import os
    import pickle
    import string
    import random
    model_name = 'NaiveBayes_' + ticker.strip("+^") + '_' +\
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    model_dir = '/home/mtolladay/Documents/finance/NaiveBayesModels/' + model_name + '/' 
    os.makedirs(model_dir)
    filename = model_dir + 'model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('##########################################################################')
    print(f'model saved: {model_name}')
    print('##########################################################################')


