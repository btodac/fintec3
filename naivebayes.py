#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 21:31:29 2023

@author: mtolladay
"""
import pandas as pd

from agents.bayesagent import BayesAgent
from agents.targetgenerators import TrendBasedTargetGen
from agenttesting.results import Results
#from results.results import Results
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone

save_model = False
ticker = "^NDX"
params = {
    'take_profit': 50,#10
    'stop_loss': 10, #10
    'time_limit': 30,#5,
    'live_tp': 50,
    'live_sl': 5,
    'live_tl': 30,#np.inf,
    'up' : 40,
    'down' : -40,
    'to' : 30,
    }
# Observation parameters
columns = [
    'range', 'mom',
    '2min_mom','4min_mom',#'8min_mom',#'32min_mom','64min_mom','128min_mom','256min_mom',
    #'4min_mean_dist','8min_mean_dist',#'8min_mean_dist','16min_mean_dist','32min_mean_dist','64min_mean_dist',
    #'128min_mean_dist','256min_mean_dist','512min_mean_dist',
    '2min_trend','4min_trend','8min_trend','16min_trend','32min_trend','64min_trend',
    '128min_trend','256min_trend','512min_trend',
    #'5min_15min_mean_diff',
    '30min_std',#'8min_std','16min_std','32min_std',#'min_std',#'15min_std',
    '10min_skew',#'16min_skew','32min_skew',#'120min_skew','240min_skew',
    #'20min_kurt', #'5min_kurt',
    ]
data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
#'''
split_time = pd.Timestamp("2023-03-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
'''
start_time = pd.Timestamp.now().normalize() - pd.Timedelta(weeks=40)
start_time = start_time.tz_localize('UTC')
split_time = pd.Timestamp.now().normalize() - pd.Timedelta(weeks=8) #
split_time = split_time.tz_localize('UTC')
import numpy as np
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
target_generator = TrendBasedTargetGen(model._params['up'], 
                                       model._params['down'], 
                                       model._params['time_limit'])
model.target_generator = target_generator
model.fit(training_data, validation_data)

predictions, probabilities, order_datetimes = model.predict(validation_data)
results = Results(predictions, order_datetimes, ticker,
                    take_profit=model._params['live_tp'],
                    stop_loss=model._params['live_sl'],
                    time_limit=model._params['live_tl'],
                    data=validation_data,
                    )

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
    model_name = 'NB_' + ticker.strip("+^") + '_' +\
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    model_dir = '/home/mtolladay/Documents/finance/NBmodels/' + model_name + '/' 
    os.makedirs(model_dir)
    with open(model_dir + 'model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(model_dir + 'results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('##########################################################################')
    print(f'model saved: {model_name}')
    print('##########################################################################')


