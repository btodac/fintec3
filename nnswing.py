#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:40:34 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

from agents.targetgenerators import TrendBasedTargetGen
from agents.mlp import NNAgent
from agenttesting.results import SteeringResults
from utillities.timesanddates import get_ticker_time_zone
from utillities.datastore import Market_Data_File_Handler

save_model = False
ticker = "^NDX"
# Observation parameters
columns = [
        'range', 'mom',
        '4min_high','8min_high','16min_high','32min_high','64min_high','128min_high','256min_high',
        '4min_low','8min_low','16min_low','32min_low','64min_low','128min_low','256min_low',
        '2min_mom','4min_mom',#'8min_mom','32min_mom','64min_mom','128min_mom','256min_mom',
        '8min_mean_dist','16min_mean_dist','32min_mean_dist','64min_mean_dist',
        '128min_mean_dist','256min_mean_dist','512min_mean_dist',
        '2min_trend','4min_trend','8min_trend','16min_trend','32min_trend','64min_trend',
        '10min_std','15min_std','30min_std','60min_std','120min_std',#'240min_std','480min_std',
        '10min_skew','15min_skew','30min_skew','60min_skew','120min_skew',#'240min_skew','480min_skew',
    ]

data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-01-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

tz = get_ticker_time_zone(ticker) #'^GDAXI'
training_data = training_data.tz_convert(tz)
validation_data = validation_data.tz_convert(tz)
#training_data = training_data.between_time("10:00", '15:30')
#validation_data = validation_data.between_time("09:30", '11:30')
params = {
    'take_profit': 40,#10
    'stop_loss': 10, #10
    'time_limit': 10,#5,
    'live_tp': 50,
    'live_sl': 10,
    'live_tl': np.nan,
    'up' : 40,
    'down' : 40,
    'to' : 30,
    }
target_generator = TrendBasedTargetGen(
    up=params['up'], 
    dowm=params['down'], 
    time_limit=params['to']
    )
model = NNAgent(
    ticker, columns, params=params, target_generator=target_generator
    )
#for i in range(10):
history = model.fit(training_data, validation_data)

predictions, probabilities, order_datetimes = model.predict(validation_data, ticker)
results = SteeringResults(predictions, order_datetimes, ticker,
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
    model_name = 'NN_' + ticker.strip("+^") + '_' +\
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    model_dir = '/home/mtolladay/Documents/finance/NNmodels/' + model_name + '/' 
    os.makedirs(model_dir)
    with open(model_dir + 'model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(model_dir + 'results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('##########################################################################')
    print(f'model saved: {model_name}')
    print('##########################################################################')


