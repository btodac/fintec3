#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:40:34 2023

@author: mtolladay
"""

import pandas as pd

from agents.nnagent import NNAgent
from agenttesting.results import Results
from agents.targetgenerators import TrendBasedTargetGen
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone

save_model = False
ticker = "^GDAXI"
# Observation parameters
columns = [
        'Range', 'Mom',
        '2min_Mom', '4min_Mom',
        '8min_MeanDist','16min_MeanDist','32min_MeanDist','64min_MeanDist',
        '128min_MeanDist','256min_MeanDist','512min_MeanDist',
        '2min_Trend','4min_Trend','8min_Trend','16min_Trend','32min_Trend','64min_Trend',
        '128min_Trend','256min_Trend','512min_Trend',
        '10min_Std','15min_Std','30min_Std','60min_Std','120min_Std',#'240min_Std','480min_Std',
        '10min_Skew','15min_Skew','30min_Skew','60min_Skew','120min_Skew',#'240min_Skew','480min_Skew',
    ]
'''
columns = [
    '2min_mom','4min_mom',
    '2min_Trend','4min_Trend','8min_Trend','16min_Trend','32min_Trend','64min_Trend',
    '128min_Trend','256min_Trend','512min_Trend',
    '10min_Std','30min_Std',
    '10min_Skew','30min_Skew',
    ]
'''
data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-03-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

tz = get_ticker_time_zone(ticker) #'^GDAXI'
training_data = training_data.tz_convert(tz)
validation_data = validation_data.tz_convert(tz)
#training_data = training_data.between_time("09:30", '11:30')
#validation_data = validation_data.between_time("09:30", '11:30')
ndx_params = {
    'take_profit': 40,#10
    'stop_loss': 10, #10
    'time_limit': 30,#5,
    'live_tp': 50,
    'live_sl': 5,
    'live_tl': 30,#np.inf,
    'up' : 40,
    'down' : -40,
    'to' : 30,
    }
gdaxi_params = {
    'take_profit': 40,#10
    'stop_loss': 10, #10
    'time_limit': 30,#5,
    'live_tp': 40,
    'live_sl': 5,
    'live_tl': 30,#np.inf,
    'up' : 30,
    'down' : -30,
    'to' : 30,
    }
target_generator = TrendBasedTargetGen(
    up=gdaxi_params['up'], 
    down=gdaxi_params['down'], 
    time_limit=gdaxi_params['to']
    )
model = NNAgent(ticker, columns, params=gdaxi_params)
#for i in range(10):
history = model.fit(training_data, validation_data)

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


