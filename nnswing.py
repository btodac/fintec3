#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:40:34 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

from agents.targetgenerators import TrendBasedTargetGen
#from agents.nnagent import NNAgent
from agents.bayesagent import BayesAgent
from agenttesting.results import SteeringResults
from utillities.timesanddates import get_ticker_time_zone
from utillities.datastore import Market_Data_File_Handler

save_model = False
ticker = "^NDX"
# Observation parameters
columns = [
        '2min_Mom', '4min_Mom', '8min_Mom','16min_Mom','32min_Mom','64min_Mom',
        '8min_MeanDist','16min_MeanDist',#'32min_MeanDist','64min_MeanDist',
        #'128min_MeanDist','256min_MeanDist','512min_MeanDist',
        '2min_Trend','4min_Trend','8min_Trend','16min_Trend','32min_Trend','64min_Trend',
        '128min_Trend','256min_Trend','512min_Trend',
        #'10min_Std',#'15min_Std','30min_Std',#'60min_Std','120min_Std',#'240min_Std','480min_Std',
        #'10min_Skew',#'15min_Skew','30min_Skew',#'60min_Skew','120min_Skew',#'240min_Skew','480min_Skew',
        #'10min_Kurt','20min_Kurt','40min_Kurt',
        '10min_20min_MeanDiff','20min_40min_MeanDiff',#'40min_80min_MeanDiff',
        '10min_StochOsc',
        #'10min_StochOsc',#'20min_StochOsc','40min_StochOsc',
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
    'time_limit': 30,#5,
    'live_tp': 40,
    'live_sl': 10,
    'live_tl': 30,#np.inf,
    'up' : 6,
    'down' : -6,
    'to' : 5,
    }
target_generator = TrendBasedTargetGen(
    up=params['up'], 
    down=params['down'], 
    time_limit=params['to']
    )
model = BayesAgent(
    ticker, columns, params=params, target_generator=target_generator
    )
#for i in range(10):
history = model.fit(training_data, validation_data)

predictions, probabilities, order_datetimes = model.predict(validation_data)
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


