#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:40:34 2023

@author: mtolladay
"""

import pandas as pd

from agents.nnagent import NNAgent
from agenttesting.results import Results
from agents.preprocessing import ObservationBuilder
from agents.targetgenerators import TrendBasedTargetGen, VelocityBasedTargetGen
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
        '15min_Std','30min_Std','60min_Std','120min_Std',#'240min_Std','480min_Std',
        '15min_Skew','30min_Skew','60min_Skew','120min_Skew',#'240min_Skew','480min_Skew',
    ]
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
#'''
columns = [
    '2min_High','4min_High','8min_High','16min_High','32min_High','64min_High','128min_High',
    '2min_Low','4min_Low','8min_Low','16min_Low','32min_Low','64min_Low','128min_Low',
    'Mom','2min_Mom', '4min_Mom', '8min_Mom','16min_Mom','32min_Mom','64min_Mom',
    '8min_MeanDist','16min_MeanDist','32min_MeanDist','64min_MeanDist',
    '128min_MeanDist','256min_MeanDist','512min_MeanDist',
    '2min_Trend','4min_Trend','8min_Trend','16min_Trend','32min_Trend','64min_Trend',
    '128min_Trend','256min_Trend','512min_Trend',
    'AvgTrueRange','2min_AvgTrueRange','4min_AvgTrueRange','8min_AvgTrueRange',
    '16min_AvgTrueRange','32min_AvgTrueRange',
    #'10min_Std','15min_Std','30min_Std','60min_Std','120min_Std',#'240min_Std','480min_Std',
    '8min_Skew','16min_Skew','32min_Skew','64min_Skew','128min_Skew',#'240min_Skew','480min_Skew',
    '8min_Kurt','16min_Kurt','32min_Kurt',
    '8min_16min_MeanDiff','16min_32min_MeanDiff','32min_64min_MeanDiff',
    '8min_StochOsc','16min_StochOsc','32min_StochOsc',
    ]
#'''
'''
columns = [
    'AvgTrueRange','2min_AvgTrueRange','4min_AvgTrueRange','8min_AvgTrueRange',
    '16min_AvgTrueRange','32min_AvgTrueRange',
    '2min_Trend','4min_Trend','8min_Trend','16min_Trend','32min_Trend',
    '64min_Trend','128min_Trend','256min_Trend',
    '10min_StochOsc','20min_StochOsc','40min_StochOsc',
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
training_data = training_data.between_time(
    #"10:00", "16:00" # NDX
    "09:30", '17:30' # GDAXI
    )
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
ndx_params = {
    'take_profit': 40,#10
    'stop_loss': 10, #10
    'time_limit': 20,#5,
    'live_tp': 40,
    'live_sl': 5,
    'live_tl': 10,#np.inf,
    'up' : 15,
    'down' : -15,
    'to' : 10,
    }

gdaxi_params = {
    'take_profit': 40,#10
    'stop_loss': 10, #10
    'time_limit': 30,#5,
    'live_tp': 40,
    'live_sl': 5,
    'live_tl': 10,#np.inf,
    'up' : 15,
    'down' : -15,
    'to' : 15,
    }
if ticker == "^GDAXI":
    params = gdaxi_params
elif ticker == "^NDX":
    params = ndx_params

observer = ObservationBuilder(columns, back_features=(6,5))
target_generator = TrendBasedTargetGen(params['up'], 
                                       params['down'], 
                                       params['to'],
                                       up_down_ratio=0.5)
model = NNAgent(ticker, columns, params=params, observer=observer, 
                target_generator=target_generator)
history = model.fit(training_data, validation_data)

predictions, probabilities, order_datetimes = model.predict(validation_data)
'''
p_half = probabilities>0.5
is_trade = p_half.any(axis=1)
pred_half = predictions[is_trade]
odt_half = order_datetimes[is_trade]
results = Results(pred_half, odt_half, ticker,
                    take_profit=model._params['live_tp'],
                    stop_loss=model._params['live_sl'],
                    time_limit=model._params['live_tl'],
                    data=validation_data,
                    )
'''
results = Results(predictions, order_datetimes, ticker,
                    take_profit=model._params['live_tp'],
                    stop_loss=model._params['live_sl'],
                    time_limit=model._params['live_tl'],
                    data=validation_data,
                    )
#'''
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


