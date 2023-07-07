#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:40:34 2023

@author: mtolladay
"""

import pandas as pd

from agents.agent import Agent
from agenttesting.results import Results
from agents.preprocessing import ObservationBuilder
from agents.targetgenerators import TrendBasedTargetGen, VelocityBasedTargetGen
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone

save_model = True
ticker = "^NDX"
# Observation parameters
columns = [
        '10min_AvgTrueRange',
        '20min_WeightedTrend', '10min_WeightedTrend',
        #'10min_Std',
        '10min_20min_MeanDiff', '10min_120min_MeanDist',
        '15min_StochOsc',
        #'60min_MeanDist',#'120min_MeanDist','240min_MeanDist'
    ]

data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-03-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

tz = get_ticker_time_zone(ticker) #'^GDAXI'
training_data = training_data.tz_convert(tz)
validation_data = validation_data.tz_convert(tz)
training_data = training_data.between_time(
    "10:00", "16:00" # NDX
    #"09:30", '17:30' # GDAXI
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
    'live_tp': 70,
    'live_sl': 10,
    'live_tl': 60,#np.inf,
    'up' : 40,
    'down' : -40,
    'to' : 30,
    }

gdaxi_params = {
    'take_profit': 40,#10
    'stop_loss': 10, #10
    'time_limit': 30,#5,
    'live_tp': 50,
    'live_sl': 10,
    'live_tl': 60,#np.inf,
    'up' : 50,
    'down' : -50,
    'to' : 60,
    }
if ticker == "^GDAXI":
    params = gdaxi_params
elif ticker == "^NDX":
    params = ndx_params

observer = ObservationBuilder(columns, back_features=(60,1))
target_generator = TrendBasedTargetGen(params['up'], 
                                       params['down'], 
                                       params['to'],
                                       up_down_ratio=0.6)
agent = Agent(ticker, columns, model="nn", params=params, 
              target_generator=target_generator, observer=observer)
history = agent.fit(training_data, validation_data)

predictions, probabilities, order_datetimes = agent.predict(validation_data)
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
                    take_profit=agent._params['live_tp'],
                    stop_loss=agent._params['live_sl'],
                    time_limit=agent._params['live_tl'],
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
        pickle.dump(agent, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(model_dir + 'results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('##########################################################################')
    print(f'model saved: {model_name}')
    print('##########################################################################')


