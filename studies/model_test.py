#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:22:57 2023

@author: mtolladay
"""
import pickle

import pandas as pd

from agenttesting.results import Results
#from results.results import Results
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone


filename = '/home/mtolladay/Documents/finance/NNmodels/NN_GDAXI_0SZ2UM/model.pkl'
#filename = '/home/mtolladay/Documents/finance/NNmodels/NN_NDX_60QS7X/model.pkl'
with open(filename,'rb') as f:
    model = pickle.load(f)

ticker = model._params['ticker']

data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-03-01", tz='UTC')
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]
tz = get_ticker_time_zone(ticker) #'^GDAXI'
validation_data = validation_data.tz_convert(tz)

predictions, probabilities, order_datetimes = model.predict(validation_data)

results = Results(predictions, order_datetimes, ticker,
                    take_profit=model._params['live_tp'],
                    stop_loss=model._params['live_sl'],
                    time_limit=model._params['live_tl'],
                    data=validation_data,
                    )
r = results.positions_results.loc[:,['N_Trades',
                                     'Win Ratio',
                                     'Profit',
                                     'Profit (week)',
                                     'Avg. Profit',
                                     'Profit Factor']]
print(r.to_string(float_format=lambda x : f'{x:.3f}'))
results.plot_all_signals_profits()
results.plot_position_profit()
results.plot_position_distribution()
results.plot_position_profit_distribution()
results.plot_weekly_profit()

