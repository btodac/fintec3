#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:54:49 2023

@author: mtolladay
"""
import sys
import logging
#import traceback

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

sys.path.insert(1, '/home/mtolladay/jobfiles/PyProjects/fintec2')
from datahandling.datastore import Market_Data_File_Handler
from traderbot.observer import Observer
from traderbot.tester import Tester
from traderbot.trainer import TrainingParams, Trainer
from traderbot.environment import Env

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

save_file = '/home/mtolladay/Documents/finance/traderbots/traderbot_4IMAA5'
ticker = "EUR=X"

data_file = Market_Data_File_Handler(dataset_name="training2")
#data_file = Market_Data_File_Handler(dataset_name="validation2")
all_data = data_file.get_ticker_data(ticker, as_list=False)

model = load_model(save_file)
observer = Observer(data=all_data, ticker = ticker, observation_length=60)
env = Env(observer=observer)
tester = Tester(env, model)
profit_loss, trade_history, funds_history = tester.test()
print(profit_loss)

history = np.array(trade_history)
funds = np.array(funds_history)
order_times = np.where(history != 0)[0]
order_lengths = order_times[1:] - order_times[:-1]
print(f'Number of orders: {len(order_times)}')
print(f'Order profit: {history[history!=0].mean()} +/- {history[history!=0].std()}')
print(f'Order times: {order_lengths.mean()} +/- {order_lengths.std()}')
fig, ax = plt.subplots()
ax.set_title(f'Mean profit/loss: {history[history!=0].mean()} +/- {history[history!=0].std()}')
plt.plot(funds, label="Total funds")
plt.plot(history.cumsum(),label="Trade history")
