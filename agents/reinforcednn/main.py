#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:07:30 2023

@author: mtolladay
"""
import sys
import string
import random
import logging
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
if not hasattr(__builtins__,'__IPYTHON__'):
    matplotlib.use('Agg')
'''
sys.path.insert(1, '/home/mtolladay/jobfiles/PyProjects/fintec3')

from agents.preprocessing import ObservationBuilder
from agents.reinforcednn.rlagent import RLAgent
from agents.reinforcednn.trainer import TrainingParams, Trainer
from agents.reinforcednn.environment import Env
#from utillities.datastore import Market_Data_File_Handler
#from traderbot.tester import Tester

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

restart = False

if restart:
    with open('finish.txt', 'r') as fh:
        save_file = fh.read()
else:
    save_file = '/home/mtolladay/Documents/finance/traderbots/traderbot_' +\
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        
    with open('finish.txt', 'w') as fh:
        fh.write(save_file)

columns = [
    '2min_Mom','4min_Mom',
    '2min_Trend','4min_Trend','8min_Trend','16min_Trend','32min_Trend','64min_Trend',
    #'128min_Trend','256min_Trend','512min_Trend',
    #'10min_Std','30min_Std',
    #'10min_20min_MeanDiff','20min_40min_MeanDiff',
    '10min_StochOsc','20min_StochOsc',
    ]

if not restart:    
    ticker = "^GDAXI"   
    params = TrainingParams()
    observer = ObservationBuilder(columns)
    env = Env(observer=observer)
    rl_agent = RLAgent(env.input_data_size)
    trainer = Trainer(rl_agent=rl_agent, env=env, model_dir=save_file,
                 training_params = params, restart=False)
else:
    trainer = Trainer(restart=True, model_dir=save_file)

trainer.run()
####################################################################################################
'''
print("Training data results")
ticker = trainer.observer.ticker
data_file = Market_Data_File_Handler(dataset_name="training2")
all_data = data_file.get_ticker_data(ticker, as_list=False)
observer = Observer(data=all_data, ticker=ticker,
                    observation_length=trainer.observer.observation_length)
env = Env(observer=observer)
tester = Tester(env, trainer.model)
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

plt.savefig(save_file + '/results_train.png', dpi=300)
####################################################################################################
print("Validation Data results")
data_file = Market_Data_File_Handler(dataset_name="validation2")
all_data = data_file.get_ticker_data(ticker, as_list=False)

observer = Observer(data=all_data, ticker=ticker,
                    observation_length=trainer.observer.observation_length)
env = Env(observer=observer)
tester = Tester(env, trainer.model)
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

plt.savefig(save_file + '/results_test.png', dpi=300)
####################################################################################################
'''
exit(101)

         






