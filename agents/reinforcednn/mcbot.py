#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:09:55 2023

@author: mtolladay
"""
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
if not hasattr(__builtins__,'__IPYTHON__'):
    matplotlib.use('Agg')
import pandas as pd


sys.path.insert(1, '/home/mtolladay/jobfiles/PyProjects/fintec2')
from datahandling.datastore import Market_Data_File_Handler

ticker = "EUR=X"
columns = ['Close','2min_mean','4min_mean','8min_mean','16min_mean','32min_mean','64min_mean',
           '128min_mean','256min_mean','512min_mean']
num_actions = 3
observation_depth = 1

data_file = Market_Data_File_Handler(dataset_name="training2")
all_data = data_file.get_ticker_data(ticker, as_list=False)

# Preprocess the historical data to get the static observation space
def df_to_numpy(data, ticker, columns):
    
    for c in columns:
        if 'mean' in c:
            t_string = c.split('_')[0]
            data.loc[:,c] = data.loc[:,'Close'].rolling(t_string).mean().to_numpy()
            
    data.columns = data.columns.droplevel(1)
    d = data.loc[:,columns]
    series = [
        d['Close'].diff() > 0,
        (d['Close'] - d['2min_mean']) > 0,
        (d['2min_mean'] - d['4min_mean']) > 0,
        (d['4min_mean'] - d['8min_mean']) > 0,
        (d['8min_mean'] - d['16min_mean']) > 0,
        (d['16min_mean'] - d['32min_mean']) > 0,
        (d['32min_mean'] - d['64min_mean']) > 0,
        (d['64min_mean'] - d['128min_mean']) > 0,
        (d['128min_mean'] - d['256min_mean']) > 0,
        (d['256min_mean'] - d['512min_mean']) > 0,
    ]
    return pd.concat(series, axis=1).to_numpy()

class Position(object):
    
    def __init__(self, opening_price, direction):
        self.opening_price = opening_price
        self.direction = direction
        
    def current_profit_loss(self, price):
        return self.direction * (price - self.opening_price) * 1e4 #- 0.2 
    
observations = df_to_numpy(all_data, ticker, columns)

q_table = np.zeros((2 ** 16, num_actions))#((observations.shape[1] + 4)), num_actions))
max_frames = 4e6
epsilon_random_frames = 2e5
epsilon_greedy_frames = 1e6
epsilon = 1
epsilon_min = 0.05
epsilon_decay_factor = (epsilon - epsilon_min) / epsilon_greedy_frames
alpha = 0.15
gamma = 0.99
last_action = 0
funds = 0
i = 0
observation_index = 0
position = None
non_obs_env = np.zeros((observation_depth, 4), dtype=bool)
state = np.concatenate((observations[i,:][np.newaxis,:], non_obs_env),axis=1)
state_index = np.packbits(state).view(np.uint16)
while i < max_frames:
    
    # epsilon greedy action selection
    if i < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
        action = np.random.choice(num_actions)
    else:
        action = np.argmax(q_table[state_index])
    
    if i > epsilon_random_frames and i < epsilon_greedy_frames:
        epsilon -= epsilon_decay_factor
    
    # get reward    
    current_price = all_data['Close'].iloc[observation_index]
    profit = 0
    score = 0
    if action == 0: # Hold
        if last_action == 0: #hold -> hold
            score = -0.5 # Do nothing punishment
        elif last_action == 1: #  buy -> hold
            score = position.current_profit_loss(current_price)
            position = Position(current_price, 1)
            profit = score
        elif last_action == 2: # sell -> hold
            score = position.current_profit_loss(current_price)
            position = Position(current_price, -1)
            profit = score
    elif action == 1: # Buy
        if last_action == 0: # hold -> buy
            position = Position(current_price, 1)
        elif last_action == 2:# sell -> buy
            score = position.current_profit_loss(current_price)
            position = Position(current_price, 1)
            profit = score
        else:
            score = position.current_profit_loss(current_price)
    elif action == 2: # Sell
        if last_action == 0: # hold -> sell
            position = Position(current_price, -1)
        elif last_action == 1:# buy -> sell
            score = position.current_profit_loss(current_price)
            position = Position(current_price, -1)
            profit = score
        else:
            score = position.current_profit_loss(current_price)
            
            
    reward = 2 * ((score > 0) - 0.5)
       
    non_obs_env = np.zeros((1,4), dtype=bool)
    non_obs_env[0,action] = True
    non_obs_env[0,3] = reward > 0
    
    i += 1
    observation_index = i % (len(observations)-1)
    
    state_next = np.concatenate((observations[observation_index,:][np.newaxis,:], non_obs_env),axis=1)
    state_next_index = np.packbits(state_next).view(np.uint16)
    # update table ()
    q = q_table[state_index, action]
    q_next = q_table[state_next_index].squeeze()
    q_table[state_index, action] += alpha * (reward + gamma * max(q_next) - q )
    state = state_next[:]
    state_index = state_next_index[:]
    last_action = action
    funds += profit
    print(f'Step: {i:8d}, action: {action:1d}, reward: {reward:2.0f}, score: {funds:8.1f}')
####################################################################################################
                             
data_file = Market_Data_File_Handler(dataset_name="validation2")
all_data = data_file.get_ticker_data(ticker, as_list=False)   
observations = df_to_numpy(all_data, ticker, columns)                   

last_action = 0
funds = 0
i = 0
position = None
non_obs_env = np.zeros((observation_depth, 4), dtype=bool)
while i < (len(observations) - 1):
    state = np.concatenate((observations[i,:][np.newaxis,:], non_obs_env),axis=1)
    state_index = np.packbits(state).view(np.uint16)
    
    action = np.argmax(q_table[state_index])
    current_price = all_data['Close'].iloc[i]
    use_score = False
    if action == 0: # Hold
        if last_action == 0: #hold -> hold
            profit = -0.5 # Do nothing punishment
        elif last_action == 1: #  buy -> hold
            score = position.current_profit_loss(current_price)
            position = Position(current_price, 1)
            funds += profit
        elif last_action == 2: # sell -> hold
            profit = position.current_profit_loss(current_price)
            position = Position(current_price, -1)
            funds += profit
    elif action == 1: # Buy
        if last_action == 0: # hold -> buy
            position = Position(current_price, 1)
        elif last_action == 2:# sell -> buy
            profit = position.current_profit_loss(current_price)
            position = Position(current_price, 1)
            funds += profit
        else:
            profit = position.current_profit_loss(current_price)
    elif action == 2: # Sell
        if last_action == 0: # hold -> sell
            position = Position(current_price, -1)
        elif last_action == 1:# buy -> sell
            profit = position.current_profit_loss(current_price)
            position = Position(current_price, -1)
            funds += profit
        else:
            profit = position.current_profit_loss(current_price)
            
            
    reward = 2 * ((profit > 0) - 0.5)
        
    
    print(f'Step: {i:10d}, action: {action:3d}, score: {funds}')
    i += 1
    last_action = action
    
    non_obs_env = np.zeros((1,4), dtype=bool)
    non_obs_env[0,action] = True
    non_obs_env[0,3] = reward > 0
    