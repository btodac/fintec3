#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:29:22 2023

@author: mtolladay
"""
import logging
import numpy as np

from agents.reinforcednn.broker import Broker
from agents.reinforcednn.marketdata import MarketDataGen

log = logging.getLogger(__name__)

DO_NOTHING_PUNISHMENT = 0.1#1.5#0.25

DATA_HISTORY = 50

class Env(object):
    def __init__(self, observer):
        log.info('Starting environment')
        self.market_data_gen = MarketDataGen(observer)
        self.broker = Broker(self.market_data_gen)
        self.input_data_size = (observer.shape[0] + DATA_HISTORY * 3,)
        self._last_action = 2
        self._profit_history = [0]
        self._action_history = [0]
        
    def step(self, action):
        score = 0

        if action != self._last_action:
            if self._last_action != 2: # Must be an open order to close
                score = self.broker.close_position()
            if action != 2: # open the new order
                self.broker.open_position(action)   
        elif action == 2:
            score = -DO_NOTHING_PUNISHMENT
        #else:
        #    score = min(0, self.broker.current_profit_loss)
                
        if self.broker.current_equity > 1000:
            score = 1000
            done = True
        elif self.broker.current_equity < -1000:
            score = -1000
            done = True
        else:
            #score = min(score,0)
            self._last_action = action
            done = False
            
        self._profit_history.append(self.broker.current_equity)# / 50)
        self._action_history.append(action)
            
        observation, reset = self.make_state()
        if reset:
            self.reset()
        
        return observation, score, done, 
    
           
    def reset(self):
        #get initial observation
        self.broker.reset()
        self._last_action = 2
        self._profit_history = [0]
        self._action_history = [2]
        observation, reset = self.make_state()
        return observation, False      
    
    def make_state(self,):
        observation, done = next(self.market_data_gen)
        
        x = self.market_data_gen.get_data_slice(DATA_HISTORY)['Open'].to_numpy()
        x -= x[-1]
        price = np.zeros(DATA_HISTORY)
        price[-len(x):] = x
        
        x = np.array(self._profit_history[-DATA_HISTORY:])
        profits = np.zeros(DATA_HISTORY)
        profits[-len(x):] = x
    
        x = np.array(self._action_history[-DATA_HISTORY:])
        actions = np.zeros(DATA_HISTORY)
        actions[-len(x):] = x
    
        state_next = np.concatenate((observation, price, actions, profits))
        return state_next, done

if __name__ == "__main__":
    from agents.preprocessing import ObservationBuilder
    observer = ObservationBuilder(['10min_Std',])
    env = Env(observer)
    state = env.make_state()
        