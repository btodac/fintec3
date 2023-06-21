#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:29:22 2023

@author: mtolladay
"""
import logging
import numpy as np

from agents.reinforcednn.broker import Broker

log = logging.getLogger(__name__)

DO_NOTHING_PUNISHMENT = 0#1.5#0.25

class Env(object):
    def __init__(self, observer, market_data_gen):
        log.info('Starting environment')
        self._observer = observer
        self.broker = Broker(market_data_gen)
        self.input_data_size = (self._observer.shape)
        self._last_action = 0
        self._profit_history = [0]
        self._action_history = [0]
        
    def step(self, action):
        score = 0

        if action != self.last_action:
            if self.last_action != 2:
                self.broker.close_position()
            if action != 2:
                self.broker.open_position(action)
        elif action == 2:
            score = -DO_NOTHING_PUNISHMENT
                
        if self.broker.funds > 50:
            score = 500
            done = True
        elif self.broker.funds < -50:
            score = -500
            done = True
        else:
            #score = min(score,0)
            self._last_action = action
            done = False
            
        profit = self.funds
        if self.position is not None:
            profit += self.broker.current_profit_loss
            
        self._profit_history.append(profit)# / 50)
        self._action_history.append(action)
            
        market_data, reset = next(self.market_data_gen)
        if reset:
            self.reset()
            
        observation = self.observer(market_data)
        return observation, score, done, 
    
           
    def reset(self):
        #get initial observation
        self.broker.reset()
        self._last_action = 0
        self._profit_history = [0]
        self._action_history = [0]
        return None                      
        