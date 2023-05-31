#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:49:49 2023

@author: mtolladay
"""
import logging 

import numpy as np

from agents import Agent

log = logging.getLogger(__name__)

PARAMS = {
    'take_profit': 25,
    'stop_loss': 5,
    'time_limit': 15,
    'live_tp': 25,
    'live_sl': 5,
    'live_tl': 10,
    'to' : 10,
}

class TestAgent(Agent):
    def __init__(self, ticker: str, columns: list, params=None, 
                 observer=None, target_generator=None):
        '''
        NNAgent is a subclass of Agent that uses a multi layer 
        perceptron model to make predictions

        Parameters
        ----------
        ticker : str
            Ticker of financial product
        columns : list
            list of strings containing features
        params : dict, optional
            see Agent.
        observer : TYPE, optional
           see Agent
        target_generator : TYPE, optional
            see Agent

        Returns
        -------
        None.

        '''
        self.direction = np.array([1,0,0])
        if params is None:
            params = PARAMS
        
        super().__init__(ticker, columns, params=params, observer=observer,
                         target_generator=target_generator)
        
    def make_prediction(self, observations):
        if len(observations.shape) == 2:
            i = np.random.randint(0,3,observations.shape[0])
            y = np.zeros((observations.shape[0],3))
            y[:,i] = 1
        else:
            y = self.direction
            self.direction = np.roll(self.direction, 1)
        log.debug(f'Signal is {y}')   
        return y
        