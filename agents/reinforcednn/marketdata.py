#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:26:38 2023

@author: mtolladay
"""

import numpy as np
import pandas as pd

from agenttesting.datagenerator import GBMDataGen

class MarketDataGen(object):
    def __init__(self, observer):
        self.observer = observer
        
        start_date = pd.Timestamp("2022-01-01")
        self._data_len = pd.Timedelta(weeks=1)
        self._data_gen = GBMDataGen(
            start_date = start_date, 
            end_date = start_date + self._data_len, 
            freq = '1min', 
            start_time = pd.Timestamp("09:00"),
            end_time = pd.Timestamp("17:30"),
            )
        
        self._make_new_data()
        self._counter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        done = self._counter >= len(self.data)
        if done:
            self._make_new_data()
            self._counter = 0
            
        observation = self.observations[self._counter,:] 
        self._counter +=1
        return observation, done
    
    def get_data_slice(self, look_back):
        index = self.data.index[:self._counter]
        index = index[-look_back:]
        index = index[index.date == index[-1].date()]
            
        return self.data.loc[index,:]
                   
    @property
    def current_price(self):
        return float(self.data['Close'].iloc[self._counter-1])
    
    def _make_new_data(self,):
        self._data_gen.drift = np.random.normal(scale=3) * 1e-9
        self._data_gen.volatility = np.random.gamma(5,1) * 1e-5
        try:
            self._data_gen.initial_value = self.data['Close'].iloc[-1]
        except AttributeError:
            self._data_gen.initial_value = 13000
            
        self.data = self._data_gen.generate()
        self.observations, _  = self.observer.make_observations(
            self.data,
            self._data_gen.start_time,
            self._data_gen.end_time,
            self._data_gen.tz
            )
        self._data_gen.start_date += self._data_len
        self._data_gen.end_date += self._data_len
        
    
        
        
        
        
        