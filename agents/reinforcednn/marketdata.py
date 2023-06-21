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
    def __init__(self,):
        self._data_len = pd.Timedelta(weeks=4)
        self._data_gen = GBMDataGen(
            start_date=pd.Timestamp("2022-01-01"), 
            end_date=pd.Timestamp("2022-01-01") + self._data_len, 
            freq='1min', 
            initial_value=10000,
            drift=0,#drift,#0.000000003, # final value = initial_value + (1+drift*nticks*tick_length)
            volatility=4e-6,#0.00005,#0.00001, # (daily_std/initial_value)**2 * freq_as_seconds
            start_time=pd.Timestamp("09:00"),
            end_time=pd.Timestamp("17:30"),
            )
        
        self.data = self._make_new_data()
        self._counter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self._counter +=1
        done = self._counter >= len(self.data)
        if done:
            self.data = self._make_new_data()
            self._counter = 0
            
        current_index = self.data.index[self._counter]
        start_index = self.data.index[self.data.index.date == current_index.date()][0]
        return self.data.loc[start_index:current_index,:], done
                   
    @property
    def current_price(self):
        return None
    
    def _make_new_data(self,):
        self._data_gen.drift = np.random.normal(scale=3) * 3e-9
        self._data_gen.volatility = np.random.gamma(5,1) * 1e-6
        self._data_gen.initial_value = self._data['Close'].iloc[-1]
        data = self._data_gen.generate()
        self._data_gen.start_date += self._data_len
        self._data_gen.end_date += self._data_len
        return data
    
        
        
        
        
        